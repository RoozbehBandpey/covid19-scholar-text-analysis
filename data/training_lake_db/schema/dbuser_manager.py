import os
import sys
import json
import logging
import string
import secrets
import urllib
from typing import List
from sqlalchemy import create_engine, inspect
from azureml.core import Workspace, Experiment, Run, Keyvault
from azureml.core.authentication import AzureCliAuthentication, MsiAuthentication
from azureml.core.run import _OfflineRun


_logger = logging.getLogger(__name__)


class DBUserManager():
	"""
   	DB user manager for creating user with specified rights and push user secrets into database
	"""

	def __init__(self, db_name=None, db_server=None, driver="{ODBC Driver 17 for SQL Server}", username=None, password=None):
		
		if db_name != None and db_server != None and username != None and password != None:
			_db_name = db_name
			_db_server = db_server
			_username = username
			_password = password
		else:
			_db_name = os.environ.get('sqlDBName')
			_db_server = os.environ.get('sqlServer')
			_username = os.environ.get('sqlServerAdminUsername')
			_password = os.environ.get('sqlServerAdminPass')

		_driver = driver
		# Should connect to master db for user creation
		_master_db_name = 'master'
		self._master_connection_string = f"Driver={_driver};Server=tcp:{_db_server},1433;Database={_master_db_name};Uid={_username};Pwd={_password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
		self._connection_string = f"Driver={_driver};Server=tcp:{_db_server},1433;Database={_db_name};Uid={_username};Pwd={_password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"

		# Connect
		master_params = urllib.parse.quote_plus(self._master_connection_string)
		params = urllib.parse.quote_plus(self._connection_string)
		self.master_db_engine = create_engine(
			"mssql+pyodbc:///?odbc_connect=%s" % master_params)
		self.db_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

		try:
			_logger.info(
				f"Connecting to metrics db {_master_db_name} (Server {_db_server}, user {_username})")
			self.master_conn = self.master_db_engine.connect()
			_logger.info(f'\tConnection: {self.master_conn}')
			_logger.info(
				f"Connecting to metrics db {_db_name} (Server {_db_server}, user {_username})")
			self.conn = self.db_engine.connect()
			_logger.info(f'\tConnection: {self.conn}')
		except Exception as e:
			_logger.error(f'\tError! {e}')
			raise e


	def get_login_detail(self, username: str):
		logins = self.master_db_engine.execute(
			"SELECT * FROM sys.sql_logins WHERE name = ?", username).fetchall()
		return logins

	def get_user(self, user: str):
		users = self.master_db_engine.execute(
			"SELECT * FROM sys.sysusers WHERE name = ?", user).fetchall()
		return users

	def add_user(self, login: str, user: str, schema: str, permission_sql: List[str]):

		logins = self.get_login_detail(login)
		password = self.generate_password()

		if len(logins) == 0:
			_logger.info(f"Creating login {login}.")
			self.master_db_engine.execute(
				f"CREATE LOGIN {login} WITH PASSWORD = '{password}'")

		users = self.get_user(user)

		if len(users) == 0:
			_logger.info(f"Creating user {user} with default schema {schema}")
			self.db_engine.execute(
				f"CREATE USER {user} FOR LOGIN {login} WITH DEFAULT_SCHEMA = {schema}")

		_logger.info(f"Granting/denying permissions for user {user}")
		for permission in permission_sql:
			_logger.info(permission)
			self.db_engine.execute(f"{permission} TO {user}")

		return login, password

	def generate_password(self, length=20):
		return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))


class SecretManager():
	"""
   	Secret manager for keyvaults in AML workspaces 
	"""

	def __init__(self, ws_config_file: str, subscription_id: str, aml_resource_group: str, aml_workspace_name: str):
		"""Initializes an AML workspace object"""
		auth = AzureCliAuthentication()
		if ws_config_file != None:
			self.ws = Workspace.from_config(
				auth=auth,
				path=os.path.join(
					os.path.dirname(os.path.realpath(__file__)),
					ws_config_file
				)
			)
		else:
			self.ws = Workspace.get(aml_workspace_name, auth,
			                        subscription_id, aml_resource_group)

		_logger.info(f"Found workspace {self.ws.name}")

	def get_keyvault(self):
		_logger.info(f"Getting keyvault to store new login data")
		self._keyvault = self.ws.get_default_keyvault()

	def get_secret(self, secret_name):
		if self._keyvault is None:
			self.get_keyvault()
		try:
			_logger.info(f"Getting secret of {secret_name}")
			secret_val = self._keyvault.get_secret(secret_name)
		except Exception as e:
			_logger.error(
				f"[ERROR] Secret with name {secret_name} does not exist in the target keyvault\n\t{e}")
			secret_val = None

		return secret_val

	def set_secret(self, secret_name, secret_value):
		if self._keyvault is None:
			self.get_keyvault()
		_logger.info(f"Setting secret for {secret_name}")
		self._keyvault.set_secret(secret_name, secret_value)

	def drop_secret(self, secret_name):
		if self.get_secret(secret_name):
			self._keyvault.delete_secret(secret_name)


def run_for_all_workspaces():
	workspace_config_files = [
            'atpdeep-cv1-aml-prd-config.json',
            'atpdeep-blockage-aml-prd-config.json',
            'atpdeep-cv2-aml-prd-config.json',
            'atpdeep-lidar-aml-prd-config.json',
            'atpdeep-monitoring-aml-prd-config.json',
            'atpdeep-radar-aml-prd-config.json',
            'atpdeep-tl-aml-prd-config.json',
            'atpdeep-ts-aml-prd-config.json',
            'atpdeep-unittests-aml-prd-config.json',
            'atpdeep-vru-aml-prd-config.json'
        ]

	dbmanager = DBUserManager()

	username, password = dbmanager.add_user('mdbwriter', 'mdbwriter', 'dbo',
                                         ['GRANT SELECT ON SCHEMA::dbo',
                                          'GRANT INSERT ON SCHEMA::dbo'])


	for config_file in workspace_config_files:
		secretmanager = SecretManager(config_file, None, None, None)
		workspace = secretmanager.ws
		secretmanager.get_keyvault()

		# Delete secret won't work due to permission!
		#TODO: test it on your own subscription
		# secretmanager.drop_secret('metricsdb-dataadmin')
		secretmanager.set_secret(f"metricsdb-{username}", password)
		_logger.info(f"Adding login password as "
                    f"secret metricsdb-{username} to keyvaults of workspace {workspace.name}")


if __name__ == "__main__":
	"""
	Runs over all AML workspaces in offline manner
		- Gets keyvault
		- Creates user for metrics db with writer right
		- Sets metrics db sql writer secrets
	"""

	run_for_all_workspaces()
