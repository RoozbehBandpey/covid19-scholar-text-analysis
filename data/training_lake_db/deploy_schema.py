from sqlalchemy import Integer, String
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import inspect
from sqlalchemy import Column
from sqlalchemy import select
from dbuser_manager import DBUserManager, SecretManager
from sqlalchemy import Table
from pprint import pprint
from metricsdb import Detail, Experiment, Image, Metric, Run, Tag, Workspace, Model
import argparse
import logging
import urllib
import os
import sys
import json
import logging

_logger = logging.getLogger(__name__)


# # For local run
# sys.path.append(os.getcwd())
# from utils.load_env import append_vars

# append_vars(os.path.join(os.path.dirname(os.path.dirname(
# 	os.path.realpath(__file__))), 'local.settings.json'))


class SchemaManager():
	"""
   	DB schema manager for 
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

		self._connection_string = f"""Driver={_driver};Server=tcp:{_db_server},1433;Database={_db_name};Uid={_username};Pwd={_password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"""
		# Connect
		params = urllib.parse.quote_plus(self._connection_string)
		self.db_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
		try:
			_logger.info(f"Connecting to metrics db {_db_name} (Server {_db_server}, user {_username})")
			self.conn = self.db_engine.connect()
			_logger.info(f'\tConnection: {self.conn}')
		except Exception as e:
			_logger.error(f'\tError! {e}')
			raise e

		self.metadata = MetaData()
		#Reflecting database metadata
		self.metadata.reflect(bind=self.db_engine)


	def drop_table(self, table_name):
		"""Drops table given the name"""
		table = self.metadata.tables.get(table_name)
		if table != None:	
			table.drop(self.db_engine)
			_logger.info(f'Table! {table_name} is dropped!')
			print(f'Table! {table_name} is dropped!')
		else:
			_logger.error(f'Table! {table_name} does not exist!')
			print(f'Table! {table_name} does not exist!')


	def table_exists(self, table_name):
		if table_name in self.metadata.tables:
			return True
		else:
			return False


	def get_tables(self, metadata):
		return [table for table in metadata.tables]

	def all_tables_exists(self, metadata):
		if set(self.get_tables(metadata)).issubset(set(self.get_tables(self.metadata))):
			return True
		else:
			return False

	

	def deploy(self, metadata, if_exist):
		if self.all_tables_exists(metadata):
			if if_exist=='drop':
				for _t in self.metadata.tables:
					_logger.info(f"Dropping table {_t}")
					print(f"Dropping table {_t}")
					self.drop_table(_t)
				print(f"Creating following tables -> {self.get_tables(metadata)}")
				_logger.info(f"Creating following tables -> {self.get_tables(metadata)}")
				Model.metadata.create_all(self.db_engine)
			elif if_exist == 'skip':
				_logger.info(f"Skipping schema deployment...")
			else:
				_logger.error(f'Wrong argument! for flag --if_exist -> Set to \'drop\' if you wish to drop and recrete schema. Set to \'skip\' if you wish to skip schema deployment')
		else:
			_logger.info(f"None of the Model tables exists! Creating following tables -> {self.get_tables(metadata)}")
			Model.metadata.create_all(self.db_engine)




def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--db_server", required=True)
	parser.add_argument("--db_name", required=True)
	parser.add_argument("--db_user", required=True)
	parser.add_argument("--db_pass", required=True)
	parser.add_argument("--db_driver", default="ODBC Driver 17 for SQL Server")
	parser.add_argument("--if_exist", default='skip',
						help="Drops tables if they exist")

	parser.add_argument("--subscription_id", required=True, type=str)
	parser.add_argument("--aml_resource_group",
						help="The Keyvault for the generated user passwords is created "
							 "through an azureml workspace. Even though this is the same "
							 "keyvault for all workspaces, each workspace needs to set the secret "
							 "to be able to read it though the azureML SDK. "
							 "This is the resource group where to look for the workspaces. "
							 "The last three letters of this resource group are taken as environment name.",
						required=True, type=str
						)



	return parser.parse_args()


def main():
	args = parse_args()

	db_driver = args.db_driver
	db_server = args.db_server
	db_name = args.db_name
	db_user = args.db_user
	db_pass = args.db_pass
	if_exist = args.if_exist

	subscription_id = args.subscription_id
	aml_resource_group = args.aml_resource_group

	schema = SchemaManager(db_name=db_name, db_server=db_server, username=db_user, password=db_pass)
	

	schema.deploy(Model.metadata, if_exist=if_exist)

	dbusermanager = DBUserManager(db_name=db_name, db_server=db_server, username=db_user, password=db_pass)

	username = 'mdbwriter'
	user_rights = ['GRANT SELECT ON SCHEMA::dbo', 'GRANT INSERT ON SCHEMA::dbo']
	
	if len(dbusermanager.get_login_detail(username)) == 0:
		_logger.info(f"Adding user with name ->{username} and rights -> {user_rights}")
		login, password = dbusermanager.add_user(username, username, 'dbo', user_rights)

		from azureml.core import Workspace, Keyvault
		workspaces = Workspace.list(subscription_id, resource_group=aml_resource_group)

		for ws in workspaces:
			secretmanager = SecretManager(None, subscription_id, aml_resource_group, ws)
			workspace = secretmanager.ws
			secretmanager.get_keyvault()
			_logger.info(f"Adding login password as "
                    f"secret metricsdb-{username} to keyvaults of workspace {workspace.name}")
			secretmanager.set_secret(f"metricsdb-{login}", password)
	else:
		_logger.error(f"User with name {username} already exists!")



if __name__ == "__main__":
	main()

