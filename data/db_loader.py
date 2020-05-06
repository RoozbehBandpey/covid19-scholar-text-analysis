import os
import sys
import urllib
import pandas as pd
from sqlalchemy import Integer, String
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import inspect
from sqlalchemy import Column
from sqlalchemy import select
from sqlalchemy import Table

sys.path.append(os.getcwd())
from utils.load_env import append_vars


DEBUG = True


class DBHandler():
	def __init__(self, username=None, password=None, database_name=''):
		self.DATA_DIR = os.path.join(os.getcwd(), r'data')
			#for local dev gets from environment variables
		if DEBUG:
			if os.environ.get('SQL_CONNSTR') is None:
				append_vars(os.path.join(os.path.dirname(
					os.path.realpath(__file__)), 'local.settings.json'))
		# Once deployed gets from ADO variable groups
		self._database_name = os.environ.get('SQL_DATABASE')
		self._hostname = f"{self._database_name}.database.windows.net"
		self._username = os.environ.get('SQL_USERNAME')
		self._password = os.environ.get('SQL_PASSWORD')
		self._driver = "{ODBC Driver 17 for SQL Server}"
		self._connection_string = os.environ.get('SQL_CONNSTR')
		# Connect
		params = urllib.parse.quote_plus(self._connection_string)
		self.db_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
		self.metadata = MetaData(bind=self.db_engine)
		try:
			conn = self.db_engine.connect()
			print(f'Successfully connected to {self._database_name}, connection is closed- > {conn.closed}')
		except Exception as e:
			print(f'[ERROR] Could not connect! -> {e}')


	def load_data_as_df(self, index_file_name):
		if os.path.exists(os.path.join(self.DATA_DIR, index_file_name)):
			df = pd.read_csv(os.path.join(self.DATA_DIR, index_file_name))
			# for index, row in df.iterrows():
			# 	print(row)
			df.drop(['Unnamed: 0'], axis=1, inplace=True)
			file_name = df.iloc[0]['file_names']
			relative_path = df.iloc[0]['relative_paths']
			print(file_name, relative_path)
		else:
			print("[ERROR] Index file does not exist!")


if __name__ == "__main__":
	db = DBHandler()
	db.load_data_as_df('files_index.csv')
