import pandas as pd
import os

DATA_DIR = os.path.join(os.getcwd(), r'data')


if os.path.exists(os.path.join(DATA_DIR, 'files_index.csv')):
	df = pd.read_csv(os.path.join(DATA_DIR, 'files_index.csv'))

# for index, row in df.iterrows():
# 	print(row)

print(df[:1]['file_names'])



class DBHandler():
	def __init__(self, username=None, password=None, database_name=''):
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
