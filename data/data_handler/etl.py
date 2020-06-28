from loader import Loader
from extractor import Extractor
import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

# # For local run
sys.path.append(os.getcwd())
from utils.load_env import append_api_vars

append_api_vars(os.path.join(os.path.dirname(
	os.path.realpath(__file__)), 'kaggle.json'))

class ETL():
	def __init__(self):
		self.loader_obj = Loader()
		self.extractor_obj = Extractor()
		pass


	def download(self):
		api = KaggleApi()
		api.authenticate()


if __name__ == "__main__":
	etl = ETL()
	etl.download()
	print(os.environ.get('username'))
	print(os.environ.get('key'))
