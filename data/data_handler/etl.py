# from loader import Loader
# from extractor import Extractor
from kaggle.api.kaggle_api_extended import KaggleApi



class ETL():
	def __init__(self):
		# self.loader_obj = Loader()
		# self.extractor_obj = Extractor()
		pass


	def download(self):
		api = KaggleApi()
		api.authenticate()


if __name__ == "__main__":
	etl = ETL()
	etl.download()
