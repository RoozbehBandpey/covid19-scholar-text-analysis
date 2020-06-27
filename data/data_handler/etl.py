from loader import Loader
from extractor import Extractor



class ETL():
	def __init__(self):
		self.loader_obj = Loader()
		self.extractor_obj = Extractor()


	def download(self)
