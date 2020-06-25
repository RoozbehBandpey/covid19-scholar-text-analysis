import os
from data.data_handler.extractor import Extractor



def test_path():
	extractor = Extractor()
	assert os.path.exists(extractor.DATA_DIR) == True