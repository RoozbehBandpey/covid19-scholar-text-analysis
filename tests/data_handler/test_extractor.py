import os
from data.data_handler.extractor import Extractor

extractor = Extractor()
print(extractor.DATA_DIR)

def test_path():
	extractor = Extractor()
	print(extractor.DATA_DIR)
	assert os.path.exists(extractor.DATA_DIR)


# if __name__ == "__main__":
	# extractor = Extractor()
	# print(extractor.DATA_DIR)
