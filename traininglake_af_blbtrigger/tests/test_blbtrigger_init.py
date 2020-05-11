# import StringIO
import io
import os
import json
import sys
import unittest
import azure.functions as func
from . import 

def test_input_stream():
	test_file_name = 'test.json'
	test_data = json.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), test_file_name), 'rb'))
	bytes_stream = func.InputStream
	bytes_stream.name = test_file_name
	bytes_stream.length = sys.getsizeof((test_data)

test_input_stream()





# buffered_message = io.BufferedIOBase(b"Test")
# buffered_message = io.BytesIO(b"{'name': 'test', 'length': 10}")
# print(buffered_message.getvalue())
# test_data_bin = io.BytesIO(json.dumps(test_data).encode())
test_data_bin = io.BufferedIOBase(json.dumps(test_data).encode())
print(test_data)
print(test_data_bin)
print(type(test_data_bin))

bytes_stream = func.InputStream
bytes_stream.write(test_data_bin)
print(bytes_stream)
# test_data_bin.write(bytes_stream.read()
# # bytes_stream = bytes_stream
# bytes_stream.name = test_file_name
# bytes_stream.length = sys.getsizeof((test_data)
#                                     excel_binary=io.BytesIO()
#                                     excel_binary.write(myblob.read())
# # print(bytes_stream)


# def functionsss(incommingblob: func.InputStream):
#     print(f"Python blob trigger function processed blob \n"
#                  f"Name: {incommingblob.name}\n"
#                  f"Blob Size: {incommingblob.length} bytes\n"
#           f"File content: {incommingblob.read}")


# functionsss(bytes_stream)
