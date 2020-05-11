# import StringIO
import io


# def test_input_stream():
# 	output = StringIO.StringIO()
# 	output.write('First line.\n')

# test_input_stream()

import azure.functions as func
# buffered_message = io.BufferedIOBase(b"Test")
# buffered_message = io.BytesIO(b"{'name': 'test', 'length': 10}")
# print(buffered_message.getvalue())
stream = func.InputStream.name('test')
print(stream)


# def functionsss(incommingblob: func.InputStream):
#     print(f"Python blob trigger function processed blob \n"
#                  f"Name: {incommingblob.name}\n"
#                  f"Blob Size: {incommingblob.length} bytes")

# functionsss(buffered_message)
