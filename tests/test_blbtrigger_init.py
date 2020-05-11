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
message = 'test'
bytes_stream = func.InputStream
bytes_stream.name = message
bytes_stream.length = len(message.encode('utf-8'))
# bytes_stream = func.InputStream.length = len(message.encode('utf-8'))
# stream = func.InputStream.read(bytes_stream)
print(bytes_stream)


def functionsss(incommingblob: func.InputStream):
    print(f"Python blob trigger function processed blob \n"
                 f"Name: {incommingblob.name}\n"
                 f"Blob Size: {incommingblob.length} bytes")


functionsss(bytes_stream)
