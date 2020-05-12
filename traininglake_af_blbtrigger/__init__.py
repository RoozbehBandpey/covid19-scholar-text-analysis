import logging

import azure.functions as func


def main(incommingblob: func.InputStream):
	logging.info(f"Python blob trigger function processed blob \n"
				 f"Name: {incommingblob.name}\n"
				 f"Blob Size: {incommingblob.length} bytes")

