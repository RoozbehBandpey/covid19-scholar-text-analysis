import os
import uuid
from azure.storage.blob import BlobServiceClient

try:
    print("Azure Blob storage v12 - Python quickstart sample")
    # Quick start code goes here
except Exception as ex:
    print('Exception:')
    print(ex)



class BlobHandler():
	def __init__(self):
		"""Connects to blob storage and contrains typical blob actions e.g., read, write, list, copy, etc.,"""
		pass

