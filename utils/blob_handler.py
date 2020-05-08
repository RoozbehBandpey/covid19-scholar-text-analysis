import os
import uuid
from load_env import append_vars
import azure.storage.blob
from azure.storage.blob import BlobServiceClient, ContainerClient

try:
	print("Azure Blob storage v12 - Python quickstart sample")
	# Quick start code goes here
except Exception as ex:
	print('Exception:')
	print(ex)

DEBUG = True

class BlobHandler(object):
	def __init__(self):
		"""Connects to blob storage and contrains typical blob actions e.g., read, write, list, copy, etc.,"""
		print(f"You are using Azure Storage SDK version {azure.storage.blob.__version__}")
		if DEBUG:
			if os.environ.get('AZURE_STORAGE_CONNECTION_STRING') is None:
				append_vars(os.path.join(os.path.dirname(
					os.path.realpath(__file__)), 'local.settings.json'))

		self._storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
		self._oauth_storage_account_name = os.getenv("OAUTH_STORAGE_ACCOUNT_NAME")

		self.blob_url = f"https://{self._storage_account_name}.blob.core.windows.net"
		self.blob_oauth_url = f"https://{self._oauth_storage_account_name}.blob.core.windows.net"

		self._connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
		self._sas_key = os.getenv("AZURE_STORAGE_ACCESS_KEY")
		self._active_directory_application_id = os.getenv("ACTIVE_DIRECTORY_APPLICATION_ID")
		self._active_directory_application_secret = os.getenv("ACTIVE_DIRECTORY_APPLICATION_SECRET")
		self._active_directory_tenant_id = os.getenv("ACTIVE_DIRECTORY_TENANT_ID")

	def authenticate_with_connstr(self):
		try:
			self.blob_service_client = BlobServiceClient.from_connection_string(self._connection_string)
			print(f"Successfully connected to blob storage [{self.blob_service_client.account_name}]")
		except Exception as e:
			print(f"[ERROR] -> {e}")

	def list_containers(self, show=True):
		self.containers = list(self.blob_service_client.list_containers())
		if show:
			for item in self.containers:
				print(item)
		#get_container_client



if __name__ == "__main__":
	blob = BlobHandler()
	blob.authenticate_with_connstr()
	blob.list_containers()
