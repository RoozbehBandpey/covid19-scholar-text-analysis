import os
import uuid
import time
from load_env import append_vars
import azure.storage.blob
from azure.storage.blob import BlobServiceClient, ContainerClient


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


	def container_exists(self, container_name):
		self.list_containers(show=False)
		container_names = [container['name'] for container in self.containers]
		if container_name in container_names:
			return True
		else:
			return False
	

	def get_container_client(self, container_name):
		if self.container_exists(container_name):
			try:
				self.container_client = self.blob_service_client.get_container_client(container_name)
				# print(self.container_client.get_container_properties())
				return self.container_client
			except Exception as e:
				print(f"[ERROR] -> {e}")
				return None
		else:
			print(f"[ERROR] container {container_name} does not exist!")
			return None


	def create_container(self, container_name):
		if self.container_exists(container_name):
			print(f"[ERROR] container {container_name} Already exists in target Blob storage!")
		else:
			try:
				self.container_client = self.blob_service_client.get_container_client(container_name)
				self.container_client.create_container()
			except Exception as e:
				print(f"[ERROR] -> {e}")

	def copy_blob(self, source_blob, target_path, target_file_name):
		#TODO need to be tested because my baby wants to cuddle
		status = None
		copied_blob = self.blob_service_client.get_blob_client(target_path, target_file_name)
		# Copy started
		copied_blob.start_copy_from_url(source_blob)
		for i in range(10):
			props = copied_blob.get_blob_properties()
			status = props.copy.status
			print("Copy status: " + status)
			if status == "success":
				# Copy finished
				break
			time.sleep(10)

		if status != "success":
			# if not finished after 100s, cancel the operation
			props = copied_blob.get_blob_properties()
			print(props.copy.status)
			copy_id = props.copy.id
			copied_blob.abort_copy(copy_id)
			props = copied_blob.get_blob_properties()
			print(props.copy.status)


if __name__ == "__main__":
	blob = BlobHandler()
	blob.authenticate_with_connstr()
	blob.get_container_client('training-lake-container')
	print(dir(blob.container_client))
