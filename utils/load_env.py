import os
import json
import argparse
import sys



def append_vars(path_to_local_config):
	env_vars = dict()

	if path_to_local_config.endswith('.json'):
		with open(path_to_local_config) as f:
			loadedenv = dict()
			try:
				loadedenv = json.load(f)
			except Exception as e:
				print(f'Error! Could not load {path_to_local_config}')
				print(f'\t{e}')
				sys.exit()
				
			if 'Values' in loadedenv.keys():
				for var in loadedenv['Values']:
					val = loadedenv['Values'][var]
					env_vars[var] = val
					os.environ[var] = val
			else:
				print('Error! Not a valid local.settings.json file! \'Values\' key is missing.')
				sys.exit()

	else:
		print('Error! Not a JSON file')
		sys.exit()


	if len(env_vars) >0:
		for var in env_vars:
			print(f"Variable '{var}' with value '{os.environ.get(var)}' were added to environment variables")
	else:
		print("Error! No variables were added")


def append_api_vars(path_to_api_config):
	env_vars = dict()

	if path_to_api_config.endswith('.json'):
		with open(path_to_api_config) as f:
			loadedenv = dict()
			try:
				loadedenv = json.load(f)
			except Exception as e:
				print(f'Error! Could not load {path_to_api_config}')
				print(f'\t{e}')
				sys.exit()
				

			for var in loadedenv:
				val = loadedenv[var]
				env_vars[var] = val
				os.environ[var] = val


	else:
		print('Error! Not a JSON file')
		sys.exit()


	if len(env_vars) >0:
		for var in env_vars:
			print(f"Variable '{var}' with value '{os.environ.get(var)}' were added to environment variables")
	else:
		print("Error! No variables were added")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Setting environment variables from a given local.settings.json file!')
	parser.add_argument('Path',
						metavar='path',
						type=str,
						help='The path pointing to the local.settings.json file')

	args = parser.parse_args()
	append_vars(args.Path)
