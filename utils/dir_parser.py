import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), r'data\CORD-19-research-challenge')

file_names = []
relative_paths = []
for dirpath, dnames, fnames in os.walk(DATA_DIR):
	if len(fnames) > 0:
		for file_name in fnames:
			if file_name.endswith('.json'):
				relative_path = dirpath.split(DATA_DIR)[-1]
				file_names.append(file_name)
				relative_paths.append(relative_path)


file_index = {'file_names': file_names, 'relative_paths': relative_paths}
df = pd.DataFrame(file_index)
df.to_csv(os.path.join(os.path.dirname(DATA_DIR), 'files_index.csv'))