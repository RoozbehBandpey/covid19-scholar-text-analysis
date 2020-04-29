import os
import json

dirs = ['biorxiv_medrxiv']

DATA_DIR = os.path.join(os.getcwd(), r'data\CORD-19-research-challenge')

sums = 0
for dirpath, dnames, fnames in os.walk(DATA_DIR):
	if len(fnames) > 0:
		for file_name in fnames:
			if file_name.endswith('.json'):
				category = dirpath.strip(DATA_DIR)
				print(file_name, category)

# 	print(len(fnames))
# 	if len(fnames) > 1000:
# 		sums += len(fnames)

# print(sums)
#     for f in fnames:
#         if f.endswith(".x"):
#             x(os.path.join(dirpath, f))
#         elif f.endswith(".xc"):
#             xc(os.path.join(dirpath, f))

# 	for file in os.listdir(os.path.join(os.path.dirname(
#                 os.path.realpath(__file__)), f'CORD-19-research-challenge/{d}/{d}/pdf_json')):
# 			print(file)
			
