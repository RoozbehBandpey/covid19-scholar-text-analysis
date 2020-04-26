import os
import json

dirs = ['biorxiv_medrxiv']


for d in dirs:
	for file in os.listdir(os.path.join(os.path.dirname(
                os.path.realpath(__file__)), f'CORD-19-research-challenge/{d}/{d}/pdf_json')):
			print(file)
			
