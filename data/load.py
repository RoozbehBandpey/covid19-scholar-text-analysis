import os


dirs = ['biorxiv_medrxiv']


# print(os.path.dirname(os.path.realpath(__file__)))


# print(os.listdir(os.path.join(os.path.dirname(
#     os.path.realpath(__file__)), f'CORD-19-research-challenge')))

for d in dirs:
	for file in os.listdir(os.path.join(os.path.dirname(
                os.path.realpath(__file__)), f'CORD-19-research-challenge/{d}/{d}/pdf_json')):
			print(file)
