import os





class DataHelper():
	def __init__(self, username=None, password=None, database_name=''):
		self.DATA_DIR = os.path.join(os.getcwd(), 'data')
		self.DATASET_DIR = os.path.join(self.DATA_DIR, 'CORD-19-research-challenge')


	def __normalize_space(self, text):
		return " ".join(text.split())

	def _get_authors(self, paper_meta):
		"""Returns authors list from paper metadata"""
		authors = []
		for author in paper_meta['authors']:
			if author['first'] != '':
				first = author['first']
			else:
				first = ''
			if len(author['middle']) != 0:
				middle_name = ''
				for m in author['middle']:
					middle_name += m
			else:
				middle_name = ''
			if author['last'] != '':
				last = author['last']
			else:
				last = ''

			authors.append(self.__normalize_space(f'{first} {middle_name} {last}'))
		return authors

	def _get_title(self, paper_meta):
		"""Returns title from paper metadata"""
		return paper_meta['title']

	def _get_text(self, text_meta):
		"""Returns text from paper text/abstract text metadata list"""
		full_text = ''
		for text in text_meta:
			full_text += text['text'] + '\n\n'

		if full_text[-1] == '\n':
			full_text = full_text[:-2]

		return full_text
