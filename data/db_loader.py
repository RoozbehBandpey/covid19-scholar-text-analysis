import pandas as pd
import os

DATA_DIR = os.path.join(os.getcwd(), r'data')


if os.path.exists(os.path.join(DATA_DIR, 'files_index.csv')):
	df = pd.read_csv(os.path.join(DATA_DIR, 'files_index.csv'))

print(df)