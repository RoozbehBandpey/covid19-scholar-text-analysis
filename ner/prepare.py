"""
Gets Wikigold dataset
Preprocess it
Register as Azure ML dataset
return the dataset as pandas dataframe
"""

import torch
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
import requests
from collections import Iterable
import math
import logging
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace, Datastore, Dataset
from utils import TokenClassificationProcessor, MAX_SEQ_LEN


ws = Workspace.from_config(
    auth=AzureCliAuthentication(),
    path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
                'config.json'
        )
    )

print(f'Workspace: {ws.name}')


logger = logging.getLogger(__name__)


# Versions
print(torch.__version__)
print(requests.__version__)

# Utility functions


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.
    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.
    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        if not os.path.isdir(work_directory):
            os.makedirs(work_directory)
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                r.iter_content(block_size),
                total=num_iterables,
                unit="KB",
                unit_scale=True,
            ):
                file.write(data)
    else:
        logger.info("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def preprocess_conll(text, sep="\t"):
    """
    Converts data in CoNLL format to word and label lists.
    Args:
        text (str): Text string in conll format, e.g.
            "Amy B-PER
             ADAMS I-PER
             works O
             at O
             the O
             University B-ORG
             of I-ORG
             Minnesota I-ORG
             . O"
        sep (str, optional): Column separator
            Defaults to \t
    Returns:
        tuple:
            (list of word lists, list of token label lists)
    """
    text_list = text.split("\n\n")
    if text_list[-1] in (" ", ""):
        text_list = text_list[:-1]

    max_seq_len = 0
    sentence_list = []
    labels_list = []
    for s in text_list:
        # split each sentence string into "word label" pairs
        s_split = s.split("\n")
        # split "word label" pairs
        s_split_split = [t.split(sep) for t in s_split]
        sentence_list.append([t[0] for t in s_split_split if len(t) > 1])
        labels_list.append([t[1] for t in s_split_split if len(t) > 1])

        if len(s_split_split) > max_seq_len:
            max_seq_len = len(s_split_split)
    print("Maximum sequence length is: {0}".format(max_seq_len))
    return sentence_list, labels_list


def dataloader_from_dataset(
    ds, batch_size=32, num_gpus=None, shuffle=False, distributed=False
):
    """Creates a PyTorch DataLoader given a Dataset object.
    Args:
        ds (torch.utils.data.DataSet): A PyTorch dataset.
        batch_size (int, optional): Batch size.
            If more than 1 gpu is used, this would be the batch size per gpu.
            Defaults to 32.
        num_gpus (int, optional): The number of GPUs to be used. Defaults to None.
        shuffle (bool, optional): If True, a RandomSampler is used. Defaults to False.
        distributed (book, optional): If True, a DistributedSampler is used.
        Defaults to False.
    Returns:
        Module, DataParallel: A PyTorch Module or
            a DataParallel wrapper (when multiple gpus are used).
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    batch_size = batch_size * max(1, num_gpus)

    if distributed:
        sampler = DistributedSampler(ds)
    else:
        sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)

    return DataLoader(ds, sampler=sampler, batch_size=batch_size)


def read_conll_file(file_path, sep="\t", encoding=None):
    """
    Reads a data file in CoNLL format and returns word and label lists.
    Args:
        file_path (str): Data file path.
        sep (str, optional): Column separator. Defaults to "\t".
        encoding (str): File encoding used when reading the file.
            Defaults to None.
    Returns:
        (list, list): A tuple of word and label lists (list of lists).
    """
    with open(file_path, encoding=encoding) as f:
        data = f.read()
    return preprocess_conll(data, sep=sep)




def load(quick_run, data_path, cache_path, model_name, num_gpus, random_seed):

	# Set QUICK_RUN = True to run the notebook on a small subset of data and a smaller number of epochs.
	QUICK_RUN = quick_run

	# Wikigold dataset
	DATA_URL = (
		"https://raw.githubusercontent.com/juand-r/entity-recognition-datasets"
		"/master/data/wikigold/CONLL-format/data/wikigold.conll.txt"
	)

	# fraction of the dataset used for testing
	TEST_DATA_FRACTION = 0.3

	# sub-sampling ratio
	SAMPLE_RATIO = 1

	# the data path used to save the downloaded data file
	DATA_PATH = data_path

	# the cache data path during find tuning
	CACHE_DIR = cache_path

	if not os.path.exists(os.path.dirname(DATA_PATH)):
		os.mkdir(os.path.dirname(DATA_PATH))
		if not os.path.exists(DATA_PATH):
			os.mkdir(DATA_PATH)
		if not os.path.exists(CACHE_DIR):
			os.mkdir(CACHE_DIR)

	# set random seeds
	RANDOM_SEED = random_seed
	torch.manual_seed(RANDOM_SEED)


	MODEL_NAME = model_name
	# MODEL_NAME = "distilbert"
	DO_LOWER_CASE = False
	MAX_SEQ_LENGTH = 200
	TRAILING_PIECE_TAG = "X"
	NUM_GPUS = num_gpus
	BATCH_SIZE = 16


	# update variables for quick run option
	if QUICK_RUN:
		SAMPLE_RATIO = 0.1
		NUM_TRAIN_EPOCHS = 1


	# download data
	file_name = DATA_URL.split("/")[-1]  # a name for the downloaded file
	maybe_download(DATA_URL, file_name, DATA_PATH)
	data_file = os.path.join(DATA_PATH, file_name)

	# parse CoNll file
	sentence_list, labels_list = read_conll_file(data_file, sep=" ", encoding='utf-8')

	# sub-sample (optional)
	random.seed(RANDOM_SEED)
	sample_size = int(SAMPLE_RATIO * len(sentence_list))
	sentence_list, labels_list = list(
		zip(*random.sample(list(zip(sentence_list, labels_list)), k=sample_size))
	)

	# train-test split
	train_sentence_list, test_sentence_list, train_labels_list, test_labels_list = train_test_split(
		sentence_list, labels_list, test_size=TEST_DATA_FRACTION, random_state=RANDOM_SEED
	)

	processor = TokenClassificationProcessor(model_name=MODEL_NAME, to_lower=DO_LOWER_CASE, cache_dir=CACHE_DIR)


	label_map = TokenClassificationProcessor.create_label_map(
		label_lists=labels_list, trailing_piece_tag=TRAILING_PIECE_TAG
	)

	train_dataset = processor.preprocess(
		text=train_sentence_list,
		max_len=MAX_SEQ_LENGTH,
		labels=train_labels_list,
		label_map=label_map,
		trailing_piece_tag=TRAILING_PIECE_TAG,
	)

	# train_data_loader = DataLoader(train_dataset)
	test_dataset = processor.preprocess(
		text=test_sentence_list,
		max_len=MAX_SEQ_LENGTH,
		labels=test_labels_list,
		label_map=label_map,
		trailing_piece_tag=TRAILING_PIECE_TAG,
	)

	torch.save(train_dataset, os.path.join(DATA_PATH, 'train.pt'))
	torch.save(test_dataset, os.path.join(DATA_PATH, 'test.pt'))
	torch.save(label_map, os.path.join(DATA_PATH, 'label_map.pt'))

	# Default datastore
	def_data_store = ws.get_default_datastore()

	# Get the blob storage associated with the workspace
	def_blob_store = Datastore(ws, "workspaceblobstore")

	# Get file storage associated with the workspace
	def_file_store = Datastore(ws, "workspacefilestore")

	try:
		def_blob_store.upload_files(
	    			[os.path.join(DATA_PATH, 'train.pt')], target_path="nerdata", overwrite=True, show_progress=True)
	except Exception as e:
		print(f"Failed to upload -> {e}")

	try:
		def_blob_store.upload_files(
                    [os.path.join(DATA_PATH, 'test.pt')], target_path="nerdata", overwrite=True, show_progress=True)
	except Exception as e:
		print(f"Failed to upload -> {e}")

	try:
		def_blob_store.upload_files(
                    [os.path.join(DATA_PATH, 'label_map.pt')], target_path="nerdata", overwrite=True, show_progress=True)
	except Exception as e:
		print(f"Failed to upload -> {e}")

	train_datastore_paths = [(def_blob_store, 'nerdata/train.pt')]
	test_datastore_paths = [(def_blob_store, 'nerdata/test.pt')]
	label_map_datastore_paths = [(def_blob_store, 'nerdata/label_map.pt')]

	# def_blob_store.upload(src_dir=DATA_PATH, target_path="nerdata", overwrite=True, show_progress=True)

	train_ds = Dataset.File.from_files(path=train_datastore_paths)
	test_ds = Dataset.File.from_files(path=test_datastore_paths)
	label_map_ds = Dataset.File.from_files(path=label_map_datastore_paths)

	train_ds = train_ds.register(workspace=ws,
                                  name='ner_bert_train_ds',
                                  description='Named Entity Recognition with BERT (Training set)',
                                  create_new_version=False)

	test_ds = test_ds.register(workspace=ws,
                                  name='ner_bert_test_ds',
                                  description='Named Entity Recognition with BERT (Testing set)',
                                  create_new_version=False)

	label_map_ds = label_map_ds.register(workspace=ws,
                            name='ner_bert_label_map_ds_ds',
                                  description='Named Entity Recognition with BERT (Testing set)',
                                  create_new_version=False)

	train_dataloader = dataloader_from_dataset(
		train_dataset, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=True, distributed=False
	)

	test_dataloader = dataloader_from_dataset(
		test_dataset, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=False, distributed=False
	)

	return (train_dataloader, test_dataloader, label_map)


if __name__ == "__main__":
	load()
