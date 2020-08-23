"""
Gets Wikigold dataset
Preprocess it
Register as Azure ML dataset
"""
from tempfile import TemporaryDirectory
import torch
import os
import requests
import math
import logging
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


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



# Set QUICK_RUN = True to run the notebook on a small subset of data and a smaller number of epochs.
QUICK_RUN = False

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
DATA_PATH = os.path.join(os.getcwd(), 'data', 'ner', 'data')

# the cache data path during find tuning
CACHE_DIR = os.path.join(os.getcwd(), 'data', 'ner', 'cache')

if not os.path.exists(os.path.dirname(DATA_PATH)):
	os.mkdir(os.path.dirname(DATA_PATH))
	if not os.path.exists(DATA_PATH):
		os.mkdir(DATA_PATH)
	if not os.path.exists(CACHE_DIR):
		os.mkdir(CACHE_DIR)

# set random seeds
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)

# model configurations
NUM_TRAIN_EPOCHS = 5
MODEL_NAME = "bert-base-cased"
DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 200
TRAILING_PIECE_TAG = "X"
NUM_GPUS = None  # uses all if available
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
sentence_list, labels_list = read_conll_file(data_file, sep=" ")

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
