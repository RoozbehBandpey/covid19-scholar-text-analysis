"""
Gets Wikigold dataset
Preprocess it
Register as Azure ML dataset
return the dataset as pandas dataframe
"""
from tempfile import TemporaryDirectory
import torch
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import requests
from collections import Iterable
import math
import logging
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)




supported_models = [
    # list(x.pretrained_config_archive_map)
    list(x.model_type)
    for x in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
]
supported_models = sorted([x for y in supported_models for x in y])

MAX_SEQ_LEN = 512

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


class TokenClassificationProcessor:
    """
    Process raw dataset for training and testing.
    Args:
        model_name (str, optional): The pretained model name.
            Defaults to "bert-base-cased".
        to_lower (bool, optional): Lower case text input.
            Defaults to False.
        cache_dir (str, optional): The default folder for saving cache files.
            Defaults to ".".
    """

    def __init__(self, model_name="bert-base-cased", to_lower=False, cache_dir="."):
        self.model_name = model_name
        self.to_lower = to_lower
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=to_lower,
            cache_dir=cache_dir,
            output_loading_info=False,
        )

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.
        Args:
            batch (tuple): A tuple containing input ids, attention mask,
                segment ids, and labels tensors.
            device (torch.device): A PyTorch device.
            model_name (bool): Model name used to format the inputs.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.
        Returns:
            dict: Dictionary containing input ids, segment ids, masks, and labels.
                Labels are only returned when train_mode is True.
        """
        batch = tuple(t.to(device) for t in batch)
        if model_name in supported_models:
            if train_mode:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            # distilbert doesn't support segment ids
            if model_name.split("-")[0] not in ["distilbert"]:
                inputs["token_type_ids"] = batch[2]

            return inputs
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    @staticmethod
    def create_label_map(label_lists, trailing_piece_tag="X"):
        """
        Create a dictionary object to map a label (str) to an ID (int).
        Args:
            label_lists (list): A list of label lists. Each element is a list of labels
                which presents class of each token.
            trailing_piece_tag (str, optional): Tag used to label trailing word pieces.
                Defaults to "X".
        Returns:
            dict: A dictionary object to map a label (str) to an ID (int).
        """

        unique_labels = sorted(set([x for y in label_lists for x in y]))
        label_map = {label: i for i, label in enumerate(unique_labels)}

        if trailing_piece_tag not in unique_labels:
            label_map[trailing_piece_tag] = len(unique_labels)

        return label_map

    def preprocess(
        self,
        text,
        max_len=MAX_SEQ_LEN,
        labels=None,
        label_map=None,
        trailing_piece_tag="X",
    ):
        """
        Tokenize and preprocesses input word lists, involving the following steps
            0. WordPiece tokenization.
            1. Convert string tokens to token ids.
            2. Convert input labels to label ids, if labels and label_map are
                provided.
            3. If a word is tokenized into multiple pieces of tokens by the
                WordPiece tokenizer, label the extra tokens with
                trailing_piece_tag.
            4. Pad or truncate input text according to max_seq_length
            5. Create input_mask for masking out padded tokens.
        Args:
            text (list): List of lists. Each sublist is a list of words in an
                input sentence.
            max_len (int, optional): Maximum length of the list of
                tokens. Lists longer than this are truncated and shorter
                ones are padded with "O"s. Default value is BERT_MAX_LEN=512.
            labels (list, optional): List of word label lists. Each sublist
                contains labels corresponding to the input word list. The lengths
                of the label list and word list must be the same. Default
                value is None.
            label_map (dict, optional): Dictionary for mapping original token
                labels (which may be string type) to integers. Default value
                is None.
            trailing_piece_tag (str, optional): Tag used to label trailing
                word pieces. For example, "criticize" is broken into "critic"
                and "##ize", "critic" preserves its original label and "##ize"
                is labeled as trailing_piece_tag. Default value is "X".
        Returns:
            TensorDataset: A TensorDataset containing the following four tensors.
                1. input_ids_all: Tensor. Each sublist contains numerical values,
                    i.e. token ids, corresponding to the tokens in the input
                    text data.
                2. input_mask_all: Tensor. Each sublist contains the attention
                    mask of the input token id list, 1 for input tokens and 0 for
                    padded tokens, so that padded tokens are not attended to.
                3. trailing_token_mask_all: Tensor. Each sublist is
                    a boolean list, True for the first word piece of each
                    original word, False for the trailing word pieces,
                    e.g. "##ize". This mask is useful for removing the
                    predictions on trailing word pieces, so that each
                    original word in the input text has a unique predicted
                    label.
                4. label_ids_all: Tensor, each sublist contains token labels of
                    a input sentence/paragraph, if labels is provided. If the
                    `labels` argument is not provided, it will not return this tensor.
        """

        def _is_iterable_but_not_string(obj):
            return isinstance(obj, Iterable) and not isinstance(obj, str)

        if max_len > MAX_SEQ_LEN:
            logging.warning(
                "Setting max_len to max allowed sequence length: {}".format(MAX_SEQ_LEN)
            )
            max_len = MAX_SEQ_LEN

        logging.warn(
            "Token lists with length > {} will be truncated".format(MAX_SEQ_LEN)
        )

        if not _is_iterable_but_not_string(text):
            # The input text must be an non-string Iterable
            raise ValueError("Input text must be an iterable and not a string.")
        else:
            # If the input text is a single list of words, convert it to
            # list of lists for later iteration
            if not _is_iterable_but_not_string(text[0]):
                text = [text]

        if labels is not None:
            if not _is_iterable_but_not_string(labels):
                raise ValueError("labels must be an iterable and not a string.")
            else:
                if not _is_iterable_but_not_string(labels[0]):
                    labels = [labels]

        label_available = True
        if labels is None:
            label_available = False
            # create an artificial label list for creating trailing token mask
            labels = [["O"] * len(t) for t in text]

        input_ids_all = []
        input_mask_all = []
        label_ids_all = []
        trailing_token_mask_all = []

        for t, t_labels in zip(text, labels):
            if len(t) != len(t_labels):
                raise ValueError(
                    "Num of words and num of labels should be the same {0}!={1}".format(
                        len(t), len(t_labels)
                    )
                )

            new_labels = []
            new_tokens = []
            for word, tag in zip(t, t_labels):
                sub_words = self.tokenizer.tokenize(word)
                for count, sub_word in enumerate(sub_words):
                    if count > 0:
                        tag = trailing_piece_tag
                    new_labels.append(tag)
                    new_tokens.append(sub_word)

            if len(new_tokens) > max_len:
                new_tokens = new_tokens[:max_len]
                new_labels = new_labels[:max_len]
            input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1.0] * len(input_ids)

            # Zero-pad up to the max sequence length.
            padding = [0.0] * (max_len - len(input_ids))
            label_padding = ["O"] * (max_len - len(input_ids))

            input_ids += padding
            input_mask += padding
            new_labels += label_padding

            trailing_token_mask_all.append(
                [True if label != trailing_piece_tag else False for label in new_labels]
            )

            if label_map:
                label_ids = [label_map[label] for label in new_labels]
            else:
                label_ids = new_labels

            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            label_ids_all.append(label_ids)

        if label_available:
            td = TensorDataset(
                torch.LongTensor(input_ids_all),
                torch.LongTensor(input_mask_all),
                torch.LongTensor(trailing_token_mask_all),
                torch.LongTensor(label_ids_all),
            )
        else:
            td = TensorDataset(
                torch.LongTensor(input_ids_all),
                torch.LongTensor(input_mask_all),
                torch.LongTensor(trailing_token_mask_all),
            )
        return td



def load():

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
	# MODEL_NAME = "bert-base-cased"
	MODEL_NAME = "distilbert"
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
	train_dataloader = dataloader_from_dataset(
		train_dataset, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=True, distributed=False
	)

	test_dataset = processor.preprocess(
		text=test_sentence_list,
		max_len=MAX_SEQ_LENGTH,
		labels=test_labels_list,
		label_map=label_map,
		trailing_piece_tag=TRAILING_PIECE_TAG,
	)
	test_dataloader = dataloader_from_dataset(
		test_dataset, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS, shuffle=False, distributed=False
	)


if __name__ == "__main__":
	load()
	# for x in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING:
	# 	# print(x)
	# 	# print(dir(x))
	# 	print(x.model_type)
