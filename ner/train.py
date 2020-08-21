import os
import random
import string
import sys
from tempfile import TemporaryDirectory

import pandas as pd
import scrapbook as sb
import torch
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils_nlp.common.pytorch_utils import dataloader_from_dataset
from utils_nlp.common.timer import Timer
from utils_nlp.dataset import wikigold
from utils_nlp.dataset.ner_utils import read_conll_file
from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.transformers.named_entity_recognition import (
    TokenClassificationProcessor, TokenClassifier)


# Set QUICK_RUN = True to run the notebook on a small subset of data and a smaller number of epochs.
QUICK_RUN = False


myenv = Environment.from_conda_specification(name='myenv', file_path='mailbot.yml')
myenv.register(workspace=ws)
