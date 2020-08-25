from prepare import load
import torch
from utils import TokenClassifier, Timer
import logging
import os
from azureml.core.run import Run


logger = logging.getLogger(__name__)

run = Run.get_context()

# the data path used to save the downloaded data file
DATA_PATH = os.path.join(os.getcwd(), 'data', 'ner', 'data')

# the cache data path during find tuning
CACHE_DIR = os.path.join(os.getcwd(), 'data', 'ner', 'cache')
NUM_TRAIN_EPOCHS = 1
MODEL_NAME = "bert-base-cased"
NUM_GPUS = None  # uses all if available
RANDOM_SEED = 1

train_dataloader, test_dataloader, label_map = load(
    quick_run=False, data_path=DATA_PATH, cache_path=CACHE_DIR, model_name=MODEL_NAME, num_gpus=NUM_GPUS, random_seed=RANDOM_SEED)



# Instantiate a TokenClassifier class for NER using pretrained transformer model
model = TokenClassifier(
    model_name=MODEL_NAME,
    num_labels=len(label_map),
    cache_dir=CACHE_DIR
)

# Fine tune the model using the training dataset
with Timer() as t:
    model.fit(
        train_dataloader=train_dataloader,
        num_epochs=NUM_TRAIN_EPOCHS,
        num_gpus=NUM_GPUS,
        local_rank=-1,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        verbose=False,
        seed=RANDOM_SEED
    )


run.log('Training time', t.interval / 3600)
print("Training time : {:.3f} hrs".format(t.interval / 3600))
