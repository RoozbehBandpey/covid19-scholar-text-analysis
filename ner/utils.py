import torch
from transformers import AutoTokenizer
import logging
from torch.utils.data import TensorDataset
from collections import Iterable
import random
import numpy as np
import torch
import time
import os
from tqdm import tqdm
from timeit import default_timer
from transformers import AdamW, get_linear_schedule_with_warmup
import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)



logger = logging.getLogger(__name__)
MAX_SEQ_LEN = 512

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


class Timer(object):
    """Timer class.
    Original code: https://github.com/miguelgfierro/codebase
    Examples:
        >>> import time
        >>> t = Timer()
        >>> t.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> with Timer() as t:
        ...   time.sleep(1)
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 1...'
    """

    def __init__(self):
        self._timer = default_timer
        self._interval = 0
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return "{:0.4f}".format(self.interval)

    def start(self):
        """Start the timer."""
        self.init = self._timer()
        self.running = True

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self._interval = self.end - self.init
            self.running = False
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() "
                "as t:"
            )

    @property
    def interval(self):
        if self.running:
            raise ValueError("Timer has not been stopped, please use stop().")
        else:
            return self._interval


class Transformer:
    def __init__(self, model_name, model, cache_dir):
        self._model_name = model_name
        self._model_type = model_name.split("-")[0]
        self.model = model
        self.cache_dir = cache_dir

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_type(self):
        return self._model_type

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_default_optimizer(model, weight_decay, learning_rate, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        return optimizer

    @staticmethod
    def get_default_scheduler(optimizer, warmup_steps, num_training_steps):
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler

    def prepare_model_and_optimizer(
        self,
        num_gpus,
        gpu_ids,
        local_rank,
        weight_decay,
        learning_rate,
        adam_epsilon,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint_state_dict=None,
    ):
        """
        This function initializes an optimizer and moves the model to a device.
        It can be used by most child classes before calling fine_tune.
        Child classes that require custom optimizers need to either override this
            function or implement the steps listed below in the specified order
            before fine-tuning.
        The steps are performed in the following order:
            1. Move model to device
            2. Create optimizer
            3. Initialize amp
            4. Parallelize model
        """

        amp = get_amp(fp16)

        # get device
        device, num_gpus = get_device(
            num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=local_rank
        )

        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # init optimizer
        self.optimizer = Transformer.get_default_optimizer(
            self.model, weight_decay, learning_rate, adam_epsilon
        )

        if fp16 and amp:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=fp16_opt_level
            )

        if checkpoint_state_dict:
            self.optimizer.load_state_dict(checkpoint_state_dict["optimizer"])
            self.model.load_state_dict(checkpoint_state_dict["model"])

            if fp16 and amp:
                amp.load_state_dict(checkpoint_state_dict["amp"])

        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
        )

        return device, num_gpus, amp

    def fine_tune(
        self,
        train_dataloader,
        get_inputs,
        device,
        num_gpus=None,
        max_steps=-1,
        global_step=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        optimizer=None,
        scheduler=None,
        fp16=False,
        amp=None,
        local_rank=-1,
        verbose=True,
        seed=None,
        report_every=10,
        save_every=-1,
        clip_grad_norm=True,
        validation_function=None,
    ):

        if seed is not None:
            Transformer.set_seed(seed, num_gpus > 0)

        # init training
        tr_loss = 0.0
        accum_loss = 0
        train_size = 0
        self.model.train()
        self.model.zero_grad()

        # train
        start = time.time()
        # TODO: Is this while necessary???
        while global_step < max_steps:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=local_rank not in [-1, 0] or not verbose,
            )
            for step, batch in enumerate(epoch_iterator):
                inputs = get_inputs(batch, device, self.model_name)
                outputs = self.model(**inputs)

                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    # Accomondate models based on older versions of Transformers,
                    # e.g. UniLM
                    loss = outputs

                if num_gpus > 1:
                    loss = loss.mean()

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16 and amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                accum_loss += loss.item()
                train_size += list(inputs.values())[0].size()[0]
                if (step + 1) % gradient_accumulation_steps == 0:

                    global_step += 1

                    if clip_grad_norm:
                        if fp16 and amp:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )

                    if global_step % report_every == 0 and verbose:
                        end = time.time()
                        endtime_string = datetime.datetime.fromtimestamp(end).strftime(
                            "%d/%m/%Y %H:%M:%S"
                        )
                        log_line = """timestamp: {0:s}, average loss: {1:.6f}, time duration: {2:f},
                            number of examples in current reporting: {3:.0f}, step {4:.0f}
                            out of total {5:.0f}""".format(
                            endtime_string,
                            accum_loss / report_every,
                            end - start,
                            # list(inputs.values())[0].size()[0],
                            train_size,
                            global_step,
                            max_steps,
                        )
                        logger.info(log_line)
                        print(log_line)
                        accum_loss = 0
                        train_size = 0
                        start = end
                    if optimizer:
                        if type(optimizer) == list:
                            for o in optimizer:
                                o.step()
                        else:
                            optimizer.step()
                    if scheduler:
                        if type(scheduler) == list:
                            for s in scheduler:
                                s.step()
                        else:
                            scheduler.step()
                    self.model.zero_grad()

                    if (
                        save_every != -1
                        and global_step % save_every == 0
                        and verbose
                        and local_rank in [-1, 0]
                    ):
                        saved_model_path = os.path.join(
                            self.cache_dir, f"{self.model_name}_step_{global_step}.pt"
                        )
                        self.save_model(saved_model_path)
                        if validation_function:
                            validation_log = validation_function(self)
                            logger.info(validation_log)
                            print(validation_log)
                if global_step > max_steps:
                    epoch_iterator.close()
                    break
        if fp16 and amp:
            self.amp_state_dict = amp.state_dict()

        # release GPU memories
        self.model.cpu()
        torch.cuda.empty_cache()

        return global_step, tr_loss / global_step

    def predict(self, eval_dataloader, get_inputs, num_gpus, gpu_ids, verbose=True):
        # get device
        device, num_gpus = get_device(num_gpus=num_gpus, gpu_ids=gpu_ids, local_rank=-1)

        # move model
        self.model = move_model_to_device(model=self.model, device=device)

        # parallelize model
        self.model = parallelize_model(
            model=self.model,
            device=device,
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=-1,
        )

        # predict
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Scoring", disable=not verbose):
            with torch.no_grad():
                inputs = get_inputs(batch, device, self.model_name, train_mode=False)
                outputs = self.model(**inputs)
                logits = outputs[0]
            yield logits.detach().cpu().numpy()

    def save_model(self, file_name=None):
        """
        Saves the underlying PyTorch module's state.
        Args:
            file_name (str, optional): File name to save the model's `state_dict()`
                that can be loaded by torch.load().
                If None, the trained model, configuration and tokenizer are saved
                using `save_pretrained()`; and the file is going to be saved under
                "fine_tuned" folder of the cached directory of the object.
                Defaults to None.
        """

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        if file_name:
            logger.info("Saving model checkpoint to %s", file_name)
            torch.save(model_to_save.state_dict(), file_name)
        else:
            output_model_dir = os.path.join(self.cache_dir, "fine_tuned")

            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(output_model_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", output_model_dir)
            model_to_save.save_pretrained(output_model_dir)

    def load_model(self, file_name):
        """
        Loads a PyTorch module's state.
        Args:
            file_name (str): File name of saved the model's `state_dict()`
        """

        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training

        model_to_load.load_state_dict(torch.load(file_name))
        logger.info("Model checkpoint loaded from %s", file_name)




class TokenClassifier(Transformer):
    """
    A wrapper for token classification use case based on Transformer.
    Args:
        model_name (str, optional): The pretained model name.
            Defaults to "bert-base-cased".
        num_labels (int, optional): The number of labels.
            Defaults to 2.
        cache_dir (str, optional): The default folder for saving cache files.
            Defaults to ".".
    """

    def __init__(self, model_name="bert-base-cased", num_labels=2, cache_dir="."):
        config = AutoConfig.from_pretrained(
            model_name, num_labels=num_labels, cache_dir=cache_dir
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, cache_dir=cache_dir, config=config, output_loading_info=False
        )
        super().__init__(model_name=model_name, model=model, cache_dir=cache_dir)

    def fit(
        self,
        train_dataloader,
        num_epochs=1,
        max_steps=-1,
        gradient_accumulation_steps=1,
        num_gpus=None,
        gpu_ids=None,
        local_rank=-1,
        weight_decay=0.0,
        learning_rate=5e-5,
        adam_epsilon=1e-8,
        warmup_steps=0,
        fp16=False,
        fp16_opt_level="O1",
        checkpoint_state_dict=None,
        verbose=True,
        seed=None,
    ):
        """
        Fine-tunes a pre-trained sequence classification model.
        Args:
            train_dataloader (Dataloader): A PyTorch DataLoader to be used for training.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            max_steps (int, optional): Total number of training steps.
                If set to a positive value, it overrides num_epochs.
                Otherwise, it's determined by the dataset length,
                gradient_accumulation_steps, and num_epochs.
                Defualts to -1.
            gradient_accumulation_steps (int, optional): Number of steps to accumulate
                before performing a backward/update pass.
                Default to 1.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used.
                If set to 0 or GPUs are not available, CPU device will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            local_rank (int, optional): Local_rank for distributed training on GPUs.
                Defaults to -1, which means non-distributed training.
            weight_decay (float, optional): Weight decay to apply after each
                parameter update.
                Defaults to 0.0.
            learning_rate (float, optional):  Learning rate of the AdamW optimizer.
                Defaults to 5e-5.
            adam_epsilon (float, optional): Epsilon of the AdamW optimizer.
                Defaults to 1e-8.
            warmup_steps (int, optional): Number of steps taken to increase learning
                rate from 0 to `learning rate`. Defaults to 0.
            fp16 (bool): Whether to use 16-bit mixed precision through Apex
                Defaults to False
            fp16_opt_level (str): Apex AMP optimization level for fp16.
                One of in ['O0', 'O1', 'O2', and 'O3']
                See https://nvidia.github.io/apex/amp.html"
                Defaults to "01"
            checkpoint_state_dict (dict): Checkpoint states of model and optimizer.
                If specified, the model and optimizer's parameters are loaded using
                checkpoint_state_dict["model"] and checkpoint_state_dict["optimizer"]
                Defaults to None.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
            seed (int, optional): Random seed used to improve reproducibility.
                Defaults to None.
        """

        # init device and optimizer
        device, num_gpus, amp = self.prepare_model_and_optimizer(
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
            local_rank=local_rank,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            fp16=fp16,
            fp16_opt_level=fp16_opt_level,
            checkpoint_state_dict=checkpoint_state_dict,
        )

        # compute the max number of training steps
        max_steps = compute_training_steps(
            dataloader=train_dataloader,
            num_epochs=num_epochs,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # init scheduler
        scheduler = Transformer.get_default_scheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

        # fine tune
        super().fine_tune(
            train_dataloader=train_dataloader,
            get_inputs=TokenClassificationProcessor.get_inputs,
            device=device,
            num_gpus=num_gpus,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=self.optimizer,
            scheduler=scheduler,
            fp16=fp16,
            amp=amp,
            local_rank=local_rank,
            verbose=verbose,
            seed=seed,
        )

    def predict(self, test_dataloader, num_gpus=None, gpu_ids=None, verbose=True):
        """
        Scores a dataset using a fine-tuned model and a given dataloader.
        Args:
            test_dataloader (DataLoader): DataLoader for scoring the data.
            num_gpus (int, optional): The number of GPUs to use.
                If None, all available GPUs will be used. If set to 0 or GPUs are
                not available, CPU device will be used.
                Defaults to None.
            gpu_ids (list): List of GPU IDs to be used.
                If set to None, the first num_gpus GPUs will be used.
                Defaults to None.
            verbose (bool, optional): Whether to print out the training log.
                Defaults to True.
        Returns
            1darray: numpy array of predicted label indices.
        """

        preds = list(
            super().predict(
                eval_dataloader=test_dataloader,
                get_inputs=TokenClassificationProcessor.get_inputs,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                verbose=verbose,
            )
        )
        preds = np.concatenate(preds)
        return preds

    def get_predicted_token_labels(self, predictions, label_map, dataset):
        """
        Post-process the raw prediction values and get the class label for each token.
        Args:
            predictions (ndarray): A numpy ndarray produced from the `predict`
                function call. The shape of the ndarray is:
                [number_of_examples, sequence_length, number_of_labels].
            label_map (dict): A dictionary object to map a label (str) to an ID (int).
                dataset (TensorDataset): The TensorDataset for evaluation.
            dataset (Dataset): The test Dataset instance.
        Returns:
            list: A list of lists. The size of the retured list is the number of
                testing samples.
            Each sublist represents the predicted label for each token.
        """

        num_samples = len(dataset.tensors[0])
        if num_samples != predictions.shape[0]:
            raise ValueError(
                "Predictions have {0} samples, but got {1} samples in dataset".format(
                    predictions.shape[0], num_samples
                )
            )

        label_id2str = {v: k for k, v in label_map.items()}
        attention_mask_all = dataset.tensors[1].data.numpy()
        trailing_mask_all = dataset.tensors[2].data.numpy()
        seq_len = len(trailing_mask_all[0])
        labels = []

        for idx in range(num_samples):
            seq_probs = predictions[idx]
            attention_mask = attention_mask_all[idx]
            trailing_mask = trailing_mask_all[idx]
            one_sample = []

            for sid in range(seq_len):
                if attention_mask[sid] == 0:
                    break

                if not bool(trailing_mask[sid]):
                    continue

                label_id = seq_probs[sid].argmax()
                one_sample.append(label_id2str[label_id])
            labels.append(one_sample)
        return labels

    def get_true_test_labels(self, label_map, dataset):
        """
        Get the true testing label values.
        Args:
            label_map (dict): A dictionary object to map a label (str) to an ID (int).
                dataset (TensorDataset): The TensorDataset for evaluation.
            dataset (Dataset): The test Dataset instance.
        Returns:
            list: A list of lists. The size of the retured list is the number
                of testing samples.
            Each sublist represents the predicted label for each token.
        """

        num_samples = len(dataset.tensors[0])
        label_id2str = {v: k for k, v in label_map.items()}
        attention_mask_all = dataset.tensors[1].data.numpy()
        trailing_mask_all = dataset.tensors[2].data.numpy()
        label_ids_all = dataset.tensors[3].data.numpy()
        seq_len = len(trailing_mask_all[0])
        labels = []

        for idx in range(num_samples):
            attention_mask = attention_mask_all[idx]
            trailing_mask = trailing_mask_all[idx]
            label_ids = label_ids_all[idx]
            one_sample = []

            for sid in range(seq_len):
                if attention_mask[sid] == 0:
                    break

                if not trailing_mask[sid]:
                    continue

                label_id = label_ids[sid]
                one_sample.append(label_id2str[label_id])
            labels.append(one_sample)
        return labels


def get_device(num_gpus=None, gpu_ids=None, local_rank=-1):
    if gpu_ids is not None:
        num_gpus = len(gpu_ids)
    if local_rank == -1:
        num_gpus = (
            min(num_gpus, torch.cuda.device_count())
            if num_gpus is not None
            else torch.cuda.device_count()
        )
        device = torch.device(
            "cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
        )
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        num_gpus = 1
    return device, num_gpus


def get_amp(fp16):
    """This function ensures that fp16 execution of torch.einsum is enabled
        if fp16 is set. Otherwise, it'll default to "promote" mode,
        where the operations are in fp32.
        Note that setting `fp16_opt_level="O2"` will remove the need for this code.
    """
    # Before we do anything with models, we want to
    if fp16:
        try:
            from apex import amp

            amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex"
            )
    else:
        amp = None
    return amp


def move_model_to_device(model, device):
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    # unwrap model
    # if isinstance(model, torch.nn.DataParallel):
    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    # move to device
    return model.to(device)


def parallelize_model(model, device, num_gpus=None, gpu_ids=None, local_rank=-1):
    """Moves a model to the specified device (cpu or gpu/s)
       and implements data parallelism when multiple gpus are specified.
    Args:
        model (Module): A PyTorch model.
        device (torch.device): A PyTorch device.
        num_gpus (int): The number of GPUs to be used.
            If set to None, all available GPUs will be used.
            Defaults to None.
        gpu_ids (list): List of GPU IDs to be used.
            If None, the first num_gpus GPUs will be used.
            If not None, overrides num_gpus. if gpu_ids is an empty list
            or there is no valid gpu devices are specified,
            and device is "cuda", model will not be moved or parallelized.
            Defaults to None.
        local_rank (int): Local GPU ID within a node. Used in distributed environments.
            If not -1, num_gpus and gpu_ids are ignored.
            Defaults to -1.
    Returns:
        Module, DataParallel, DistributedDataParallel: A PyTorch Module or
            a DataParallel/DistributedDataParallel wrapper,
            when one or multiple gpus are used.
    """
    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    model_module = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        if device.type == "cuda":
            if num_gpus is not None:
                if num_gpus < 1:
                    raise ValueError("num_gpus must be at least 1 or None")
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices < 1:
                raise Exception("CUDA devices are not available.")
            if gpu_ids is None:
                num_gpus = (
                    num_cuda_devices
                    if num_gpus is None
                    else min(num_gpus, num_cuda_devices)
                )
                gpu_ids = list(range(num_gpus))
            else:
                gpu_ids = list(set(list(range(num_cuda_devices))).intersection(gpu_ids))
            if len(gpu_ids) > 0:
                model = torch.nn.DataParallel(model_module, device_ids=gpu_ids)
    return model


def compute_training_steps(
    dataloader, num_epochs=1, max_steps=-1, gradient_accumulation_steps=1
):
    """Computes the max training steps given a dataloader.
    Args:
        dataloader (Dataloader): A PyTorch DataLoader.
        num_epochs (int, optional): Number of training epochs. Defaults to 1.
        max_steps (int, optional): Total number of training steps.
            If set to a positive value, it overrides num_epochs.
            Otherwise, it's determined by the dataset length,
            gradient_accumulation_steps, and num_epochs.
            Defaults to -1.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate
            before performing a backward/update pass.
            Default to 1.
    Returns:
        int: The max number of steps to be used in a training loop.
    """
    try:
        dataset_length = len(dataloader)
    except Exception:
        dataset_length = -1
    if max_steps <= 0:
        if dataset_length != -1 and num_epochs > 0:
            max_steps = dataset_length // gradient_accumulation_steps * num_epochs
    if max_steps <= 0:
        raise Exception("Max steps cannot be determined.")
    return max_steps
