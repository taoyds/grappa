import numpy as np
import time
import torch

from allennlp.data.tokenizers.token import Token
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

DATA_DIR_RAW = Path("../dataset/wikitable/raw_input/WikiTableQuestions/tagged")


def check_example(example, example_dict, is_training):

    if is_training:
        # if it does not trigger any programs, then no need to train it
        if (example["id"], example["context"]) not in example_dict:
            return False

        # if the sentence is too long, alignment model will take up too much time
        if len(example["tokens"]) > 30:
            return False

        target_lfs = example_dict[(example["id"], example["context"])]
        table_id = example["context"]
        # table_lines = tables[table_id]["raw_lines"]
        table_filename = DATA_DIR_RAW / f"{table_id.split('_')[1]}-tagged" / f"{table_id.split('_')[2]}.tagged"

        target_value, target_can = example["answer"] # (targeValue, targetCan)
        tokenized_question = [ Token(token, pos_=pos) for token,pos in  zip(example["tokens"], example["pos_tags"])]
        if len(tokenized_question) == 1: return False # ignore the single-token one

        return True

    else:
        if (example["id"], example["context"]) not in example_dict:
            return False
        else:
            return True


def get_dataloader(examples, config, rank, is_training):

    if config.nodes != 0 and is_training:
        sampler = torch.utils.data.distributed.DistributedSampler(
            examples,
            num_replicas=config.world_size,
            rank=rank,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=examples,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
            collate_fn=lambda x: x,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=examples,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
    return dataloader

class WTBDataset(Dataset):

    def __init__(self, examples, example_dict, mode):
        self.examples = examples
        self.example_dict = example_dict
        self.mode = mode
        self.dev_dummy = torch.zeros(len(self.examples), len(self.examples))
        self.test_dummy = torch.zeros(len(self.examples), len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        if self.mode == "train":
            return [b[index] for b in [self.examples]]
        elif self.mode == "dev":
            return [b[index] for b in [self.examples, self.dev_dummy]]
        elif self.mode == "test":
            return [b[index] for b in  [self.examples, self.dev_dummy, self.test_dummy]]

class WTBDataLoader(DataLoader):

    def __init__(self, examples, example_dict, config, rank, mode):

        dataset = WTBDataset(
            examples,
            example_dict,
            mode,
        )

        if config.nodes != 0 and rank is not None:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=config.world_size,
                rank=rank,
            )
            super(WTBDataLoader, self).__init__(
                dataset=dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler,
                collate_fn=lambda x: x,
            )
        else:
            super(WTBDataLoader, self).__init__(
                dataset=dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
