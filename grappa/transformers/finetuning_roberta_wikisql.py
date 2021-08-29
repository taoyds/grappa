# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import wandb

from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers.modeling_roberta_wikisql import RobertaForMaskedLM
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
SEP_TOKEN = '</s>'
CLS_TOKEN = '<s>'
MASK_TOKEN='<mask>'
UNK_TOKEN="<unk>"


def simple_accuracy(preds, labels):
    right = 0
    total = 0
    preds = preds.tolist()
    labels = labels.tolist()
    for pds, lbs in zip(preds, labels):
        right_one = 0
        total_one = 0
        for pd_one, lb_one in zip(pds, lbs):
            total += 1
            if lb_one != -1:
                total_one += 1
                if pd_one == lb_one:
                    right_one += 1

            if total_one == right_one:
                right += 1

    return float(right) / total


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, mlm_loss=False):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.mlm_loss = mlm_loss

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with open(corpus_path, "r", encoding=encoding) as f:
                for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        #remove last added sample because there won't be a subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        #store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in f:
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        line = self.random_sent(item)
        q_label = line.split("|||")
        assert len(q_label) == 2

        t1, label_str = q_label[0].strip(), q_label[1].strip()
        raw_label = [int(i) for i in label_str.split(" ")]
        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, raw_label=raw_label)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.mlm_loss)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.col_label_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.q_tab_inds))

        return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        assert len(t1) > 0

        return t1

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "" :
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                # skip empty rows that are used for separating documents and keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    self.current_doc = self.current_doc+1
            self.line_buffer = t2

        assert t1 != "" and "|||" in t1
        return t1, t2


    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, raw_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.raw_label = raw_label  # col labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, col_label_ids, lm_label_ids, q_tab_inds):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.col_label_ids = col_label_ids
        self.lm_label_ids = lm_label_ids
        self.q_tab_inds = q_tab_inds


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if token != CLS_TOKEN and token != SEP_TOKEN and token != SEP_TOKEN and prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = MASK_TOKEN

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.encoder.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.encoder[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.encoder[UNK_TOKEN])
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length, tokenizer, mlm_loss):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens_a = example.tokens_a[:max_seq_length]
    raw_label = example.raw_label
    # CHECK: </s> appears
    col_ids = [i for i, x in enumerate(tokens_a) if x == SEP_TOKEN][:-1]
    if len(col_ids) != len(raw_label):
        print("tokens_a: ", tokens_a)
        print("raw_label: ", raw_label)
    assert len(col_ids) == len(raw_label)

    col_label_ids = [-1] * len(tokens_a)
    for cid, clb in zip(col_ids, raw_label):
        col_label_ids[cid] = clb
        
    #sep token 
    q_tab_tfs = [1 if x == SEP_TOKEN or x == CLS_TOKEN else 2 if x == "*" else 0 for x in tokens_a]

    tab_inds = [0]*len(tokens_a)
    is_next_table = False
    for i in range(len(q_tab_tfs)-1, 0, -1):
        cur_tf = q_tab_tfs[i]
        if cur_tf == 2:
            is_next_table = True
        elif is_next_table and cur_tf == 1:
            tab_inds[i] = 1
            is_next_table = False

    q_tab_inds = []
    cur_idx = 0
    for idx, tf in enumerate(tab_inds):
        if tf == 1:
            cur_idx = idx
        q_tab_inds.append(cur_idx)

    #mlm loss
    tokens_b, lm_label_ids = random_word(tokens_a, tokenizer)
    if mlm_loss:
        tokens_a = tokens_b
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # TODO: check if this case in roberta
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(1) # roberta pad index is 1
        input_mask.append(0)
        col_label_ids.append(-1)
        lm_label_ids.append(-1)
        q_tab_inds.append(0)

    assert len(input_ids) == max_seq_length
    assert len(q_tab_inds) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    assert len(col_label_ids) == max_seq_length

    if example.guid < 5:
        print("If using MLM loss: ", mlm_loss)
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("col_label_ids: %s" % " ".join([str(x) for x in col_label_ids]))
        print("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))
        print("q_tab_inds: %s" % " ".join([str(x) for x in q_tab_inds]))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             col_label_ids=col_label_ids,
                             lm_label_ids=lm_label_ids,
                             q_tab_inds=q_tab_inds)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus, each line contains numbers that are the roberta tokenized indices.")
    parser.add_argument("--train_eval_corpus",
                        default=None,
                        type=str,
                        required=False,
                        help="The input train eval corpus, each line contains numbers that are the roberta tokenized indices.")
    parser.add_argument("--eval_corpus",
                        default=None,
                        type=str,
                        required=False,
                        help="The input eval corpus, each line contains numbers that are the roberta tokenized indices.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: roberta-base, "
                             "roberta-base, roberta-large")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=208,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run evaluation.")
    parser.add_argument("--mlm_loss",
                        action='store_true',
                        help="Whether to add mlm loss.")
    parser.add_argument("--concat_tabcol",
                        action='store_true',
                        help="Whether to concatenate table and column representations.")
    parser.add_argument("--train_batch_size",
                        default=48,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=1000.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    
    wandb.init(project="column_roberta", name=args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(init_method='env://', backend='nccl')
    print("\n=====Using device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

#     if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
#         raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
#     if not os.path.exists(args.output_dir) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
#         os.makedirs(args.output_dir)

    # NOTICE: Not using RobertaTokenizer because we already tokenized our inputs in another script.
    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_corpus)
        train_dataset = BERTDataset(args.train_corpus, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory, mlm_loss=args.mlm_loss)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    # if evaluation on dev
    if args.do_eval:
        print("Loading Eval Dataset", args.eval_corpus)
        eval_dataset = BERTDataset(args.eval_corpus, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory)
        train_eval_dataset = BERTDataset(args.train_eval_corpus, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory)
#     Prepare model
    model = RobertaForMaskedLM.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=True, find_unused_parameters=True)
#         model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
      
    wandb.watch(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        print("***** Running training *****")
        print("  Num examples:", len(train_dataset))
        print("  Batch size:", args.train_batch_size)
        print("  Num steps:", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            eval_sampler = SequentialSampler(eval_dataset) #if args.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
            train_eval_sampler = SequentialSampler(train_eval_dataset) #if args.local_rank == -1 else DistributedSampler(train_eval_dataset)
            train_eval_dataloader = DataLoader(train_eval_dataset, sampler=train_eval_sampler, batch_size=args.train_batch_size)

        for epoch in range(int(args.num_train_epochs)):
            tr_loss = 0
            best_acc = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            total_steps = len(train_dataloader)
            step_check = int(total_steps * 0.1) if int(total_steps * 0.1) != 0 else 1
            save_check = int(total_steps * 0.5) if int(total_steps * 0.5) != 0 else 1
            print("\n====================Epoch: ", epoch)
            for step, batch in enumerate(train_dataloader):
                model.train()
                input_ids, input_mask, col_label_ids, lm_label_ids, q_tab_inds = batch
                input_ids, input_mask, col_label_ids, lm_label_ids, q_tab_inds = to_device(device, input_ids, input_mask, col_label_ids, lm_label_ids, q_tab_inds)
                
                if not args.concat_tabcol:
                    q_tab_inds = None
                    
                if args.mlm_loss:
                    loss = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                masked_lm_labels=lm_label_ids,
                                masked_col_labels=col_label_ids,
                                q_tab_inds=q_tab_inds)
                else:
                    loss = model(input_ids=input_ids,
                                attention_mask=input_mask,
                                masked_col_labels=col_label_ids,
                                q_tab_inds=q_tab_inds)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                wandb.log({'batch_training_loss': loss.item()})
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if step % step_check == 0:
                    print("Finishing training for current epoch:\t" + str(round(step/total_steps, 3)))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

            epoch_loss = tr_loss / total_steps
            print("Train epoch loss:\t" + str(epoch_loss))
            wandb.log({'epoch_training_loss': epoch_loss})

            # Evaluation on dev
            if args.do_eval and args.local_rank in [-1, 0]:
                print("***** Running evaluation on train *****")
                train_eval_loss = 0.0
                train_preds = None
                train_nb_eval_steps = 0
                train_out_label_ids = None
                
                for step, batch in enumerate(train_eval_dataloader):
                    model.eval()
                    input_ids, input_mask, col_label_ids, _, q_tab_inds = batch
                    input_ids, input_mask, col_label_ids, q_tab_inds = to_device(device, input_ids, input_mask, col_label_ids, q_tab_inds)
                    
                    if not args.concat_tabcol:
                        q_tab_inds = None
                    
                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids=input_ids,
                                                    attention_mask=input_mask,
                                                    masked_col_labels=col_label_ids,
                                                    q_tab_inds=q_tab_inds,
                                                    is_train=False)
                        col_logits = logits[1]
                        train_eval_loss += tmp_eval_loss.mean().item()

                    train_nb_eval_steps += 1
                    if train_preds is None:
                        train_preds = col_logits.detach().cpu().numpy()
                        train_out_label_ids = col_label_ids.detach().cpu().numpy()
                    else:
                        train_preds = np.append(train_preds, col_logits.detach().cpu().numpy(), axis=0)
                        train_out_label_ids = np.append(train_out_label_ids, col_label_ids.detach().cpu().numpy(), axis=0)

                train_eval_loss = train_eval_loss / train_nb_eval_steps
                train_preds = np.argmax(train_preds, axis=2)
                train_cur_acc = simple_accuracy(train_preds, train_out_label_ids)
                print("Train eval epoch loss:\t" + str(train_eval_loss))
                print("Train eval accuracy:\t" + str(train_cur_acc))
                wandb.log({'epoch_train_eval_loss': train_eval_loss})
                wandb.log({'train_eval_accuracy': train_cur_acc})

                print("***** Running evaluation on dev *****")
                eval_loss = 0.0
                preds = None
                nb_eval_steps = 0
                out_label_ids = None
                for step, batch in enumerate(eval_dataloader):
                    model.eval()
                    input_ids, input_mask, col_label_ids, _, q_tab_inds = batch
                    input_ids, input_mask, col_label_ids, q_tab_inds = to_device(device, input_ids, input_mask, col_label_ids, q_tab_inds)
                    
                    if not args.concat_tabcol:
                        q_tab_inds = None
                    
                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids=input_ids,
                                                    attention_mask=input_mask,
                                                    masked_col_labels=col_label_ids,
                                                    q_tab_inds=q_tab_inds,
                                                    is_train=False)
                        col_logits = logits[1]
                        eval_loss += tmp_eval_loss.mean().item()

                    nb_eval_steps += 1
                    if preds is None:
                        preds = col_logits.detach().cpu().numpy()
                        out_label_ids = col_label_ids.detach().cpu().numpy()
                    else:
                        preds = np.append(preds, col_logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, col_label_ids.detach().cpu().numpy(), axis=0)

                eval_loss = eval_loss / nb_eval_steps
                preds = np.argmax(preds, axis=2)
                cur_acc = simple_accuracy(preds, out_label_ids)
                print("Eval epoch loss:\t" + str(eval_loss))
                print("==============Eval accuracy:\t" + str(cur_acc))
                wandb.log({'dev_eval_accuracy': cur_acc})
                wandb.log({'epoch_dev_eval_loss': eval_loss})

                if cur_acc > best_acc:
                    best_acc = cur_acc
                    print("** ** * Saving fine-tuned model for epoch  " + str(epoch))
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    if args.do_train:
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                if epoch % 15 == 0:
                    # Save a trained model
                    print("** ** * Saving fine-tuned model for epoch  " + str(epoch))
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME.replace(".bin", "_"+str(epoch)+".bin"))
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME.replace(".json", "_"+str(epoch)+".json"))
                    if args.do_train:
                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)


def to_device(device, *args):
    """
    Move tensors to device.
    """
    return [None if x is None else x.to(device) for x in args]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
