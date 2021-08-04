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
import codecs
from tqdm import tqdm, trange

from pytorch_transformers.tokenization_roberta import RobertaTokenizer

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def main(train_corpus, bert_model, output_file):
    table_file = codecs.open(output_file, "w", "utf-8")
    tokenizer = RobertaTokenizer.from_pretrained(bert_model)
    total_count = 0
    with open(train_corpus, "r", encoding="utf-8") as f:
        for ln in f:
            if total_count % 100000 == 0:
                print("processed: ", total_count)
            line = ln.replace("@@ ", "").strip()
            line = line.replace("<special7>", "<s>").replace("<special8>", "</s>").replace("<special9>", "<unk>")
            tokens = tokenizer.tokenize(line)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids_str = " ".join([str(ti) for ti in token_ids])
            
            print(tokenizer.encoder["<mask>"])
            
            table_file.write(token_ids_str)
            table_file.write("\n\n")
            total_count += 1
            
    table_file.close() 

if __name__ == "__main__":
#     train_corpus = "/export/home/table_xlm/data/comb/processed/use_pretrained/data_comb_col_only/train.en"
#     bert_model = "roberta-base"
#     output_file = "data/data_comb_col_only_train_use_pretrained.txt"
    train_corpus = "data/wikitable_for_debug.txt"
    bert_model = "roberta-base"
    output_file = "data/wikitable_for_debug_ids.txt"
    main(train_corpus, bert_model, output_file)
