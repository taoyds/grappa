import argparse
import torch
import numpy as np 
import argparse
import json
from collections import defaultdict

from allennlp.data import Vocabulary
from wikitable_roberta.reader.util import load_jsonl, load_jsonl_table
import pdb

arg_parser = argparse.ArgumentParser(description="WikiTable dataset reader.")
arg_parser.add_argument('-train_file', help="file path of training file")
arg_parser.add_argument('-dev_file', help="file path of dev file")
arg_parser.add_argument('-test_file', help="file path of test file")
arg_parser.add_argument('-table_file', help="file path of table file")
arg_parser.add_argument('-tagged_dir', help="file path of tagged table file")

arg_parser.add_argument('-embed_file', help='file path of pretrained embeddings')
arg_parser.add_argument('-output_file', help='output file path')



class WTReader():
    """
    Generate vocabulary and its corresponding embed files
    """
    def __init__(self, tables, train_examples, dev_examples, test_examples, embed_file):
        self.table_dict = tables
        self.train_examples = train_examples
        self.dev_examples = dev_examples
        self.test_examples = test_examples
        self.embed_file = embed_file
    
    def process(self):
        raise NotImplementedError
    
    def gen_glove(self):
        token2id = self.vocab.get_token_to_index_vocabulary()
        vocab_dic = {}
        with open(self.embed_file) as f:
            for line in f:
                s_s = line.split(" ")
                if s_s[0] in token2id:
                    vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])

        ret_mat = []
        unk_counter = 0
        for i in range(self.vocab.get_vocab_size()):
            token = self.vocab.get_token_from_index(i)
            # token = token.lower()
            if token in vocab_dic:
                ret_mat.append(vocab_dic[token])
            elif i == 0:
                assert token == "@@PADDING@@"
                ret_mat.append(np.random.normal(-0.01, 0.01, 300).astype("float32"))
                unk_counter += 1
            elif i == 1:
                assert token == "@@UNKNOWN@@"
                ret_mat.append(np.random.normal(-0.01, 0.01, 300).astype("float32"))
                unk_counter += 1
        ret_mat = np.array(ret_mat)
        print("{0} unk out of {1} vocab".format(unk_counter, self.vocab.get_vocab_size()))
        self.wordvec = ret_mat
        return ret_mat

    def gen_vocab(self):
        pos_set = set()
        counter = defaultdict(int)
        for example in self.train_examples + self.dev_examples + self.test_examples:
            # tokens = example["tmp_tokens"]
            tokens = example["tokens"]
            for token in tokens:
                counter[token] += 1
            
            pos = example["pos_tags"]
            for _p in pos:
                pos_set.add(_p)
        
        self.id2pos = list(pos_set)
        self.pos2id = {v:k for k,v in enumerate(self.id2pos)}
        print(f"POS numbers: {len(self.id2pos)}")

        column_name_set = set()
        for table_name, table in self.table_dict.items():
            raw_table_names = set(table["props"]) - \
                set(table["datetime_props"]) - set(table["num_props"])
            for col_name in raw_table_names:
                real_name = col_name[2:] # r.
                real_name = "-".join(real_name.split("-")[:-1]) # remove type
                col_tokens = real_name.split("_")
                for token in col_tokens:
                    counter[token] += 1
        print(f"Number of raws tokens, {len(counter)}")
        self.vocab = Vocabulary(counter={"tokens":counter}, min_count={"tokens":2}, 
                    max_vocab_size={"tokens":50000}, pretrained_files={"tokens":self.embed_file}, 
                    only_include_pretrained_words=True)


    def check(self):
        for example in self.train_examples + self.dev_examples + self.test_examples:
            # check context
            if example["context"] not in self.table_dict:
                print("Context not found!")
                pdb.set_trace()
            # keep the raw tokens
            if "<DATE>" in example["tokens"] or "<NUMBER>" in example["tokens"]:
                pdb.set_trace()
        
        # check rows
        for table_name in self.table_dict:
            table = self.table_dict[table_name]
            lens = [len(table["kg"][row]) for row in table["row_ents"]]
            if [lens[0]] * len(lens) != lens:
                # print("Abnormal rows")
                pass


if __name__ == "__main__":
    args = arg_parser.parse_args()

    tables = load_jsonl_table(args.table_file)
    train_examples = load_jsonl(args.train_file)
    dev_examples = load_jsonl(args.dev_file)
    test_examples = load_jsonl(args.test_file)

    wt_reader = WTReader(tables, train_examples, dev_examples, test_examples, args.embed_file)
    wt_reader.gen_vocab()
    wt_reader.gen_glove()
    
    wt_reader.check()
    #print("Vocab size: ", len(wt_reader.id2qtk))

    import pickle
    with open(args.output_file, "wb") as f:
        pickle.dump(wt_reader, f)

