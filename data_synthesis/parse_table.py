import os
import re
import json
import random
import argparse
import pickle
import csv
from template_config import *
from collections import defaultdict

def reshape(v):
    res = [[] for _ in range(len(v[0]))]
    for i in range(len(v[0])):
        for j in range(len(v)):
            res[i].append(v[j][i])
    return res
def getType(v):
    res = []
    for r in v:
        if sum((cell.isdigit() for cell in r)) > 0:
            ctype = "real"
        else:
            ctype = "text"
        res.append(ctype)
    return res
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def check_name(inpStr):
    return len(inpStr) > 1 and "-" not in inpStr and not hasNumbers(inpStr)

def gen_name(title, must_have=False):
    title_tokens = title.split(" ")
    qualify_words = []
    for w in title_tokens:
        if check_name(w):
            qualify_words.append(w)

    if random.random() < 0.4:
        name = " ".join(qualify_words[-2:])
    else:
        name = " ".join(qualify_words[-1:])

    if name != "":
        return name

    if must_have:
        return title_tokens[0]
    else:
        return name
# Read and parse wikitable question
def read_wtq_table(PATH):
    # header format for tagged file:
    # row, col, id, content, tokens, lemmaTokens, posTags, nerTags, nerValues, number, date, num2, list, listId
    all_table = []
    for csv_file in range(200, 205):
        tagged_path = PATH+str(csv_file)+"-tagged/"
        page_path = PATH+str(csv_file)+"-page/"
        for i in range(1000):
            try:
                table = {}
                skip = False
                with open(page_path+str(i)+".json", encoding="utf-8") as f:
                    data = json.load(f)
                table_title = gen_name(data['title'])
                if table_title=="":
                    continue
                f = open(tagged_path+str(i)+".tagged", encoding="utf-8")
                reader = csv.reader(f, delimiter='\t')
                columns = []
                values = [[] for _ in range(3)]
                for row in reader:
                    # header
                    if row[0]=='row':
                        continue
                    # actual header
                    index = int(row[0])
                    if index ==-1:
                        lemmaToken = row[5].replace('|', ' ')
                        columns.append(lemmaToken)
                    elif index <= 2:
                        # first three rows of values
                        lemmaToken = row[5].replace('|', ' ')
                        values[index].append(lemmaToken)
                    else:
                        break
                values = reshape(values)
                table["values"] = [["all" for _ in range(3)]] + values
                table["column_types"] = ["text"] + getType(values)
                table["name"] = table_title
                title_prefix = '_'.join(table_title.split(' '))
                table["columns"] = [title_prefix+" "+ hd for hd in [table_title+" *"]+columns]
                table["columns_original"] = ["*"] + columns
                f.close()
                all_table.append(table)
            except:
                pass
    return all_table
# Read and parse wikitable
def read_wt_table(train_corpus):
    total_count = 0
    webtables = []
    with open(train_corpus, "r", encoding="utf-8") as f:
        for line in f:
            skip = False
            tokens = line.lower().replace("<special7>", "<tabn>").replace("<special8>", "<coln>").replace("<special9>", "<entry>").replace("*", "").replace("|||", "")
            table = {"columns": [], "values": [], "columns_original": [], "column_types": []}
            chunks = tokens.split(" <coln> ")
            for chunk in chunks:
                if "<tabn>" in chunk:
                    page_title = chunk.replace("<tabn>", "").strip()
                    table_name = gen_name(page_title)
                    table["name"] = table_name
                    if table_name == "" or len(table_name) < 2:
                        skip = True
                else:
                    assert "<entry>" in chunk
                    chunk_toks = chunk.split(" <entry> ")
                    if len(chunk_toks) == 2:
                        col_name, entry = chunk_toks[0].strip(), chunk_toks[1].strip()
                        if len(col_name) > 1:
                            table["columns"].append(" ".join(col_name.split(" ")[:5]))
                            table["columns_original"].append(col_name)
                            ctype = "text"
                            if entry.isdigit():
                                ctype = "real"
                            table["column_types"].append(ctype)
                            table["values"].append(" ".join(entry.split(" ")[:5]))


            if len(table["columns"]) < 3:
                skip = True
            if not skip:
                table_name = table["name"]
                table["columns"] = [table_name + " *"] + table["columns"]
                table["columns_original"] = ["*"] + table["columns_original"]
                table["column_types"] = ["text"] + table["column_types"]
                table["values"] = ["all"] + table["values"]
                tabn_str = "_".join(table_name.split(" "))
                table["columns"] = [tabn_str +" "+ hd for hd in table["columns"]]
                if "*" not in table['columns'][0]:
                    print(table['columns'])
                webtables.append(table)
            total_count += 1
    return webtables
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Please provide one of the following data set name\n wikitablequestion or wikitable")
    parser.add_argument("--path", help="Please provide the data set path")
    parser.add_argument("--output", help="Please provide the output path")

    args = parser.parse_args()
    if args.dataset == "wikitablequestion":
        if args.path is None:
            path = "data/csv/"
        else:
            path = args.path
        result = read_wtq_table(path)
    elif args.dataset == "wikitable":
        if args.path is None:
            path = "data/wikitable_dup1_row1.txt"
        else:
            path = args.path
        result = read_wt_table(path)
    else:
        print("Unsupported dataset")
        exit(0)
    if args.output is None:
        output = "temp/"+args.dataset+"_processed.json"
    else:
        output = args.output
    with open(output, "w") as f:
        json.dump(result, f)
    print("Finish writing to "+output)
