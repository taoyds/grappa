import os, json
import codecs
import pickle
import argparse
import random
from transformers.tokenization_roberta import RobertaTokenizer

SEP_TOKEN = "</s>"
AGG_MAP = {0: '', 1: 'max', 2: 'min', 3: 'count', 4: 'sum', 5: 'average'}
COND_MAP = {0: 'equals', 1: 'is larger than', 2: 'is smaller than', 3: 'op'}
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_TOKEN_LEN = 190

def load_wikisql_data(path_wikisql, mode="train"):
    path_sql = os.path.join(path_wikisql, mode+'_tok.jsonl')
    path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1

#     data = data[:10000]

    return data, table


def load_dataset(path_wikisql, add_dev_test):
    print("Loading from WikiSQL dataset")
    train_sql_data, train_table_data = load_wikisql_data(path_wikisql)
    if add_dev_test:
        dev_sql_data, dev_table_data = load_wikisql_data(path_wikisql, "dev")
        test_sql_data, test_table_data = load_wikisql_data(path_wikisql, "test")
        sql_data = train_sql_data + dev_sql_data + test_sql_data
        table_data = {**train_table_data, **dev_table_data, **test_table_data}
    else:
        sql_data, table_data = train_sql_data, train_table_data
        
    return sql_data, table_data

def convert_label(sql_label, col_num):
    """"sql":{"sel":3,"conds":[[5,0,"Butler CC (KS)"]],"agg":0}"""
    sel_col = sql_label['sel']
    sel_agg = sql_label['agg']
    conds = sql_label['conds']
    conds = {cond[0]:cond[1] for cond in conds}
    col_labels = []
    for i in range(col_num):
        col_label = 0
        if i == sel_col:
            col_label = col_label + 1 + sel_agg
        elif i in conds.keys():
            col_label = 7 + conds[i]
        col_labels.append(col_label)
    label_str = " ".join([str(i) for i in col_labels])

    return label_str, col_labels

def generate_sql_str(sql_label, cols):
    """"sql":{"sel":3,"conds":[[5,0,"Butler CC (KS)"]],"agg":0}"""
    sel_col = sql_label['sel']
    sel_agg = sql_label['agg']
    conds = sql_label['conds']
    
    if AGG_MAP[sel_agg] != '':
        select_str = "select " + AGG_MAP[sel_agg] + " " + cols[sel_col]
    else:
        select_str = "select " + cols[sel_col]
        
    cond_strs = []
    for cond in conds:
        cond_col, cond_op = cond[0], cond[1]
        if type(cond[2]) == str:
            prob = random.random()
            cond_val = cond[2]
            if prob < 0.5:
                cond_val = '''"''' + cond[2] + '''"'''                
        else:
            cond_val = str(cond[2])
        cond_str_one = cols[cond_col] + " " + COND_MAP[cond_op] + " " + cond_val
        cond_strs.append(cond_str_one)
    cond_str = " and ".join(cond_strs)
    
    if len(cond_strs) > 0:
        sql_str = select_str + " where " + cond_str
    else:
        sql_str = select_str

    return sql_str

def write_to_file(sql_data, tables, output_file, replace_q_sql, add_values):
    table_file = codecs.open(output_file, "w", "utf-8")
    valid_count = 0
    num_sql = len(sql_data)
    check_point = int(num_sql*0.01)
    max_col_num = 0
    unique_labels = set()
    skip_count = 0
    for tn, sql_one in enumerate(sql_data):
        if tn % check_point == 0:
            print("processed: ", str(round(tn/num_sql, 2)))
        columns = tables[sql_one['table_id']]["header"]
        columns = [col if len(col.split(" ")) < 10 else " ".join(col.split(" ")[:5]) for col in columns]
        if add_values:
            rows = tables[sql_one['table_id']]["rows"]
            row_len = len(rows)
            if row_len > 0:
                rid = random.randint(0, row_len-1)
                row_selected = rows[rid]
                columns = [col + " " + " ".join(str(val).split(" ")[:4]) for col, val in zip(columns , row_selected)]
        question = sql_one['question']
        sql_str = generate_sql_str(sql_one['sql'], columns)
        cur_col_num = len(columns)
        label_str, col_labels = convert_label(sql_one['sql'], cur_col_num)
        unique_labels.update(col_labels)
        if cur_col_num > max_col_num:
            max_col_num = cur_col_num
#         if cur_col_num > 25:
#             continue
        #add page title and caption
        if replace_q_sql:
            question = sql_str
        q_col_str = "<s> " + question + " </s> " + " </s> ".join(columns).strip() + " </s> "
        caption = q_col_str + " ||| " + label_str
        tokens = tokenizer.tokenize(q_col_str)
        if len(tokens) > MAX_TOKEN_LEN:
            skip_count += 1
            continue
        valid_count += 1
        table_file.write(caption.strip().replace("\n", ""))
        #add column names in another new line
        table_file.write("\n")

    table_file.close()

    return valid_count, max_col_num, unique_labels, skip_count

def main(path_wikisql, output_file, replace_q_sql=False, add_dev_test=False, add_values=False):
    sql_data, table_data = load_dataset(path_wikisql, add_dev_test)

    valid_count, max_col_num, unique_labels, skip_count = write_to_file(sql_data, table_data, output_file, replace_q_sql, add_values)
    print('\nFile writing done for all tables!!! Total number of tables: ', valid_count)
    print('\nskiped example number: ', skip_count)
    print('\nMax number of columns in all tables: ', max_col_num)
    print('\nunique labels: ', unique_labels)

if __name__ == "__main__":
    main("../data/wikisql", "data/wikisql_input_all_add_dev_test.txt", add_dev_test=True)
