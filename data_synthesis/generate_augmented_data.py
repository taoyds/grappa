import os
import re
import json
import random
import codecs
import argparse
from template_config import *
from nltk import word_tokenize
from collections import defaultdict
from transformers.tokenization_roberta import RobertaTokenizer

ADD_INDEX_ID = 0.7
ADD_INDEX_NAME = 0.3
OP_VAL_EQUAL = 0.4
USE_TABLE_1 = 0.5
USE_1_FOR_INTEGER = 0.5

SEP_TOKEN = "</s>"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_TOKEN_LEN = 150
MAX_COL_NUM = 25
OPS = ["=", ">", "<", ">=", "<=", "!=", "LIKE"]

# read NL-SQL templates
def read_NL_SQL_template(nlsql_templates_file):
    templates = []
    with open(nlsql_templates_file) as fp:
        lines = fp.readlines()
        template_one = {}
        for line in lines:
            if "\n" == line:
                templates.append(template_one)
            elif "SQL Pattern:" in line:
                template_one = {}
                sps = line.strip().replace("SQL Pattern: ", "").split("|||")
                template_one["questions"] = []
                if len(sps) == 1:
                    template_one["SQL pattern"] = sps[0]
                    template_one["SQL constraints"] = []
                elif len(sps) == 2:
                    template_one["SQL pattern"] = sps[0]
                    template_one["SQL constraints"] = [x.strip() for x in sps[1].split("|") if x != " "]
                else:
                    print("\n======Error warning!!!!")
            elif "count: " in line:
                sql_count = int(line.strip().replace("count: ", ""))
                template_one["count"] = sql_count
            elif "question:  " in line:
                sps = line.strip().replace("question:  ", "").split("|||")
                question = sps[0]
                if len(sps) == 2:
                    q_constraints = [x.strip() for x in sps[1].split("|") if x != " "]
                else:
                    q_constraints = []
                template_one["questions"].append((question, q_constraints))
    return templates
# Sieve through the templates and get valid single-table questions
def get_templates_for_one_table(templates):
    templates_one_table = []
    for template in templates:
        sql_constraints = template['SQL constraints']
        sql_pattern = template["SQL pattern"]
        questions = template["questions"]
        skip = False
        for constraint in sql_constraints:
            if "id" in constraint or "T1" in constraint:
                skip = True
        questions_after = []
        if not skip:
            for q, qc in questions:
                if "TABLE1" not in q:
                    questions_after.append((q, qc))
            if len(questions_after) > 0:
                template_one = {}
                template_one['SQL constraints'] = sql_constraints
                template_one['SQL pattern'] = sql_pattern
                template_one["questions"] = questions_after
                templates_one_table.append(template_one)
    return templates_one_table
# Read json file
def read_json(file):
    with open(file) as json_file:
        res = json.load(json_file)
    return res
# Unify and combine tables as databases
def create_dbs(tables):
    dbs = []
    cur_cols = []
    db_one = []
    ahd_cols = []
    for i, tab in enumerate(tables):
        # if i % 100000 == 0:
        #     print("processed: ", i)
        if len(db_one) <= random.choice([0, 1]) and len(ahd_cols) < MAX_COL_NUM:
            db_one.append(tab)
            cur_cols.extend([col+"."+tab["name"] for col in tab["columns"]])
            if i+1 < len(tables):
                ahd_cols = cur_cols + [col+"."+tables[i+1]["name"] for col in tables[i+1]["columns"]]
            else:
                 break
        else:
            if len(cur_cols) == len(list(set(cur_cols))):
                if len(db_one) > 1:
                    db_one_new = []
                    for tab in db_one:
                        if tab["columns"][0] == "id":
                            tab["columns"] = tab["columns"][1:]
                            tab["column_types"] = tab["column_types"][1:]
                            tab["columns_original"] = tab["columns_original"][1:]
                            tab["values"] = tab["values"][1:]

                        if random.random() < ADD_INDEX_ID:
                            index_col = "id"
                            if random.random() < ADD_INDEX_NAME:
                                index_col = "name"

                            if index_col not in tab["columns"]:
                                tabn_str = "_".join(tab["name"].split(" "))
                                tab["columns"] = [tab["columns"][0]] + [tabn_str +" "+ index_col] + tab["columns"][1:]
                                val_add = 1
                                if index_col == "name":
                                    val_add = "value"
                                tab["values"] = [tab["values"][0]] + [val_add] + tab["values"][1:]
                                tab["column_types"] = [tab["column_types"][0]] + ["text"] + tab["column_types"][1:]
                                tab["columns_original"] = [tab["columns_original"][0]] + [index_col] + tab["columns_original"][1:]
                        db_one_new.append(tab)
                    dbs.append(db_one_new)
                else:
                    dbs.append(db_one)
            db_one = []
            cur_cols = []
            ahd_cols = []

    return dbs
def get_sql_slots(sql_pattern):
    sql_tokens = sql_pattern.split(" ")
    columns = {}
    ops = {}
    values = {}
    aggs = {}
    dasc = False
    slots = []
    val_pros = []
    for i, tok in enumerate(sql_tokens):
        if "{" in tok and "}" in tok and "FROM" not in tok:
            if tok not in slots:
                slots.append(tok)

        if "AGG" in tok:
            if i + 2 < len(sql_tokens) and "(" == sql_tokens[i+1]:
                if "COLUMN" in sql_tokens[i+2]:
                    if sql_tokens[i+2] not in columns.keys():
                        columns[sql_tokens[i+2]] = ["number"]
                    else:
                        columns[sql_tokens[i+2]].append("number")
                    aggs[tok] = sql_tokens[i+2]
                else:
                    print("\nTemplate Error: AGG format is wrong!!!")
                    print(sql_pattern)
        elif "COLUMN" in tok:
            if tok not in columns.keys():
                columns[tok] = []
        elif "OP" in tok:
            if i - 1 >= 0 and "COLUMN" in sql_tokens[i-1]:
                ops[tok] = [sql_tokens[i-1]]
                if i + 1 < len(sql_tokens) and "VALUE" in sql_tokens[i+1]:
                    ops[tok].append(sql_tokens[i+1])
                    val_pros.append(sql_tokens[i+1])
            elif i - 2 >= 0 and ")" == sql_tokens[i-1] and ("COLUMN" in sql_tokens[i-2] or "*" == sql_tokens[i-2]):
                ops[tok] = [sql_tokens[i-2]]
                if i + 1 < len(sql_tokens) and "VALUE" in sql_tokens[i+1]:
                    ops[tok].append(sql_tokens[i+1])
                    val_pros.append(sql_tokens[i+1])
            else:
                print("\nTemplate Error: OP format is wrong!!!")
                print(sql_pattern)
        elif "VALUE" in tok and tok not in val_pros:
            """
            OP} {VALUE0}
            LIMIT {VALUE0}
            {COLUMN1} BETWEEN {VALUE0} AND {VALUE1}
            HAVING COUNT ( * ) {OP1} {VALUE1}
            = {VALUE1}
            """
            if i - 2 >= 0 and ("BETWEEN" == sql_tokens[i-1] or "AND" == sql_tokens[i-1]):
                values[tok] = "number"
                if "BETWEEN" == sql_tokens[i-1]:
                    columns[sql_tokens[i-2]].append("number")
            elif i - 1 >= 0 and "LIMIT" == sql_tokens[i-1]:
                values[tok] = "integer"
            elif i - 1 >= 0 and "=" == sql_tokens[i-1]:
                assert "COLUMN" in sql_tokens[i-2]
                columns[sql_tokens[i-2]].append(tok)
            else:
                print("\nTemplate Error: VALUE format is wrong!!!")
                print(sql_pattern)
        elif "DASC" in tok:
            dasc = True

    return (list(set(slots)), columns, ops, values, aggs, dasc)


def get_q_slots(question):
    q_toks = question.strip().split(" ")
    q_slots = list(set([tok for tok in q_toks if "TABLE" in tok or "SC" in tok or ("{" in tok and "}" in tok)]))

    return q_slots


def process_constraints(constraints, columns, slots):
    slot_values = {}
    skip_db_with_one_table = False
    for constraint in constraints:
        if "P0==" == constraint:
            assert "{OP0}" in slots
            slot_values["{OP0}"] = "="
        elif "P1==" == constraint:
            assert "{OP1}" in slots
            slot_values["{OP1}"] = "="
        elif "P0=P1==" == constraint:
            assert "{OP0}" in slots and "{OP1}" in slots
            slot_values["{OP0}"] = "="
            slot_values["{OP1}"] = "="
        elif "P0=P1=P2==" == constraint:
            assert "{OP0}" in slots and "{OP1}" in slots and "{OP2}" in slots
            slot_values["{OP0}"] = "="
            slot_values["{OP1}"] = "="
            slot_values["{OP2}"] = "="
        elif "P0=>" == constraint:
            assert "{OP0}" in slots
            slot_values["{OP0}"] = ">"
        elif "P0=<" == constraint:
            assert "{OP0}" in slots
            slot_values["{OP0}"] = "<"
        elif "{AGG0}=MIN" == constraint:
            assert "{AGG0}" in slots
            slot_values["{AGG0}"] = "MIN"
        elif "{AGG0}=MAX" == constraint:
            assert "{AGG0}" in slots
            slot_values["{AGG0}"] = "MAX"
        elif "C0-id" == constraint:
            skip_db_with_one_table = True
            assert "{COLUMN0}" in slots and "{COLUMN0}" in columns.keys()
            columns["{COLUMN0}"].append("id")
        elif "C1-id" == constraint:
            skip_db_with_one_table = True
            assert "{COLUMN1}" in slots and "{COLUMN1}" in columns.keys()
            columns["{COLUMN1}"].append("id")
        elif "C2-id" == constraint:
            skip_db_with_one_table = True
            assert "{COLUMN2}" in slots and "{COLUMN2}" in columns.keys()
            columns["{COLUMN2}"].append("id")
        elif "C3-T1" == constraint:
            skip_db_with_one_table = True
            assert "{COLUMN3}" in slots and "{COLUMN3}" in columns.keys()
            columns["{COLUMN3}"].append("T1")
        elif "T0-T1-JOIN" == constraint or 'T0-T1-NO-JOIN' == constraint:
            skip_db_with_one_table = True
            columns["{COLUMN0}"].append("T0")
            if "{COLUMN1}" in columns.keys():
                columns["{COLUMN1}"].append("T1")

    return (slot_values, columns, skip_db_with_one_table)


# helper function
def gen_col_info(col_str, columns, columns_inf):
    col_conds = columns[col_str]
    value_slot = [cc for cc in col_conds if "VALUE" in cc]
    col = ""
    value_val = None
    if "id" in col_conds:
        has_id = False
        for c, t, v in columns_inf:
            if "id" in col or "name" in col:
                has_id = True
                col, ctype, values = c, t, v
                break
        if not has_id:
            col, ctype, value = columns_inf[0]
    elif "number" in col_conds:
        for colinfo in columns_inf[1:]:
            if colinfo[1] == "real":
                col, ctype, value = colinfo
    if col == "":
        col, ctype, value = random.choice(columns_inf[1:])

    if len(value_slot) > 0:
        assert len(value_slot) < 3
        if len(value_slot) == 1:
            value_val = [(value_slot[0], value)]
        else:
            value_val = [(value_slot[0], value), (value_slot[1], value)]

    return (col, value_val)


def replace_dict(inp, dicts):
    for rep_in, rep_out in dicts.items():
        inp = inp.replace(rep_in, str(rep_out))

    return inp

def get_labels(sql_pattern):
    STRUCT_KEYWORDS = ["WHERE", "GROUP_BY", "HAVING", "ORDER_BY", "SELECT"]
    EXTRA_OPS = ["NOT_IN", "IN", "BETWEEN", "="]
    COUNT = "COUNT"
    OTHER_KEYWORDS = ["LIMIT"] #AGG, OP, DASC, OR, =
    NEST_KEYWORDS = ["EXCEPT", "UNION", "INTERSECT"]

    sql_tokens = sql_pattern.replace("GROUP BY", "GROUP_BY").replace("ORDER BY", "ORDER_BY").replace("NOT IN", "NOT_IN").split(" ")
    columns = {}
    cur_nest = ""
    cur_struct = ""
    cur_len = len(sql_tokens)
    select_count = 0
    for i, tok in enumerate(sql_tokens):
        if tok in NEST_KEYWORDS:
            if cur_nest == "" or cur_nest == "OP_SEL":
                cur_nest = tok
            else:
                cur_nest = cur_nest + " " + tok
        elif tok in STRUCT_KEYWORDS:
            cur_struct = tok
            if tok == "SELECT":
                select_count += 1
                if select_count > 1 and cur_nest == "":
                    cur_nest = "OP_SEL"
        elif "COLUMN" in tok or "*" == tok:
            if tok not in columns.keys():
                columns[tok] = []
            # SELECT {COLUMN0}
            # SELECT {COLUMN0} , {COLUMN1}
            # SELECT {AGG0} ( {COLUMN0} )
            # SELECT {COLUMN0} {FROM} WHERE {COLUMN1} {OP} ( SELECT {AGG0} ( {COLUMN1} ) {FROM} ) AND {COLUMN2} {OP0} {VALUE0}
            if cur_struct == "SELECT":
                if "," == sql_tokens[i-1] or "SELECT" == sql_tokens[i-1]:
                    columns[tok].append(cur_nest + " " + cur_struct)
                elif "(" == sql_tokens[i-1]:
                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2])
                else:
                    print("\nWarning: unexcepted SELECT format")
                    print(sql_pattern)
            # WHERE {COLUMN} {OP}
            # WHERE {COLUMN2} {OP0}
            # WHERE OR {COLUMN2} {OP0}
            # WHERE {COLUMN2} BETWEEN
            elif cur_struct == "WHERE":
                assert "OP" in sql_tokens[i+1] or sql_tokens[i+1] in EXTRA_OPS
                last_tok = sql_tokens[i-1]
                if "OR" == last_tok or (i+3 < cur_len and "OR" == sql_tokens[i+3]):
                    columns[tok].append(cur_nest + " " + cur_struct + " OR " + sql_tokens[i+1])
                elif "WHERE" == last_tok or "AND" == last_tok:
                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i+1])
                else:
                    print("\nWarning: unexcepted WHERE format")
            # GROUP BY {COLUMN0} , {COLUMN0}
            elif cur_struct == "GROUP_BY":
                columns[tok].append(cur_nest + " " + cur_struct)
            # HAVING COUNT ( * ) {OP0}
            # HAVING {AGG0} ( {COLUMN2} ) {OP0}
            elif cur_struct == "HAVING":
                last_tok = sql_tokens[i-1]
                if last_tok != "(" and not ("AGG" in sql_tokens[i-2] or COUNT == sql_tokens[i-2]):
                    print("\nWarning: unexcepted HAVING format")
                columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2] + " " + sql_tokens[i+2])
            # ORDER BY COUNT ( * ) {DASC} LIMIT
            # ORDER BY COUNT ( * ) {DASC}
            # ORDER BY {COLUMN1} {DASC} LIMIT
            # ORDER BY {COLUMN1} LIMIT
            # ORDER BY {COLUMN1} , {COLUMN1} {DASC} LIMIT
            # ORDER BY {COLUMN1} {DASC} if no DASC then is ASC
            elif cur_struct == "ORDER_BY":
                last_tok = sql_tokens[i-1]
                if last_tok == "(":
                    dasc_tok = "{DASC}"
                    limit_tok = ""
                    if sql_tokens[i+2] != "{DASC}":
                        dasc_tok = "ASC"
                        if sql_tokens[i+2] == "LIMIT":
                            limit_tok = "LIMIT"
                    elif i+3 < cur_len and sql_tokens[i+3] == "LIMIT":
                        limit_tok = "LIMIT"

                    columns[tok].append(cur_nest + " " + cur_struct + " " + sql_tokens[i-2] + " " + dasc_tok + " " + limit_tok)
                elif last_tok == "ORDER_BY" or last_tok == ",":
                    dasc_tok = "ASC"
                    limit_tok = ""
                    # small dirty pass
                    if i+1 < cur_len and sql_tokens[i+1] == "{DASC}":
                        dasc_tok = "{DASC}"
                        if i+2 < cur_len and sql_tokens[i+2] == "LIMIT":
                            limit_tok = "LIMIT"
                    elif i+1 < cur_len and sql_tokens[i+1] == "LIMIT":
                        limit_tok = "LIMIT"

                    columns[tok].append(cur_nest + " " + cur_struct + " " + dasc_tok + " " + limit_tok)

            else:
                print("\n------------Warning: unexcepted COLUMN label format")

    column_labels = {}
    for col, labels in columns.items():
        label_str = " ".join([l.strip() for l in labels])
        column_labels[col] = label_str

    return column_labels

def populate_one(db, templates, templates_one, sql_components):
    """
    'P0=P1==', 'P0=P1=P2==', 'P0==', 'P1==', 'P0=>', 'P0=<', '{AGG0}=MAX', '{AGG0}=MIN'
    'T0-T1-JOIN', 'T0-T1-NO-JOIN',
    'C0-id',, 'C2-id', , 'C1-id',  'C3-T1'
    """
    if len(db) > 1:
        template = random.choice(templates)
    else:
        template = random.choice(templates_one)

    sql_constraints = template['SQL constraints']
    sql_pattern = template["SQL pattern"]
    question, q_constraints = random.choice(template["questions"])
    constraints = list(set(sql_constraints + q_constraints))

    slots, columns, ops, vals, aggs, dasc = get_sql_slots(sql_pattern)
    slot_values, columns, skip_db_with_one_table = process_constraints(constraints, columns, slots)

    q_slots = get_q_slots(question)
    q_slot_values = {}

    # 1 process ops - update columns and values constraints
    for op, colv in ops.items():
        if colv[0] == "*":
            if op not in slot_values.keys():
                op_val = random.choice([">", "<", ">=", "<=", "="])
                slot_values[op] = op_val
                if len(colv) == 2:
                    slot_values[colv[1]] = random.randint(1, 10)
        else:
            if colv[0] not in columns.keys():
                print("\n-----colv[0] not in columns.keys(): ")
                print(columns.keys())
                print(ops)
            assert colv[0] in columns.keys()
            if op not in slot_values.keys():
                if random.random() < OP_VAL_EQUAL:
                    op_val = "="
                else:
                    op_val = random.choice(OPS)
                slot_values[op] = op_val
                if op_val in [">", "<", ">=", "<="]:
                    columns[colv[0]].append("number")
            if len(colv) == 2:
                columns[colv[0]].append(colv[1])

    # 2 process columns
    random.shuffle(db)
    table_0, table_1 = None, None
    table_label_0 = ""
    table_label_1 = ""
    use_table_1 = False

    if "{COLUMN0}" in columns.keys() or "{TABLE0}" in q_slots:
        table_label_0 = "SELECT"

    if len(db) >= 2:
        table_0, table_1 = db[:2]
        if "{TABLE1}" in q_slots:
            table_label_1 = "SELECT"
            if "{TABLE0}" in q_slots:
                # p<0.5 from T0, T1 AND to SELECT T1 *
                # otherwise all from T0 AND to SELECT T1 *
                if random.random() < USE_TABLE_1:
                    use_table_1 = True
            else:
                # p<0.4 all from T0
                # AND to SELECT T1 *
                if random.random() < 0.6:
                    use_table_1 = True
                    if "{COLUMN1}" in columns.keys():
                        table_label_1 = "SELECT"
        else:
            # p<0.5 from T0, T1 AND to SELECT T1 *
            # otherwise all from T0, NOT to SELECT T1 *
            if random.random() < USE_TABLE_1:
                use_table_1 = True
                if "{COLUMN1}" in columns.keys():
                    table_label_1 = "SELECT"
    else:
        table_0, table_1 = db[0], db[0]

    T0 = table_0["name"]
    T1 = table_1["name"]
    columns_inf_0 = list(zip(table_0["columns"], table_0["column_types"], table_0["values"]))[1:]
    if use_table_1:
        columns_inf_1 = list(zip(table_1["columns"], table_1["column_types"], table_1["values"]))[1:]

    if "{COLUMN0}" in columns.keys():
        col_0, value_0 = gen_col_info("{COLUMN0}", columns, columns_inf_0)
        slot_values["{COLUMN0}"] = col_0
        if value_0 is not None:
            for k, v in value_0:
                slot_values[k] = v
        if len(columns_inf_0) > 2:
            columns_inf_0 = [(col, ctype, val) for col, ctype, val in columns_inf_0 if col != col_0]

    if use_table_1:
        columns_input = columns_inf_1
    else:
        columns_input = columns_inf_0

    if "{COLUMN1}" in columns.keys():
        col_1, value_1 = gen_col_info("{COLUMN1}", columns, columns_input)
        slot_values["{COLUMN1}"] = col_1
        if value_1 is not None:
            for k, v in value_1:
                slot_values[k] = v
        columns_input_org = columns_input
        if len(columns_input) > 3:
            columns_input = [(col, ctype, val) for col, ctype, val in columns_input if col != col_1]
        if len(columns_input) < 2:
            columns_input = columns_input_org

    if "{COLUMN2}" in columns.keys():
        col_2, value_2 = gen_col_info("{COLUMN2}", columns, columns_input)
        slot_values["{COLUMN2}"] = col_2
        if value_2 is not None:
            for k, v in value_2:
                slot_values[k] = v
        columns_input_org = columns_input
        if len(columns_input) > 2:
            columns_input = [(col, ctype, val) for col, ctype, val in columns_input if col != col_2]
        if len(columns_input) < 2:
            columns_input = columns_input_org

    if "{COLUMN3}" in columns.keys():
        col_3, value_3 = gen_col_info("{COLUMN3}", columns, columns_input)
        slot_values["{COLUMN3}"] = col_3
        if value_3 is not None:
            for k, v in value_3:
                slot_values[k] = v

    # 3 aggs
    for agg in aggs.keys():
        if agg not in slot_values.keys():
            slot_values[agg] = random.choice(["MAX", "MIN", "SUM", "AVG"])
    # 4 values
    NUM = 1
    for val, cond in vals.items():
        assert val not in slot_values.keys()
        if cond == "integer":
            if random.random() < USE_1_FOR_INTEGER:
                slot_values[val] = 1
            else:
                NUM = random.randint(2, 10)
                slot_values[val] = NUM
        else:
            slot_values[val] = random.randint(0, 100)

    # 5 dasc - true
    if dasc == True:
        slot_values["{DASC}"] = random.choice(["ASC", "DESC"])

    # 6 check if all sql slot values are done
    if len(slots) != len(slot_values):
        print("\nlen(slots) != len(slot_values)")
        print("sql_pattern: ", sql_pattern)
        print("slots: ", slots)
        print("slot_values: ", slot_values.keys())
    assert len(slots) == len(slot_values)

    # 7 for the questions slots:
    for qs in q_slots:
        if qs == "{TABLE0}":
            q_slot_values["{TABLE0}"] = T0
        elif qs == "{TABLE1}":
            q_slot_values["{TABLE1}"] = T1
        elif "SC" in qs:
            sc = slot_values["{DASC}"]
            if "SC" == qs:
                q_slot_values[qs] = random.choice(sql_components["SC"][sc])
            elif "SC_COL_LIMIT" == qs:
                if NUM > 1:
                    sc =  sc + "_NUM"
                    q_slot_values[qs] = random.choice(sql_components["SC_COL_LIMIT"][sc]).replace("[NUM]", str(NUM))
                else:
                    q_slot_values[qs] = random.choice(sql_components["SC_COL_LIMIT"][sc])
            elif "SC_COL_COUNT_LIMIT" in qs:
                sc_type = qs.replace("SC_COL_COUNT_LIMIT", "")
                if NUM > 1:
                    sc =  sc + "_NUM" + sc_type
                    q_slot_values[qs] = random.choice(sql_components["SC_COL_COUNT_LIMIT"][sc]).replace("[NUM]", str(NUM))
                else:
                    sc =  sc + sc_type
                    q_slot_values[qs] = random.choice(sql_components["SC_COL_COUNT_LIMIT"][sc])
            else:
                if "-" not in qs:
                    print("qs wrong", qs)
                assert "-" in qs
                if "C1" in qs:
                    sc_col = slot_values["{COLUMN1}"]
                elif "C2" in qs:
                    sc_col = slot_values["{COLUMN2}"]
                q_slot_values[qs] = random.choice(sql_components["SC_COL"][sc]).replace("[COL]", sc_col)
        else:
            if qs not in slot_values.keys():
                print("qs not in sv: ", qs)
                print("sql_pattern: ", sql_pattern)
                print("slot_values: ", slot_values)
            assert qs in slot_values.keys()
            if "OP" in qs:
                q_slot_values[qs] = random.choice(sql_components["OP"][slot_values[qs]])
            elif "AGG" in qs:
                q_slot_values[qs] = random.choice(sql_components["AGG"][slot_values[qs]])
            elif "COLUMN" in qs:
                q_slot_values[qs] = " ".join(slot_values[qs].split(" ")[1:6])
            elif "VALUE" in qs:
                q_slot_values[qs] = " ".join(str(slot_values[qs]).split(" ")[:5])
            else:
                print("\nWarning: some q slot type not considered!")
                print(qs)

    # 8 check if all question slots are processed
    assert len(q_slots) == len(q_slot_values)

    # 9 generate final SQL-question pair
    question_gen = replace_dict(question, q_slot_values)


    # 10 generate column labels
    slot_values_new = {}
    for sl, vl in slot_values.items():
        if "COLUMN" in sl:
            slot_values_new[sl] = "_=_".join(vl.split(" "))
        else:
            slot_values_new[sl] = vl

    column_labels = get_labels(sql_pattern)
    column_lables_real = {}
    for col, label in column_labels.items():
        if col != "*":
            col = slot_values[col]
        for slot, value in slot_values.items():
            label = label.replace(slot, str(value))
        column_lables_real[col] = label

    # also add labels for table column *
    if table_label_0 != "":
        column_lables_real[table_0["columns"][0]] = table_label_0
    if table_label_1 != "":
        column_lables_real[table_1["columns"][0]] = table_label_1

    sql_gen = replace_dict(sql_pattern.replace(" {FROM}", ""), slot_values_new)

    return (sql_gen, question_gen, column_lables_real)

# augmentation for one db
def augment_db(db, templates, templates_one_table, sql_components, aug_limit):
    count = 1
    augment_pairs = []
    while count < aug_limit or (count == int(aug_limit)+1 and random.random()<aug_limit+1-count):
        sql_gen, question_gen, column_lables = populate_one(db, templates, templates_one_table, sql_components)
        augment_pairs.append((question_gen, sql_gen, column_lables))
        count += 1

    return augment_pairs

def augment_all_dbs(dbs, templates, templates_one_table, sql_components, aug_limit):
    augment_data = {}
    for idx, db in enumerate(dbs):
        # if idx % 10000 == 0:
        #     print("processed: ", idx)
        db_cols = ["*"]
        db_values = [""]
        for tab in db:
            db_cols.extend(tab["columns"])
            db_values.extend(tab["values"])
        assert len(db_cols) == len(db_values)
        schema_str = " </s> ".join(db_cols)
        values_str = " </s> ".join([str(k) for k in db_values])
        schema_str = schema_str + " |-| " + values_str
        augment_pairs = augment_db(db, templates, templates_one_table, sql_components, aug_limit)
        augment_data[schema_str] = augment_pairs

    return augment_data

# Return the mapping of all the labels to an integer
def get_label_map(data):
    label_dict = defaultdict(int)
    for schema_str, example_list in data.items():
        for example in example_list:
            (question, sql, col_labels) = example
            for val in col_labels.values():
                label_dict[val] += 1
    label_list = sorted(label_dict.items(), key=lambda kv: kv[1], reverse=True)
    label_map = {}
    count = 1
    for label, _ in label_list:
        label_map[label] = count
        count += 1

    return label_map
def map_labels(data, label_map, is_dev=False):
    data_new = {}
    skip_count = 0
    count = 0
    for schema_str, exs in data.items():
        count += 1
        # if count % 100000 == 0:
        #     print("processed: ", count)
        data_new[schema_str] = []
        for ex in exs:
            skip = False
            label_dict = ex[2]
            label_dict_new = {}
            for col, label in label_dict.items():
                if label in label_map.keys():
                    label_dict_new[col] = label_map[label]
                else:
                    skip = True
                    skip_count += 1
                    #else just skip
            if not skip:
                data_new[schema_str].append((ex[0], ex[1], ex[2], label_dict_new))

    # print("skip_count: ", skip_count)
    return data_new
def write_final_file(augment_data):
    data_json = []
    skip_count = 0
    line_count = 0
    dup_count = 0
    pro_count = 0
    for schema_str, exs in augment_data.items():
        for ex in exs:
            line_count += 1
            # if line_count % 100000 == 0:
            #     print("processed: ", line_count)
            question, sql, label_strs, label_ints = ex
            col_str, val_str = schema_str.split(" |-| ")
            colns = col_str.split(" </s> ")
            values = val_str.split(" </s> ")
            assert len(colns) == len(values)
            cols = []
            label_num = len(label_ints)
            label_count = 0
            for idx, coln in enumerate(colns):
                col = {}
                col["name"] = coln
                col["value"] = values[idx]
                if coln != "*":
                    col["name"] = " ".join(coln.split(" ")[1:])
                col["label_int"] = 0
                if coln in label_ints.keys():
                    col["label_int"] = label_ints[coln]
                    label_count += 1
                cols.append(col)

            assert label_count >= label_num
            if label_count > label_num:
                dup_count += 1

            col_list = []
            label_list = []
            value_list = []
            col_count = 0
            for i, col in enumerate(cols):
                if col_count > 40 and col["label_int"] == 0:
                    continue
                col_list.append(col["name"])
                value_list.append(col["value"])
                col_count += 1
                label_list.append(int(col["label_int"]))
            assert len(col_list) == len(value_list)

            label_str = " ".join([str(k) for k in label_list])
            q_col_str = "<s> " + question.lower() + " </s> " + " </s> ".join(col_list).strip() + " </s> "
            caption = q_col_str + " ||| " + label_str
            tokens = tokenizer.tokenize(q_col_str)
            if len(tokens) > MAX_TOKEN_LEN:
                continue

            data_json.append({"question": question.lower(),
                              "columns": col_list,
                              "rows": [value_list],
                              "column_labels": label_list
                             })
            pro_count += 1

    print("total line: ", line_count)
    print("skiped line: ", skip_count)
    print("dup line: ", dup_count)
    print("pro line: ", pro_count)

    return data_json
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_file", help="Please provide a processed table file")
    parser.add_argument("nlsql_templates_file", help="Please provide a template file")
    parser.add_argument("sql_components_file", help="Please provide the SQL component file")
    parser.add_argument("output", help="Please provide the output path")
    parser.add_argument("size", type=int, help="Please provide the output path")

    args = parser.parse_args()
    # read input files
    table_file = args.table_file
    nlsql_templates_file = args.nlsql_templates_file
    sql_components_file = args.sql_components_file
    templates = read_NL_SQL_template(nlsql_templates_file)
    sql_components = read_json(sql_components_file)
    all_tables = read_json(table_file)
    table_dbs = create_dbs(all_tables)
    single_table_templates = get_templates_for_one_table(templates)

    sample_size_per_db = 1.0 * args.size / len(table_dbs)
    augment_data = augment_all_dbs(table_dbs, templates, single_table_templates, sql_components, sample_size_per_db)
    label_map = get_label_map(augment_data)
    augment_data = map_labels(augment_data, label_map)
    json_data = write_final_file(augment_data)
    with open(args.output, "w") as f:
        json.dump(json_data, f)
