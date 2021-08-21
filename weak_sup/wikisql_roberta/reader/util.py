import json

def load_jsonl(filename):
    examples = []
    with open(filename) as f:
        for line in f:
            _example = json.loads(line)
            examples.append(_example)
    return examples

def load_jsonl_table(filename):
    tables = dict()
    with open(filename) as f:
        for line in f:
            _table = json.loads(line)
            tables[_table["name"]] = _table
    return tables

def load_actions(filename):
    actions = []
    with open(filename) as f:
        for line in f:
            actions.append(tuple(line.strip().split("&")))
    return actions

def load_productions(filename):
    productions = []
    with open(filename) as f:
        for line in f:
            productions.append(line.strip())
    return productions