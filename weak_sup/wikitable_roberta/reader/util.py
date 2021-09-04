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
    actions = set()
    with open(filename) as f:
        for line in f:
            actions.add(tuple(line.strip().split("&")))
    return list(actions)