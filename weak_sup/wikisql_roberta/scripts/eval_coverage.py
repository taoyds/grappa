import os

from allennlp.data.tokenizers.token import Token

from reader.reader import WSReader
from reader.util import load_jsonl, load_jsonl_table

from wikisql.sempar.action_walker import ActionSpaceWalker
from wikisql.sempar.context.wikisql_context import WikiSQLContext
from wikisql.sempar.domain_languages.wikisql_language import WikiSQLLanguage

from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict
import pickle
import sys

import logging
logger = logging.getLogger("root")  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

DATA_DIR = Path("../dataset/wikisql/processed_input/preprocess_2")
TRAIN_FILE = DATA_DIR / "train_split.jsonl"
DEV_FILE = DATA_DIR / "dev_split.jsonl"
TEST_FILE= DATA_DIR / "test_split.jsonl"
TABLE_FILE= DATA_DIR / "tables.jsonl"
EMBED_FILE = "../glove/glove.6B.300d.txt"

log_path = "log/eval_coverage.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def coverage(examples: Dict,
           max_sketch_length: int,
           table_dict: Dict,
           output_path: str) -> None :
    coverage_counter = 0
    sketch_trigger_example = defaultdict(list)
    output_file_pointer = open(output_path, "w")
    for example in tqdm(examples):
        sketch_candidates = []
        table_id = example["context"]
        processed_table = table_dict[table_id]

        target_value = example["answer"] # (targeValue, targetCan)
        tokenized_question = [ Token(token) for token in  example["tokens"]]

        context = WikiSQLContext.read_from_json(example, processed_table)
        context.take_features(example)

        world = WikiSQLLanguage(context)
        walker = ActionSpaceWalker(world)

        sketch2lf = defaultdict(list)
        all_logical_forms = walker.get_logical_forms_by_sketches(max_sketch_length, None)
        # output the correct logical form
        for sketch, logical_form in all_logical_forms:
            sketch = world.action_sequence_to_logical_form(sketch)
            if world.evaluate_logical_form(logical_form, target_value):
                sketch2lf[sketch].append(logical_form)

        question_id = example["id"]
        utterance = example["question"]
        print(f"{question_id} {utterance}", file=output_file_pointer)
        print(f"Table: {table_id}", file=output_file_pointer)
        if len(sketch2lf) == 0:
            print("NO LOGICAL FORMS FOUND!", file=output_file_pointer)
        else:
            coverage_counter += 1
        for sketch in sketch2lf:
            sketch_trigger_example[sketch].append((question_id, table_id))
            print("Sketch:", sketch, file=output_file_pointer)
            for lf in sketch2lf[sketch]:
                print("\t", lf, file=output_file_pointer)

        print(file=output_file_pointer)
        print(file=output_file_pointer)
    output_file_pointer.close()
    print(f"Coverage: {coverage_counter}/{len(examples)}")

    with open(str(output_path) + ".sketch.stat", "wb") as f:
        pickle.dump(sketch_trigger_example, f)
    return sketch_trigger_example

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python eval_coverage.py exp_id max_sketch_length")
        sys.exit(0)

    exp_id = sys.argv[1]
    max_sketch_length = int(sys.argv[2])
    print(f"Exp id: {exp_id}")

    # load examples
    train_examples = load_jsonl(TRAIN_FILE)
    dev_examples = load_jsonl(DEV_FILE)
    test_examples = load_jsonl(TEST_FILE)
    tables = load_jsonl_table(TABLE_FILE)

    wt_reader = WSReader(tables, train_examples, dev_examples, test_examples, EMBED_FILE)
    wt_reader.check()

    # evaluate the sketches
    output_path = f"log/{exp_id}.train.programs"
    coverage(wt_reader.train_examples, max_sketch_length,  wt_reader.table_dict, output_path)

    output_path = f"log/{exp_id}.dev.programs"
    coverage(wt_reader.dev_examples, max_sketch_length, wt_reader.table_dict, output_path)

    output_path = f"log/{exp_id}.test.programs"
    coverage(wt_reader.test_examples, max_sketch_length, wt_reader.table_dict, output_path)
