import os
import logging
import torch
import sys
import copy
import pickle
import datetime
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict

from wikisql_roberta.sempar.context.wikisql_context import WikiSQLContext
from wikisql_roberta.sempar.domain_languages.wikisql_language import WikiSQLLanguage
from wikisql_roberta.model.baseline import Programmer
from wikisql_roberta.model.struct import StructProgrammer
from wikisql_roberta.reader.reader import WSReader
from wikisql_roberta.reader.util import load_jsonl, load_jsonl_table, load_actions, load_productions
from wikisql_roberta.train_config.train_seq_config import configs
from wikisql_roberta.trainer.util import get_sketch_prod, filter_sketches, create_opt, clip_model_grad, weight_init, set_seed
from wikitable_roberta.trainer.dataloader import WTBDataLoader, WTBDataset, get_dataloader

from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.data.tokenizers.token import Token


class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikisql_roberta.reader.reader"
        return super().find_class(module, name)

def check_example(example, target_lf_dict, is_training):
    if is_training:
        # if it does not trigger any programs, then no need to train it
        if (example["id"], example["context"]) not in target_lf_dict:
            return False

        # if the sentence is too long, alignment model will take up too much time
        if len(example["tokens"]) > 40 or len(example["tokens"]) == 1:
            return False

        return True

    else:
        if (example["id"], example["context"]) not in target_lf_dict:
            return False
        else:
            return True


def run(gpu, exp_id, exp_name, config):

    # init wandb
    if config.wandb:
        import wandb
        wandb.init(project="wikisql_roberta", group=f"{exp_name}_{exp_id}", job_type="eval")

    # init logger
    # logdir = f"log/{exp_id}_{exp_name}"
    # filename = f"{logdir}/model_{exp_id}_{exp_name}_eval.log"
    # logging.basicConfig(
    #     filename=filename,
    #     filemode="w",
    #     level=logging.INFO,
    # )
    # logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
    # logger.info(str(config))

    # use only one gpu
    assert config.nodes == 0

    # seed
    set_seed(config.seed)

    # load raw data
    with open(config.reader_pkl, 'rb') as f:
        unpickler = ReaderUnpickler(f)
        wt_reader = unpickler.load()
    with open(config.sketch_train_pkl, 'rb') as f:
        train_target_lf = pickle.load(f)
    with open(config.sketch_dev_pkl, 'rb') as f:
        dev_target_lf = pickle.load(f)
    with open(config.sketch_test_pkl, 'rb') as f:
        test_target_lf = pickle.load(f)
    sketch_lf_actions = load_actions(config.sketch_action_file)
    id2prod = load_productions(config.prod_file)
    prod2id = {v:k for k,v in enumerate(id2prod)}

    # load data
    train_examples = wt_reader.train_examples
    dev_examples = wt_reader.dev_examples
    test_examples = wt_reader.test_examples
    tables = wt_reader.table_dict

    # model
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    assert config.model_type == "struct"
    P = StructProgrammer
    
    programmer = P(config.token_embed_size, config.token_rnn_size,
                            config.token_dropout, config.token_indicator_size, sketch_lf_actions,
                            config.slot_dropout, config.prod_embed_size, prod2id, config.prod_rnn_size,
                            config.prod_dropout, config.column_type_embed_size, config.column_indicator_size,
                            config.op_embed_size, config.slot_hidden_score_size, device,
                            config.roberta_path, train_target_lf, dev_target_lf, test_target_lf, tables)
    programmer.to(device)

    if config.checkpoint_file:
        logger.info(f"Loading checkpoint from checkpoints/{config.checkpoint_file}")
        state_dict = torch.load(f"{config.checkpoint_file}", map_location="cpu")
        def filter_name(x):
            if x.startswith('module.'):
                return x[7:]
            else:
                return x
        state_dict = {filter_name(k):v for (k,v) in state_dict.items()}
        programmer.load_state_dict(state_dict)
    programmer.to(gpu)

    # load dev set
    # dev_examples = dev_examples[:10]
    total_dev_examples = len(dev_examples)
    dev_examples = list(filter(lambda example: check_example(
        example,
        dev_target_lf,
        False,
    ), dev_examples))
    dev_dataloader = WTBDataLoader(
        examples=dev_examples,
        example_dict=dev_target_lf,
        config=config,
        rank=None,
        mode="dev",
    )

    # load test set
    # test_examples = test_examples[:10]
    total_test_examples = len(test_examples)
    test_examples = list(filter(lambda example: check_example(
        example,
        test_target_lf,
        False,
    ), test_examples))
    test_dataloader = WTBDataLoader(
        examples=test_examples,
        example_dict=test_target_lf,
        config=config,
        rank=None,
        mode="test",
    )

    # evaluate on dev
    # evaluate_epoch(
    #     dev_dataloader, total_dev_examples, programmer,
    #     gpu, config, logger, mode="dev",
    # )

    # evaluate on test
    evaluate_epoch(
        test_dataloader, total_test_examples, programmer,
        gpu, config, logger, mode="test",
    )

def evaluate_epoch(dataloader, total_examples, programmer, gpu, config, logger, mode=None):
    s_counter = 0.0
    p_counter = 0.0
    r_counter = 0.0
    gpu_total_examples = 0.0

    for step, batch in enumerate(tqdm(dataloader)):

        programmer.eval()
        with torch.no_grad():
            if mode == "dev":
                ret_dics = programmer(batch)
            elif mode == "test":
                ret_dics = programmer(batch)
        for ret_dic in ret_dics:
            if ret_dic is not None:
                if ret_dic['sketch_triggered']: s_counter += 1.0
                if ret_dic['lf_triggered']: p_counter += 1.0
                if ret_dic['is_multi_col']: r_counter += 1.0

            logger.info(f"[GPU: {gpu}] Question: {ret_dic['question']}")
            logger.info(f"[GPU: {gpu}] Question-ID: {ret_dic['id']}")
            logger.info(f"[GPU: {gpu}] Question Table ID: {ret_dic['context']}")
            logger.info(f"[GPU: {gpu}] Best logical form: {ret_dic['best_program_lf']}")
            logger.info(f"[GPU: {gpu}] Best score: {ret_dic['best_score']}")
            logger.info(f"[GPU: {gpu}] Correctness: {ret_dic['lf_triggered']}")
            logger.info(f"[GPU: {gpu}] MultiCol: {ret_dic['is_multi_col']}")
            logger.info(f"\n")

        gpu_total_examples += len(batch)

    logger.info(f"[GPU: {gpu}] {mode} Overall total examples {total_examples}")
    logger.info(f"[GPU: {gpu}] {mode} GPU total examples {gpu_total_examples}")
    logger.info(f"[GPU: {gpu}] {mode} Coverage {gpu_total_examples * 1.0 / total_examples}")

    p_acc = p_counter / (total_examples)
    logger.info(f"[GPU: {gpu}] {mode} p_counter {p_counter}")
    logger.info(f"[GPU: {gpu}] {mode} accuracy: %f", p_acc)

    s_acc = s_counter / (total_examples)
    logger.info(f"[GPU: {gpu}] {mode} s_counter {s_counter}")
    logger.info(f"[GPU: {gpu}] {mode} accuracy of sketch: %f", s_acc)

    r_percent = r_counter / (total_examples)
    logger.info(f"[GPU: {gpu}] {mode} r_counter {s_counter}")
    logger.info(f"[GPU: {gpu}] {mode} MulCol percents: %f", r_percent)

    if config.wandb:
        wandb.log({f'{mode}_accuracy': p_acc})
        wandb.log({f'{mode}_accuracy_of_sketch': s_acc})
        wandb.log({f'{mode}_mulcol_percents': r_percent})

    return p_acc

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("please specify exp_id exp_name checkpoint_path")
        sys.exit(0)
    exp_id = sys.argv[1]
    exp_name = sys.argv[2]
    checkpoint_file = sys.argv[3]
    print(f"Experiment {exp_id}_{exp_name}")

    # get config
    config = configs[exp_name]

    # parse config arguments
    print(str(config))

    # only use one gpu
    config.nodes = 0
    config.gpus = 1

    run(config.gpu_ids[0], exp_id, exp_name, config)
