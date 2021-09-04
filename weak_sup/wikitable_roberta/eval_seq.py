import argparse
import torch
import sys
import os
import pickle
import copy
import wandb
import logging
import shutil
import datetime

from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict

from wikitable_roberta.sempar.context.table_question_context import TableQuestionContext
from wikitable_roberta.sempar.domain_languages.wikitable_abstract_language import (
    WikiTableAbstractLanguage,
)
from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.data.tokenizers.token import Token

from wikitable_roberta.model.baseline import Programmer
from wikitable_roberta.model.seq import SeqProgrammer
from wikitable_roberta.model.struct import StructProgrammer
from wikitable_roberta.reader.reader import WTReader
from wikitable_roberta.reader.util import load_jsonl, load_jsonl_table, load_actions
from wikitable_roberta.train_config.train_seq_config import configs
from wikitable_roberta.trainer.util import (
    get_sketch_prod,
    filter_sketches,
    create_opt,
    clip_model_grad,
    weight_init,
    set_seed,
)
from wikitable_roberta.trainer.dataloader import (
    WTBDataLoader,
    WTBDataset,
    check_example,
    get_dataloader,
)

DATA_DIR_RAW = Path("../dataset/wikitable/raw_input/WikiTableQuestions/tagged")


class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikitable.reader.reader"
        return super().find_class(module, name)


def run(gpu, exp_id, exp_name, config, checkpoint_path):
    # init wandb
    wandb.init(
        project="wikitable_roberta", group=f"{exp_name}_{exp_id}", job_type="eval"
    )

    # init logger
    logdir = f"log/eval_{exp_id}_{exp_name}/"
    filename = f"{logdir}/model_{exp_id}_{exp_name}_GPU{gpu}.log"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(
        filename=filename, filemode="w", level=logging.INFO,
    )
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
    logger.info(str(config))

    # calculate rank and init dist process
    if config.nodes != 0:
        rank = config.node_rank * config.gpus + gpu
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config.world_size,
            rank=rank,
        )
    else:
        rank = None

    # seed
    set_seed(config.seed)

    # load raw data
    with open(config.reader_pkl, "rb") as f:
        unpickler = ReaderUnpickler(f)
        wt_reader = unpickler.load()
    with open(config.sketch_pkl, "rb") as f:
        example_dict = pickle.load(f)
    with open(config.sketch_test_pkl, "rb") as f:
        test_example_dict = pickle.load(f)
    sketch_lf_actions = load_actions(config.sketch_action_file)

    # load data
    train_examples = wt_reader.train_examples
    dev_examples = wt_reader.dev_examples
    test_examples = wt_reader.test_examples
    tables = wt_reader.table_dict
    id2prod, prod2id = get_sketch_prod(train_examples, tables)

    # model
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    if config.model_type == "seq":
        P = SeqProgrammer
    elif config.model_type == "struct":
        P = StructProgrammer
    else:
        P = Programmer
    programmer = P(
        config.token_embed_size,
        config.var_token_size,
        wt_reader.vocab,
        config.token_rnn_size,
        config.token_dropout,
        config.token_indicator_size,
        sketch_lf_actions,
        config.slot_dropout,
        wt_reader.pos2id,
        config.pos_embed_size,
        config.prod_embed_size,
        prod2id,
        config.prod_rnn_size,
        config.prod_dropout,
        config.column_type_embed_size,
        config.column_indicator_size,
        config.op_embed_size,
        config.slot_hidden_score_size,
        device,
        config.roberta_path,
        config.use_roberta,
        config.use_tablebert,
        example_dict[1],
        test_example_dict[1],
    )
    programmer.to(gpu)

    # wrap model
    if config.nodes != 0:
        programmer = torch.nn.parallel.DistributedDataParallel(
            programmer, device_ids=[gpu], find_unused_parameters=True,
        )

    # load test set
    total_test_examples = len(test_examples)
    test_examples = list(
        filter(
            lambda example: check_example(example, test_example_dict[1], False,),
            test_examples,
        )
    )
    test_dataloader = WTBDataLoader(
        examples=test_examples,
        example_dict=test_example_dict[1],
        config=config,
        rank=None,
        mode="test",
    )

    print(f"Loading from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=f"cuda:{gpu}")
    def filter_name(x): return x[7:] if x.startswith('module.') else x
    state_dict = {filter_name(k): v for (k, v) in state_dict.items()}
    programmer.load_state_dict(state_dict)

    # evaluate on test
    evaluate_epoch(
        test_dataloader,
        total_test_examples,
	programmer,
        gpu,
        config,
        logger,
        mode="test",
    )


def evaluate_epoch(
    dataloader, total_examples, programmer, gpu, config, logger, mode=None
):
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
                if ret_dic["sketch_triggered"]:
                    s_counter += 1.0
                if ret_dic["lf_triggered"]:
                    print("Yeah!")
                    p_counter += 1.0
                if ret_dic["is_multi_col"]:
                    r_counter += 1.0

            # logger.info(f"[GPU: {gpu}] Question: {ret_dic['question']}")
            # logger.info(f"[GPU: {gpu}] Question-ID: {ret_dic['id']}")
            # logger.info(f"[GPU: {gpu}] Question Table ID: {ret_dic['context']}")
            # logger.info(f"[GPU: {gpu}] Best logical form: {ret_dic['best_program_lf']}")
            # logger.info(f"[GPU: {gpu}] Best score: {ret_dic['best_score']}")
            # logger.info(f"[GPU: {gpu}] Correctness: {ret_dic['lf_triggered']}")
            # logger.info(f"[GPU: {gpu}] MultiCol: {ret_dic['is_multi_col']}")
            # logger.info(f"\n")

        gpu_total_examples += len(batch)

    logger.info(f"[GPU: {gpu}] {mode} Overall total examples {total_examples}")
    logger.info(f"[GPU: {gpu}] {mode} GPU total examples {gpu_total_examples}")

    p_acc = p_counter / (total_examples / float(config.gpus))
    logger.info(f"[GPU: {gpu}] {mode} p_counter {p_counter}")
    logger.info(f"[GPU: {gpu}] {mode} accuracy: %f", p_acc)

    s_acc = s_counter / (total_examples / float(config.gpus))
    logger.info(f"[GPU: {gpu}] {mode} s_counter {s_counter}")
    logger.info(f"[GPU: {gpu}] {mode} accuracy of sketch: %f", s_acc)

    r_percent = r_counter / (total_examples / float(config.gpus))
    logger.info(f"[GPU: {gpu}] {mode} r_counter {s_counter}")
    logger.info(f"[GPU: {gpu}] {mode} MulCol percents: %f", r_percent)

    wandb.log({f"{mode}_accuracy": p_acc})
    wandb.log({f"{mode}_accuracy_of_sketch": s_acc})
    wandb.log({f"{mode}_mulcol_percents": r_percent})

    return p_acc


def parse_args():
    parser = argparse.ArgumentParser(description="WTB Train")
    parser.add_argument("id", help="identifier of exp, e.g., 1")
    parser.add_argument("name", help="config name, see train_config.py")
    parser.add_argument("ck_path", help="checkpoint path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 1. collect args and config
    args = parse_args()
    exp_id = args.id
    exp_name = args.name
    ck_path = args.ck_path
    print(f"Experiment {exp_id}_{exp_name}")

    # parse config arguments
    config = configs[exp_name]
    print(str(config))

    # manully set
    config.nodes = 0
    config.gpus = 1

    # 2. distributed eval
    # OS environment variables
    os.environ["MASTER_ADDR"] = config.addr
    os.environ["MASTER_PORT"] = config.port

    # begin training
    if config.nodes == 0:
        assert config.gpus == 1
        run(config.gpu_ids[0], exp_id, exp_name, config, ck_path)
    else:
        assert config.gpus > 1
        torch.multiprocessing.spawn(
            run, nprocs=config.gpus, args=(exp_id, exp_name, config, ck_path)
        )

