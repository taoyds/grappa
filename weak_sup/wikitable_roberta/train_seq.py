import argparse
import torch
import sys
import os
import pickle
import copy
import logging
import shutil
import datetime

from tqdm import tqdm
from pathlib import Path
from typing import List
from collections import defaultdict

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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


class ReaderUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "wikitable_roberta.reader.reader"
        return super().find_class(module, name)

def setup(rank, world_size):
    dist.init_process_group(
	backend="nccl",
	init_method="env://",
	world_size=world_size,
	rank=rank,
    )

def cleanup():
    dist.destroy_process_group()
    

def run(gpu, exp_id, exp_name, config, checkpoint_dir):
    # init wandb
    if config.wandb:
        import wandb
        wandb.init(
            project="wikitable_roberta", group=f"{exp_name}_{exp_id}", job_type="train"
        )
        wandb.config.update(config)

    # init logger
    filename = f"log/{exp_id}_{exp_name}/model_{exp_id}_{exp_name}_GPU{gpu}.log"
    logging.basicConfig(
        filename=filename, filemode="w", level=logging.INFO,
    )
    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
    logger.info(str(config))

    # calculate rank and init dist process
    if config.nodes != 0:
        rank = config.node_rank * config.gpus + gpu
        setup(rank, config.world_size)
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
        programmer = DDP(
            programmer, device_ids=[gpu], find_unused_parameters=True,
        )

    # optimizer and scheduler
    optimizer, scheduler, roberta_optimizer = create_opt(
        programmer,
        "Adam",
        config.lr,
        config.l2,
        config.roberta_lr,
        config.roberta_finetune,
    )

    # load train set
    total_train_examples = len(train_examples)
    train_examples = list(
        filter(
            lambda example: check_example(example, example_dict[1], True,),
            train_examples,
        )
    )
    train_dataloader = WTBDataLoader(
        examples=train_examples,
        example_dict=example_dict[1],
        config=config,
        rank=rank,
        mode="train",
    )

    # load dev set
    total_dev_examples = len(dev_examples)
    dev_examples = list(
        filter(
            lambda example: check_example(example, example_dict[1], False,),
            dev_examples,
        )
    )
    dev_dataloader = WTBDataLoader(
        examples=dev_examples,
        example_dict=example_dict[1],
        config=config,
        rank=None,
        mode="dev",
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

    # begin training epochs
    best_model_path = None
    best_epoch = 0
    best_accuracy = 0.0
    for epoch in range(config.epochs):
        logger.info("")
        if roberta_optimizer:
            logger.info(
                f"[GPU: {gpu}] Beginning epoch {epoch + 1}; \
                optimizer_lr: {optimizer.param_groups[0]['lr']}; \
                roberta_optimizer_lr: {roberta_optimizer.param_groups[0]['lr']}"
            )
        else:
            logger.info(
                f"[GPU: {gpu}] Beginning epoch {epoch + 1}; \
                optimizer_lr: {optimizer.param_groups[0]['lr']}; \
                roberta_optimizer_lr: None"
            )

        # run training
        print(f"[GPU: {gpu}] training for epoch {epoch + 1}")
        train_epoch(
            train_dataloader,
            total_train_examples,
            programmer,
            optimizer,
            roberta_optimizer,
            gpu,
            config,
            epoch,
            logger,
        )
        scheduler.step()

        # run evaluation on dev
        print(f"[GPU: {gpu}] evaluation on epoch {epoch + 1}")
        accuracy = evaluate_epoch(
            dev_dataloader,
            total_dev_examples,
            programmer,
            gpu,
            config,
            logger,
            mode="dev",
        )
        if accuracy > best_accuracy:
            best_epoch = epoch + 1
            best_accuracy = accuracy
            this_model_path = f"checkpoints/{checkpoint_dir}/{exp_id}_{exp_name}_GPU{gpu}_epoch{best_epoch}_acc{accuracy:.3f}.pt"
            logger.info(
                f"[GPU: {gpu}] Dumping epoch {best_epoch} model to {this_model_path}"
            )
            torch.save(programmer.state_dict(), this_model_path)

            best_model = copy.deepcopy(programmer)
            best_model_path = this_model_path
    
    # save best model
    final_best_model_path = f"checkpoints/{checkpoint_dir}/{exp_id}_{exp_name}_gpu_{gpu}_best_model.pt"
    shutil.copyfile(best_model_path, final_best_model_path)
    logger.info(f"[GPU: {gpu}] Dumping best model to {final_best_model_path}")

    # evaluate on test
    # dist.barrier() # make sure all training are done
    config.gpus = 1  #TODO: hard-code computing acc on one gpu, change this
    evaluate_epoch(
        test_dataloader,
        total_test_examples,
        best_model,
        gpu,
        config,
        logger,
        mode="test",
    )

    cleanup()


def train_epoch(
    train_dataloader,
    total_train_examples,
    programmer,
    opt,
    roberta_opt,
    gpu,
    config,
    epoch,
    logger,
):

    counter = 0.0
    gpu_total_examples = 0.0
    tr_loss = 0.0
    total_steps = 0.0

    for step, batch in enumerate(tqdm(train_dataloader)):

        # reset gradients
        programmer.train()
        opt.zero_grad()
        if roberta_opt:
            roberta_opt.zero_grad()

        # step through batch
        loss = programmer(batch)
        if loss is not None:
            counter += loss.size()[0]
            loss = torch.mean(loss)

            # backward pass
            if config.gpus > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            # wandb.log({"batch_training_loss": loss.item()})

            clip_model_grad(programmer, config.clip_norm)
            opt.step()
            if roberta_opt:
                roberta_opt.step()

        # count total examples
        gpu_total_examples += len(batch)
        total_steps += 1

    logger.info(
        f"[GPU: {gpu}] Epoch {epoch + 1} Overall total examples {total_train_examples}"
    )
    logger.info(
        f"[GPU: {gpu}] Epoch {epoch + 1} GPU total examples {gpu_total_examples}"
    )

    epoch_coverage = counter / (total_train_examples / float(config.gpus))
    logger.info(f"[GPU: {gpu}] Epoch {epoch + 1} counter {counter}")
    logger.info(f"[GPU: {gpu}] Epoch {epoch + 1} coverage {epoch_coverage}")

    epoch_loss = tr_loss / total_steps
    logger.info(f"[GPU: {gpu}] Epoch {epoch + 1} Loss {epoch_loss}")

    # wandb.log({"epoch_training_coverage": epoch_coverage})
    # wandb.log({"epoch_training_loss": epoch_loss})


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

    # wandb.log({f"{mode}_accuracy": p_acc})
    # wandb.log({f"{mode}_accuracy_of_sketch": s_acc})
    # wandb.log({f"{mode}_mulcol_percents": r_percent})

    return p_acc


def parse_args():
    parser = argparse.ArgumentParser(description="WTB Train")
    parser.add_argument("id", help="identifier of exp, e.g., 1")
    parser.add_argument("name", help="config name, see train_config.py")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 1. collect args and config
    args = parse_args()
    exp_id = args.id
    exp_name = args.name
    print(f"Experiment {exp_id}_{exp_name}")

    # parse config arguments
    config = configs[exp_name]
    print(str(config))

    # 2. make checkpoint directory
    datetime_str = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    checkpoint_dir = f"{exp_id}_{exp_name}_{datetime_str}"
    os.mkdir(f"log/{checkpoint_dir}")

    # 3. distributed training
    os.environ["MASTER_ADDR"] = config.addr
    os.environ["MASTER_PORT"] = config.port

    # begin training
    if config.nodes == 0:
        assert config.gpus == 1
        run(config.gpu_ids[0], exp_id, exp_name, config, checkpoint_dir)
    else:
        assert config.gpus > 1
        torch.multiprocessing.spawn(
            run, nprocs=config.gpus, args=(exp_id, exp_name, config, checkpoint_dir),
            join=True
        )

