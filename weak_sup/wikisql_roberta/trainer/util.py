import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable

from allennlp.common.util import START_SYMBOL
from sempar.action_walker import ActionSpaceWalker
from sempar.context.wikisql_context import WikiSQLContext
from sempar.domain_languages.wikisql_language import WikiSQLLanguage
from allennlp.data.tokenizers.token import Token

from typing import List, Dict
from collections import defaultdict


def create_opt(programmer, opt, lr, l2, roberta_lr, roberta_finetune):
    params_trainer = []
    params_roberta_trainer = []
    for name, param in programmer.named_parameters():
        if param.requires_grad:
            if "roberta_model" in name:
                params_roberta_trainer.append(param)
            else:
                params_trainer.append(param)

    if opt == "SGD":
        optimizer = optim.SGD(params_trainer, lr=lr, weight_decay=l2)
    elif opt == "Adam":
        optimizer = optim.Adam(params_trainer, lr=lr, weight_decay=l2)
    elif opt == "Adagrad":
        optimizer = optim.Adagrad(params_trainer, lr=lr, weight_decay=l2)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12], gamma=0.1)

    if roberta_finetune:
        roberta_optimizer = optim.Adam(params_roberta_trainer, lr=roberta_lr)
    else:
        roberta_optimizer = None

    return optimizer, scheduler, roberta_optimizer


def weight_init(m: nn.Module):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def clip_model_grad(model, clip_norm: int):
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type=2)

def set_seed(seed:float):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_sketch_prod_and_slot(examples: List, 
                                table_dict: Dict, 
                                sketch_list: List, 
                                sketch_action_list: List):
    """
    If it contains all three types of columns, then the grammar is complete
    Also return sketch action list and their slots 
    Used for pruned version
    """
    for example in examples:

        table_id = example["context"]
        processed_table = table_dict[table_id]
        context = WikiSQLContext.read_from_json(example, processed_table)
        context.take_features(example)
        world = WikiSQLLanguage(context)

        if len(context.column_types) >= 2 and len(context._num2id) > 0 and \
            len(context._entity2id) > 0:
            actions = world.get_nonterminal_productions()
            sketch_actions = world._get_sketch_productions(actions)

            # index all the possible actions
            action_set = set()
            for k, v in sketch_actions.items():
                action_set = action_set.union(set(v))
            id2prod = list(action_set)
            prod2id = {v:k for k,v in enumerate(id2prod)}

            # lf to actions
            sketch_lf2actions = dict()
            for sketch_actions in sketch_action_list:
                lf = world.action_sequence_to_logical_form(sketch_actions)
                sketch_lf2actions[lf] = sketch_actions

            # sort by length in decreasing order
            slot_dict = defaultdict(dict)
            sketch_action_seqs = []
            for sketch in sketch_list:
                sketch_actions = sketch_lf2actions[sketch]
                sketch_actions = tuple(sketch_actions)
                sketch_action_seqs.append(sketch_actions)

                for action_ind, action in enumerate(sketch_actions):
                    assert action in prod2id
                    lhs, rhs = action.split(" -> ")
                    if lhs in ["Column", "StringColumn", "NumberColumn", "ComparableColumn", 
                        "DateColumn", "str", "Number", "Date"] and rhs == "#PH#":
                        slot_dict[sketch_actions][action_ind] = lhs
                    elif lhs == "List[Row]" and rhs == "#PH#":
                        slot_dict[sketch_actions][action_ind] = lhs

            return id2prod, prod2id, sketch_action_seqs, slot_dict 


def filter_sketches(sketch_dict: Dict, sketch_threshold:int) -> List:
    """
    filter sketches with only one constraint
    """
    example_dict = defaultdict(set)
    for sketch in sketch_dict:
        for q, t in sketch_dict[sketch]:
            example_dict[(q, t)].add(sketch)

    filtered_sketch_set = set()
    for example in example_dict:
        if len(example_dict[example]) <= sketch_threshold:
            for sketch in example_dict[example]:
                filtered_sketch_set.add(sketch)
    print(f"{len(filtered_sketch_set)} sketches loaded")

    return list(filtered_sketch_set)


def get_sketch_prod(examples:List, table_dict:Dict) -> List:
    """
    If it contains all three types of columns, then the grammar is complete
    Also return sketch action list and their slots 
    """
    for example in examples:

        table_id = example["context"]
        processed_table = table_dict[table_id]
        context = WikiSQLContext.read_from_json(example, processed_table)
        context.take_features(example)
        world = WikiSQLLanguage(context)

        if len(context.column_types) >= 2 and len(context._num2id) > 0 and \
            len(context._entity2id) > 0:
            actions = world.get_nonterminal_productions()
            sketch_actions = world._get_sketch_productions(actions)

            # index all the possible actions
            action_set = set()
            for k, v in sketch_actions.items():
                action_set = action_set.union(set(v))
            id2prod = list(action_set)
            prod2id = {v:k for k,v in enumerate(id2prod)}

            return id2prod, prod2id