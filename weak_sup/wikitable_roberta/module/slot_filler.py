import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, Set, Tuple
from collections import defaultdict

from allennlp.common.util import START_SYMBOL
from allennlp.semparse.type_declarations import type_declaration as types
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld
from allennlp.modules.attention.linear_attention import LinearAttention
from nltk.tree import Tree

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logging.getLogger("allennlp").setLevel(logging.ERROR)

class SlotFiller(nn.Module):
    """
    Fill in the slots of sketches created by sketcher
    """
    def __init__(self,
                token_rnn_feat_size: int,
                token2id: Dict,
                token_embed: nn.Embedding,
                prod2id: Dict,
                prod_embed: nn.Embedding,
                slot_feat_size: int) -> None:
        super(SlotFiller, self).__init__()
        self.token_rnn_feat_size = token_rnn_feat_size
        self.token2id = token2id
        self.token_dim = token_embed.embedding_dim
        self.token_embed = token_embed
        self.prod2id = prod2id
        self.prod_embed = prod_embed
        self.slot_feat_size = slot_feat_size
        self.prod_embed_dim = self.prod_embed.embedding_dim

        # choose the filter function
        self.att = LinearAttention(token_rnn_feat_size, slot_feat_size + \
                            self.prod_embed_dim)
        self.score_func = nn.Linear(token_rnn_feat_size + 
                                    self.prod_embed_dim + slot_feat_size, 1)
        self.token2prod = nn.Linear(token_rnn_feat_size, self.prod_embed_dim)
        self.column2prod = nn.Linear(self.token_dim * 2, self.prod_embed_dim)


    def forward(self,
                ques_rep: torch.Tensor,
                sampled_actions: List[str],
                slot_dic: Dict,
                target_list: List,
                world: WikiTablesVariableFreeWorld) -> Dict:
        """
        It takes in a sampled path and finish the selection part
        based on alignments to the question, table and fileter/same_as function.

        Operations for selecting one row: filter_eq, filter_in
        Operations for selecting multiple rows: all filters and all_rows
        """
        _, _token_rnn_feat_size = ques_rep.size()
        assert self.token_rnn_feat_size == _token_rnn_feat_size
        id2column, column2id, column_type_dic, column_reps = self.collect_column_reps(
                                                    world.table_context)

        actions = world.get_valid_actions()
        filtered_actions = self.filter_functions(actions)

        possible_paths = self.get_all_sequences(ques_rep, column2id, column_reps, \
                        sampled_actions, filtered_actions, \
                        slot_dic, world)
        
        correct_lf = []
        candidate_scores = []
        gold_ids = []
        for candidate_path, candidate_score in possible_paths:
            lf = world.get_logical_form(candidate_path)
            candidate_scores.append(candidate_score)
            if world._executor.evaluate_logical_form(lf, target_list):
                correct_lf.append(lf)
                gold_ids.append(1)
            else:
                gold_ids.append(0)

        gold_id_v = torch.FloatTensor(gold_ids)
        if torch.sum(gold_id_v) == 0:
            return 0
        else:
            score_v = torch.stack(candidate_scores, 0)
            score_prob = F.softmax(score_v, 0)
            reward_v = gold_id_v * score_prob
            return torch.sum(reward_v, 0)
    

    def evaluate(self,
                ques_rep: torch.Tensor,
                sampled_actions: List[str],
                slot_dic: Dict,
                target_list: List,
                world: WikiTablesVariableFreeWorld) -> Dict:
        _, _token_rnn_feat_size = ques_rep.size()
        assert self.token_rnn_feat_size == _token_rnn_feat_size
        id2column, column2id, column_type_dic, column_reps = self.collect_column_reps(
                                                    world.table_context)

        actions = world.get_valid_actions()
        filtered_actions = self.filter_functions(actions)

        possible_paths = self.get_all_sequences(ques_rep, column2id, column_reps, \
                        sampled_actions, filtered_actions, \
                        slot_dic, world)
        
        max_path, max_score = possible_paths[0]
        for candidate_path, candidate_score in possible_paths[1:]:
            if candidate_score > max_score:
                max_path = candidate_path

        lf = world.get_logical_form(max_path)
        if world._executor.evaluate_logical_form(lf, target_list):
            return True
        else:
            return False


    def _score_prod(self,
                    token_reps: torch.Tensor,
                    slot_key: str,
                    slot_rep: torch.Tensor,
                    actions: List[str],
                    column2id: Dict,
                    column_vecs: torch.Tensor,
                    world: WikiTablesVariableFreeWorld) -> Dict:
        """
        Generate a slot specific score vecs for each production
        it requires rnn_feat_size == prod_embed
        """
        sent_len, _token_rnn_feat_size = token_reps.size()
        assert self.token_rnn_feat_size == _token_rnn_feat_size
        action_list = []
        for k, v in actions.items():
            if slot_key == "number_column" and k == "f":
                action_list += v
            elif slot_key == "date_column" and k == "m":
                action_list += v
            elif slot_key == "string_column" and k == "t":
                action_list += v
            elif slot_key in ["one_row_select", "mul_rows_select"]:
                action_list += v
        action_list = list(set(action_list))
        prod_num = len(action_list)
        
        prod_vecs = []
        for prod in action_list:
            if prod in self.prod2id:
                prod_id = self.prod2id[prod]
                prod_vec = self.prod_embed.weight[prod_id]
            elif prod.startswith("s -> string:"):
                entity = prod[5:]
                ent_s, ent_e = world.ent2id[entity]
                token_vec = torch.mean(token_reps[ent_s:ent_e], 0)
                prod_vec = self.token2prod(token_vec)
            elif prod.startswith("n -> "):
                num_ = prod[5:]
                num_id = world.num2id[num_]
                token_vec = token_reps[num_id]
                prod_vec = self.token2prod(token_vec)
            elif "_column:":
                column_name = prod.split("_column:")[1]
                column_id = column2id[column_name]
                column_v = column_vecs[column_id]
                prod_vec = self.column2prod(column_v)
            else:
                raise NotImplementedError
            prod_vecs.append(prod_vec)
        
        slot_rep_mat = slot_rep.unsqueeze(0).expand(len(prod_vecs), 
                    self.slot_feat_size)
        prod_mat = torch.stack(prod_vecs, 0)
        feat_vec = torch.cat([prod_mat, slot_rep_mat], 1)
        att_score = self.att(feat_vec, token_reps)  
        att_vec = torch.mm(att_score, token_reps) # prod_num * token_rnn_feat_size 
        score_feat = torch.cat([feat_vec, att_vec], 1) 
        score_vecs = self.score_func(score_feat) # prod_num * 1
        score_vecs = score_vecs.squeeze(1)

        prod_score_dic = dict()
        for i, action in enumerate(action_list):
            prod_score_dic[action] = score_vecs[i] 

        return prod_score_dic

                
    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        if "[" in right_side:
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts = [right_side]
        return right_side_parts
        
    

    def collect_column_reps(self,
                            context: TableQuestionContext) \
                            -> torch.LongTensor:
        # order matters
        id2column = list(context.column_types.keys())
        column2id = {v:k for k,v in enumerate(id2column)}

        # map type to column name
        type_dic = defaultdict(list)
        for i, column in enumerate(id2column):
            type_dic[context.column_types[column]].append(i)
        
        column_vec = []
        for column in id2column:
            column_tokens = []
            for token in column.split("_"):
                token_id = self.token2id[token]
                column_tokens.append(self.token_embed.weight[token_id])
            col_vec = torch.mean(torch.stack(column_tokens), 0)

            column_type = context.column_types[column]
            column_type_sig = f"#{column_type}_col#"
            column_type_id = self.token2id[column_type_sig]
            column_type_v = self.token_embed.weight[column_type_id]
            column_vec.append(torch.cat([col_vec, column_type_v], 0))
        column_vec = torch.stack(column_vec, 0)
        return id2column, column2id, type_dic, column_vec
            

    def _walk(self,
            non_terminal: str,
            actions: List,
            prod_score_dic: Dict,
            world: WikiTablesVariableFreeWorld) -> None:
        incomplete_paths = [([non_terminal], [f"#S# -> {non_terminal}"], None)]

        _completed_paths = []
        multi_match_substitutions = world.get_multi_match_mapping()

        while incomplete_paths:
            next_paths = []
            for nonterminal_buffer, history, cur_score in incomplete_paths:
                # Taking the last non-terminal added to the buffer. We're going depth-first.
                nonterminal = nonterminal_buffer.pop()
                next_actions = []
                if nonterminal in multi_match_substitutions:
                    for current_nonterminal in [nonterminal] + multi_match_substitutions[nonterminal]:
                        if current_nonterminal in actions:
                            next_actions.extend(actions[current_nonterminal])
                elif nonterminal not in actions:
                    continue
                else:
                    next_actions.extend(actions[nonterminal])
                # Iterating over all possible next actions.
                for action in next_actions:
                    if action in history: continue
                    new_history = history + [action]
                    new_nonterminal_buffer = nonterminal_buffer[:]
                    # Since we expand the last action added to the buffer, the left child should be
                    # added after the right child.
                    for right_side_part in reversed(self._get_right_side_parts(action)):
                        if types.is_nonterminal(right_side_part):
                            new_nonterminal_buffer.append(right_side_part)
                    if cur_score is None:
                        new_prod_score = prod_score_dic[action]
                    else:
                        new_prod_score = prod_score_dic[action] + cur_score
                    next_paths.append((new_nonterminal_buffer, new_history, new_prod_score))
            incomplete_paths = []
            for nonterminal_buffer, path, score in next_paths:
                # An empty buffer means that we've completed this path.
                if not nonterminal_buffer:
                    _completed_paths.append( (path, score))
                elif len(path) < 6: #TODO: set this
                    incomplete_paths.append((nonterminal_buffer, path, score))

        strip_path = []
        for path, score in _completed_paths:
            strip_path.append((path[1:], score)) # the first node is faked
        return strip_path
    

    def filter_functions(self, actions: List) -> List:
        def is_float(num_str):
            """
            check if it's a float
            """
            num_str = num_str.replace("_", ".")
            num_str = num_str.replace("~", "-")
            try:
                _ = float(num_str)
                return True
            except:
                return False

        new_action_dict = defaultdict(list)
        for key_type, action_list in actions.items():
            if key_type in ["m", "f", "t", "r", "e"]:
                new_action_dict[key_type] = action_list[:]
            else:
                for action in action_list:
                    lhs, rhs = action.split(" -> ")
                    if rhs in ["date", "-1"] or "string:" in rhs or is_float(rhs):
                        new_action_dict[key_type].append(action)
                    elif rhs.startswith("filter") or rhs.startswith("same_as"):
                        new_action_dict[key_type].append(action)

        new_action_dict["r"].remove("r -> one_row_select")
        new_action_dict["e"].remove("e -> mul_rows_select")
        new_action_dict["e"].append("e -> all_rows")
        return new_action_dict



    def get_all_sequences(self, 
                            token_reps: torch.Tensor,
                            column2id: Dict,
                            column_vecs: torch.Tensor,
                            abs_path: List[str],
                            filtered_actions: Dict,
                            slot_dic: Dict,
                            world: WikiTablesVariableFreeWorld) -> List[Any]:
        """
        get all the possible valid actions sequences give the current prefix
        """
        path = abs_path[:]
        for slot_key in slot_dic:
            if slot_key == "number_column":
                for slot_id, slot_rep in slot_dic[slot_key]:
                    prod_score_dic = self._score_prod(token_reps, slot_key, slot_rep, filtered_actions,
                                                    column2id, column_vecs, world)
                    partial_pathes = self._walk("f", filtered_actions, prod_score_dic, world)
                    path[slot_id] = partial_pathes
            elif slot_key == "date_column":
                for slot_id, slot_rep in slot_dic[slot_key]:
                    prod_score_dic = self._score_prod(token_reps, slot_key, slot_rep, filtered_actions,
                                                    column2id, column_vecs, world)
                    partial_pathes = self._walk("m", filtered_actions, prod_score_dic, world)
                    path[slot_id] = partial_pathes
            elif slot_key == "string_column":
                for slot_id, slot_rep in slot_dic[slot_key]:
                    prod_score_dic = self._score_prod(token_reps, slot_key, slot_rep, filtered_actions,
                                                    column2id, column_vecs, world)
                    partial_pathes = self._walk("t", filtered_actions, prod_score_dic, world)
                    path[slot_id] = partial_pathes
            elif slot_key == "one_row_select":
                for slot_id, slot_rep in slot_dic[slot_key]:
                    prod_score_dic = self._score_prod(token_reps, slot_key, slot_rep, filtered_actions,
                                                    column2id, column_vecs, world)
                    partial_pathes = self._walk("r", filtered_actions, prod_score_dic, world)
                    path[slot_id] = partial_pathes
            elif slot_key == "mul_rows_select":
                for slot_id, slot_rep in slot_dic[slot_key]:
                    prod_score_dic = self._score_prod(token_reps, slot_key, slot_rep, filtered_actions,
                                                    column2id, column_vecs, world)
                    partial_pathes = self._walk("e", filtered_actions, prod_score_dic, world)
                    path[slot_id] = partial_pathes
        
        possible_paths = []

        def recur_find(prefix, i):
            if i == len(path) - 1:
                possible_paths.append(prefix)
                return
            next_itemes = path[i+1]
            if isinstance(next_itemes, list):
                for _actions, _score in next_itemes:
                    new_prefix = prefix[0][:]
                    pre_score = prefix[1]
                    new_prefix += _actions
                    if pre_score is None:
                        new_score = _score
                    else:
                        new_score = pre_score + _score
                    recur_find([new_prefix, new_score], i+1)
            else:
                new_prefix = prefix[0][:]
                new_score = prefix[1]
                new_prefix.append(next_itemes)
                recur_find([new_prefix, new_score], i+1)
        recur_find([[], None], -1)

        return possible_paths