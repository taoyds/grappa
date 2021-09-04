import torch
import logging
from torch import nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, Set
from collections import defaultdict

from allennlp.common.util import START_SYMBOL
from allennlp.semparse.type_declarations import type_declaration as types
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld
from allennlp.modules.attention.linear_attention import LinearAttention
from torch.distributions import Categorical


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Sketcher(nn.Module):
    """
    Give a probability distribution over all the valid sketches.
    Integrated functionality for sampling from this distribution
    """
    def __init__(self, 
                max_path_length: int,
                token_rnn_feat_size: int,
                prod2id: Dict, 
                prod_embed: nn.Embedding,
                prod_hidden_size: int,
                att_hidden_size: int) -> None:
        super(Sketcher, self).__init__()
        self.token_rnn_feat_size = token_rnn_feat_size
        self.prod2id = prod2id
        self.prod_embed = prod_embed
        self.prod_embed_size = prod_embed.embedding_dim
        self.att = LinearAttention(token_rnn_feat_size, self.prod_embed_size)
        self.hidden_func = nn.Linear(token_rnn_feat_size + self.prod_embed_size, 
                                        att_hidden_size)
        self.score_func = nn.Linear(att_hidden_size, 1)
        self._max_path_length = max_path_length

        self.prod_rnn = nn.LSTM(self.prod_embed_size, prod_hidden_size, 1, 
                bidirectional=True, batch_first=True)

    
    def forward(self, 
                token_reps: List[torch.Tensor],
                world: WikiTablesVariableFreeWorld) -> Dict:
        actions = world.get_valid_actions()
        actions = self._filter_abstract(actions)

        prod_score_dic = self._score_prod(token_reps, actions, world)
        # sketch_lfs = self.get_all_skethch_lf(actions, prod_score_dic, world)
        sketches = self.get_all_sketches(actions, prod_score_dic, world)
        logger.info("%s skethces generated", len(sketches))

        score_list = []
        for actions, score in sketches:
            score_list.append(score)
        score_vec = torch.stack(score_list, 0)
        lf_prob = F.softmax(score_vec, dim=0)
        m = Categorical(lf_prob)
        lf_sample_t = m.sample()
        lf_sample_idx = lf_sample_t.item()

        sampled_lf_actions, sampled_score = sketches[lf_sample_idx]
        sampled_log_probs = m.log_prob(lf_sample_t)

        slot_rep = self._gen_slot_rep(prod_score_dic, sampled_lf_actions)

        return (sampled_lf_actions, sampled_log_probs, slot_rep) 
    
    def forward_enumerate(self, 
                token_reps: List[torch.Tensor],
                world: WikiTablesVariableFreeWorld) -> Dict:
        actions = world.get_valid_actions()
        actions = self._filter_abstract(actions)

        prod_score_dic = self._score_prod(token_reps, actions, world)
        # sketch_lfs = self.get_all_skethch_lf(actions, prod_score_dic, world)
        sketches = self.get_all_sketches(actions, prod_score_dic, world)
        logger.info("%s skethces generated", len(sketches))

        score_list = []
        for actions, score in sketches:
            score_list.append(score)
        score_vec = torch.stack(score_list, 0)
        lf_prob = F.softmax(score_vec, dim=0)

        for i, (lf_actions, lf_score) in enumerate(sketches):
            slot_rep = self._gen_slot_rep(prod_score_dic, lf_actions)
            yield (lf_actions, torch.log(lf_prob[i]), slot_rep)
    
    def predict(self, 
                token_reps: List[torch.Tensor],
                world: WikiTablesVariableFreeWorld) -> Dict:
        actions = world.get_valid_actions()
        actions = self._filter_abstract(actions)

        prod_score_dic = self._score_prod(token_reps, actions, world)
        # sketch_lfs = self.get_all_skethch_lf(actions, prod_score_dic, world)
        sketches = self.get_all_sketches(actions, prod_score_dic, world)
        logger.info("%s skethces generated", len(sketches))

        score_list = []
        for actions, score in sketches:
            score_list.append(score)
        score_vec = torch.stack(score_list, 0)
        lf_prob = F.softmax(score_vec, dim=0)

        max_v, max_id = torch.max(lf_prob, dim=0)

        max_lf_actions, max_score = sketches[max_id]
        max_log_probs = torch.log(lf_prob[max_id])

        slot_rep = self._gen_slot_rep(prod_score_dic, max_lf_actions)

        return (max_lf_actions, max_log_probs, slot_rep) 

    

    def _gen_slot_rep(self, 
                        prod_score_dic: Dict,
                        actions: List[str]) -> Dict:
        """
        genetate representation for each slot 
        """
        one_row_select_actions = []
        mul_row_select_actions = []
        str_column_actions = []
        date_column_actions = []
        num_column_actions = []

        action_ids = []
        for i, action in enumerate(actions):
            action_ids.append(self.prod2id[action])

            if action == "r -> one_row_select":
                one_row_select_actions.append(i)
            elif action == "e -> mul_rows_select":
                mul_row_select_actions.append(i)
            elif action == "t -> string_column:#PH#":
                str_column_actions.append(i)
            elif action == "f -> number_column:#PH#":
                num_column_actions.append(i)
            elif action == "m -> date_column:#PH#":
                date_column_actions.append(i)
        action_id_v = torch.LongTensor(action_ids)
        action_vec = self.prod_embed(action_id_v)

        action_vec = action_vec.unsqueeze(0)
        lstm_output, (ht, ct) = self.prod_rnn(action_vec)
        lstm_output = lstm_output.squeeze(0)

        info_dict = defaultdict(list)
        if len(one_row_select_actions) != 0:
            for item in one_row_select_actions:
                info_dict["one_row_select"].append((item, lstm_output[item]))
        if len(mul_row_select_actions) != 0:
            for item in mul_row_select_actions:
                info_dict["mul_rows_select"].append((item, lstm_output[item]))
        if len(str_column_actions) != 0:
            for item in str_column_actions:
                info_dict["string_column"].append((item, lstm_output[item]))
        if len(num_column_actions) != 0:
            for item in num_column_actions:
                info_dict["number_column"].append((item, lstm_output[item]))
        if len(date_column_actions) != 0:
            for item in date_column_actions:
                info_dict["date_column"].append((item, lstm_output[item]))
        return info_dict

    
    def _score_prod(self, 
                    token_reps: torch.Tensor,
                    actions: List[str],
                    world: WikiTablesVariableFreeWorld) -> Dict:
        """
        produce scores for each production rule
        return dict that has a scalar score for each production rule 
        """
        sent_len, _token_rnn_feat_size = token_reps.size()
        assert self.token_rnn_feat_size == _token_rnn_feat_size
        action_list = []
        for k, v in actions.items():
            action_list += v
        action_list += [f"{START_SYMBOL} -> {type_}" for type_ in world.get_valid_starting_types()]
        action_list = list(set(action_list))
        prod_num = len(action_list)
        
        prod_id_list = [self.prod2id[prod] for prod in action_list]
        prod_id_tensor = torch.LongTensor(prod_id_list)
        prod_id_vec = self.prod_embed(prod_id_tensor) # prod_num * prod_embed_size
        token_mat = token_reps.unsqueeze(0).expand(prod_num, sent_len,
                                                            self.token_rnn_feat_size)
        
        att_scores = self.att(prod_id_vec, token_mat)# prod_num * sent_len
        att_rep_vec = torch.mm(att_scores, token_reps) # prod_num * token_embed
        feat_vec = torch.cat([prod_id_vec, att_rep_vec], 1)
        hiddden_vec = F.relu(self.hidden_func(feat_vec))
        score_vec = self.score_func(hiddden_vec) # prod_num * 1
        score_vec = score_vec.squeeze(1) # prod_num

        prod_score_dic = dict()
        for i, prod in enumerate(action_list):
            prod_score_dic[prod] = score_vec[i]
        return prod_score_dic

    
    def _walk(self,
                actions: Dict,
                prod_score_dic: Dict,
                world: WikiTablesVariableFreeWorld) -> List:
        """
        search in the action space without data selection operations like lookup.
        the operations used reflect the semantics of a question, it is more abstract and the space would be much smaller
        """
        # Buffer of NTs to expand, previous actions
        incomplete_paths = [([str(type_)], [f"{START_SYMBOL} -> {type_}"], 
                            prod_score_dic[f"{START_SYMBOL} -> {type_}"] ) 
                            for type_ in world.get_valid_starting_types()]

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
                    if action not in ["e -> mul_row_select", "r -> one_row_select"] and action in history: continue
                    new_history = history + [action]
                    new_nonterminal_buffer = nonterminal_buffer[:]
                    # Since we expand the last action added to the buffer, the left child should be
                    # added after the right child.
                    for right_side_part in reversed(self._get_right_side_parts(action)):
                        if types.is_nonterminal(right_side_part):
                            new_nonterminal_buffer.append(right_side_part)
                    new_prod_score = prod_score_dic[action] + cur_score
                    next_paths.append((new_nonterminal_buffer, new_history, new_prod_score))
            incomplete_paths = []
            for nonterminal_buffer, path, score in next_paths:
                # An empty buffer means that we've completed this path.
                if not nonterminal_buffer:
                    # if path only has two operations, it is start->string:
                    if len(path) > 2:
                        _completed_paths.append((path, score))
                elif len(path) < self._max_path_length:
                    incomplete_paths.append((nonterminal_buffer, path, score))
        return _completed_paths
    
    def get_all_sketches(self,
                            actions: Dict,
                            prod_score_dic: Dict,
                            world: WikiTablesVariableFreeWorld) -> List:
        paths = self._walk(actions, prod_score_dic, world)
        _score_sorted_paths = sorted(paths, key=lambda x: x[1])
        return _score_sorted_paths
    

    def get_all_skethch_lf(self,
                            actions: Dict,
                            prod_score_dic: Dict,
                            world: WikiTablesVariableFreeWorld) -> List:
        paths = self.get_all_sketches(actions, prod_score_dic, world)
        logical_forms = [(world.get_logical_form(path), score) for (path, score) in paths]
        return logical_forms
    
    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        if "[" in right_side:
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts = [right_side]
        return right_side_parts

    @staticmethod 
    def _filter_abstract(actions: List[str]) -> List[str]:
        """
        sketch logics that does not include filter and same as; columns are replaced with placeholder
        note that the is_termianl function of world.py is changed.
        """

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

        new_action_dic = defaultdict(list)
        pruned = []
        for key_type, action_list in actions.items():
            # keep only one just for placeholder
            if key_type in ["m", "f", "t"]:
                # new_action_dic[key_type] = [action_list[0]]
                new_action_template = action_list[0]
                new_action_items = new_action_template.split("_column:")
                new_action = "_column:".join(new_action_items[:-1] + ["#PH#"])
                new_action_dic[key_type] = [new_action]
                continue
            
            # not using any selection operation
            for action in action_list:
                lhs, rhs = action.split(" -> ")
                if rhs.startswith("filter") or rhs.startswith("same_as"):
                    pruned.append(rhs)
                elif rhs in ["date", "-1"]: # not composing date value
                    pruned.append(rhs)
                elif "string:" in rhs or is_float(rhs): # no non-terminals
                    pruned.append(rhs)
                else:
                    new_action_dic[key_type].append(action)

        new_action_dic["e"].remove("e -> all_rows")

        # add placeholder for each type    
        # new_action_dic["s"].append("s -> string:#PH#")
        # new_action_dic["n"].append("n -> num:#PH#")
        # new_action_dic["d"].append("d -> date:#PH#")

        # print(f"Pruned actions: {' '.join(pruned)}")
        return new_action_dic