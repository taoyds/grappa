import torch
import logging
from torch import nn
import random
import heapq
from operator import itemgetter


import torch.nn.functional as F
from typing import Union, List, Dict, Any, Set
from collections import defaultdict

from wikisql_roberta.sempar.action_walker import ActionSpaceWalker
from wikisql_roberta.sempar.context.wikisql_context import WikiSQLContext
from wikisql_roberta.sempar.domain_languages.wikisql_language import WikiSQLLanguage
from wikisql_roberta.model.util import construct_row_selections, construct_junction, construct_same
from wikisql_roberta.model.baseline import Programmer
from wikisql_roberta.module.lattice import Lattice

from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.modules.attention.bilinear_attention import BilinearAttention
from allennlp.modules.attention.linear_attention import LinearAttention

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def log_sum_exp(score_list: List):
    if isinstance(score_list, list):
        score_v = torch.stack(score_list, 0)
    else:
        score_v = score_list
    ret_v = score_v - F.log_softmax(score_v, dim=0)
    ret_scalar = ret_v.mean(0)
    return ret_scalar

class StructProgrammer(Programmer):
    """
    Override the slot filling process of programming with structured attention
    """
    def __init__(self, *args):
        super(StructProgrammer, self).__init__(*args)
        # self.align2feat = nn.Linear(self.rnn_hidden_size * 2, self.slot_hidden_score_size)
        self.null_token = nn.Parameter(nn.init.normal_(torch.empty(self.rnn_hidden_size * 2)).to(self.device))
        self.score_func_1 = nn.Linear(self.rnn_hidden_size * 2 + self.sketch_prod_rnn_hidden_size * 2, 
                self.slot_hidden_score_size)
        self.score_func_2 = nn.Linear(self.slot_hidden_score_size, 1)
        self.score_mlp = lambda x: self.score_func_2(self.slot_dropout(F.relu(self.score_func_1(x))))

        self.match_row_slot_1 = nn.Linear(self.rnn_hidden_size * 4 + self.op_embed_size + \
            self.token_embed_size + self.column_type_embed_size + self.column_indicator_size,
            self.slot_hidden_score_size)
        self.match_row_slot_2 = nn.Linear(self.slot_hidden_score_size, 1)
        self.match_row_slot = lambda x: self.match_row_slot_2(self.slot_dropout(F.relu(self.match_row_slot_1(x))))

        self.match_col_slot_1 = nn.Linear(self.token_embed_size + self.column_type_embed_size + 
            self.column_indicator_size + self.rnn_hidden_size * 2, self.slot_hidden_score_size)
        self.match_col_slot_2 = nn.Linear(self.slot_hidden_score_size, 1)
        self.match_col_slot = lambda x: self.match_col_slot_2(self.slot_dropout(F.relu(self.match_col_slot_1(x))))

        # for selecting candidates
        self.ROW_SLOT_BOUND = 5
        self.COL_SLOT_BOUND = 6

        # for alignment ranking, only useful for recursive search 
        self.ALIGN_NUM_BOUND = 512

        # in case it won't fit in the limited memory
        self.CANDIDATE_ACTION_NUM_BOUND = 16

        self.lattice = Lattice(self.device)
    
    
    def collect_candidate_scores(self, 
                            world:WikiSQLLanguage, 
                            token_encodes:torch.Tensor, 
                            candidate_rep_dic: Dict,
                            sketch_encodes:torch.Tensor, 
                            slot_dict: Dict):
        """
        Collect candidate score for each slot
        """
        aligned_rep = self.collect_aligned_rep(world, token_encodes, sketch_encodes, slot_dict)

        ret_score_dict = dict()
        for idx in slot_dict:
            slot_type = slot_dict[idx]
            if slot_type == "List[Row]":
                slot_rep_v = aligned_rep[idx]
                candidate_v, candidate_a = candidate_rep_dic[slot_type]
                num_candidate = candidate_v.size()[0]

                slot_rep_v = slot_rep_v.unsqueeze(0).expand(num_candidate, -1)
                feat2score_v = torch.cat([slot_rep_v, candidate_v], 1)
                att_over_sel = self.match_row_slot(feat2score_v).squeeze()

                att_over_sel = F.log_softmax(att_over_sel, dim=0)
                ret_score_dict[idx] = att_over_sel
            else:
                assert "Column" in slot_type
                slot_rep_v = aligned_rep[idx]
                candidate_v, candidate_a = candidate_rep_dic[slot_type]
                num_candidate = candidate_v.size()[0]
                # candidate_feat_v = self.col2feat(candidate_v)

                slot_rep_v = slot_rep_v.unsqueeze(0).expand(num_candidate, -1)
                feat2score_v = torch.cat([slot_rep_v, candidate_v], 1)

                # att_over_col = torch.mm(candidate_feat_v, slot_rep_v).transpose(0,1) # num_slot * num_column
                att_over_col = self.match_col_slot(feat2score_v).squeeze()
                att_over_col = F.log_softmax(att_over_col, dim=0)
                ret_score_dict[idx] = att_over_col
        return ret_score_dict

    
    def collect_aligned_rep(self, 
                        world: WikiSQLLanguage,
                        token_encodes: torch.Tensor, 
                        sketch_encodes: torch.Tensor,
                        slot_dic: Dict):
        """
        Each word should be aligned to at least one slot or sketch itself softly. 
        Return: slot_reps that maps idx to its represenation
        """
        id2rep = {}
        for _idx in slot_dic:
            id2rep[_idx] = sketch_encodes[_idx]
        slot_reps = self.structured_attention(token_encodes, world, slot_dic, id2rep) 
        return slot_reps

        
    def structured_attention(self, 
                token_encodes: torch.Tensor,
                world: WikiSQLLanguage,
                id2slot_type: Dict,
                id2slot_rep: Dict):
        """
        Each slot is aligned with a span of the question
        Return an aligned representation for each slot
            aligned_dic: slot_id to vector
        """
        sent_len, feat_size = token_encodes.size()
        ent_spans = world.table_context.get_entity_spans()

        # special symbol for all_rows
        ent_spans.append(None)

        # computes all the scores for each alignment between span and slot
        cache_dic_score = defaultdict(dict)
        span_rep_cache = {}
        slot_signs = set()
        for idx, slot_type in id2slot_type.items():
            slot_rep = id2slot_rep[idx]
            slot_signs.add((idx, slot_type))
            if slot_type != "List[Row]":
                slot_rep = slot_rep.unsqueeze(0).expand(sent_len, -1)
                score_feat_v = torch.cat([token_encodes, slot_rep], 1)
                score_v = self.score_mlp(score_feat_v).squeeze(1)
                top_k_scores, top_k_idx = torch.topk(score_v, min(self.COL_SLOT_BOUND, sent_len), dim=0)
                for _t_idx in top_k_idx.cpu():
                    _t_idx = _t_idx.item()
                    cache_dic_score[idx, slot_type][_t_idx] = score_v[_t_idx]
            else:
                _score_dic = dict()
                for span in ent_spans:
                    if span is None:
                        if span not in span_rep_cache:
                            span_rep_cache[span] = self.null_token
                        span_v = span_rep_cache[span]
                        score_feat_v = torch.cat([span_v, slot_rep], 0).unsqueeze(0)
                        score_v = self.score_mlp(score_feat_v).squeeze()
                        _score_dic[span] = score_v
                    else:
                        s, e = span # inclusive
                        for _s in range( max(0, s - self.ROW_SLOT_BOUND), s + 1):
                            for _e in range(e, min(e + self.ROW_SLOT_BOUND + 1, sent_len)):
                                _span = (_s, _e)
                                if _span not in span_rep_cache:
                                    span_v = torch.mean(token_encodes[_s: _e + 1], 0) 
                                    span_rep_cache[_span] = span_v
                                span_v = span_rep_cache[_span]
                                score_feat_v = torch.cat([span_v, slot_rep], 0).unsqueeze(0)
                                score_v = self.score_mlp(score_feat_v).squeeze()
                                _score_dic[_span] = score_v
                cache_dic_score[idx, slot_type] = _score_dic

        slot_signs = list(slot_signs)
        # align_dict = self.recur_search(cache_dic_score, len(id2slot_type), \
        #        sent_len, slot_signs, span_rep_cache, token_encodes)
        align_dict = self.dp_search(cache_dic_score, len(id2slot_type), \
                sent_len, slot_signs, span_rep_cache, token_encodes)

        return align_dict


    def recur_search(self,
                      cache_dic_score: Dict,
                      num_slot: int,
                      num_tokens: int,
                      slot_signs: List,
                      span_rep_cache: torch.Tensor,
                      token_encodes: torch.Tensor):
        """
        Exhausitve search of all legitimate alignments
        cache_dic_scores: cache all the scores for each slot
        slot_signs: (idx, slot_type) list
        """
        # search all possible assignments
        possible_paths = []
        path_scores = []
        def recur_find(completed, uncompleted, score, idx_set):
            if len(completed) == num_slot and completed not in possible_paths:
                possible_paths.append(completed)
                path_scores.append(score)
                return
            else:
                for _slot_sign in uncompleted:
                    if _slot_sign[1] == "List[Row]":
                        for span in cache_dic_score[_slot_sign]:
                            if span is None:
                                _span_idxs = set()
                            else:
                                _span_idxs = set(range(span[0], span[1] + 1))
                            if len(_span_idxs.intersection(idx_set)) > 0:
                                continue
                            _item_score = cache_dic_score[_slot_sign][span]
                            _completed = completed.copy()
                            _completed[_slot_sign] = span 
                            _score = score + _item_score
                            _uncompleted = uncompleted.copy()
                            _uncompleted.remove(_slot_sign)
                            _idx_set = idx_set.copy()
                            _idx_set = _idx_set.union(_span_idxs)

                            recur_find(_completed, _uncompleted, _score, _idx_set)
                    else:
                        score_v = cache_dic_score[_slot_sign]
                        for _i in range(num_tokens):
                            if _i in idx_set or _i not in cache_dic_score[_slot_sign]:
                                continue
                            _completed = completed.copy()
                            _completed[_slot_sign] = _i
                            _score = score + score_v[_i]
                            _uncompleted = uncompleted.copy()
                            _uncompleted.remove(_slot_sign)
                            _idx_set = idx_set.copy()
                            _idx_set.add(_i)

                            recur_find(_completed, _uncompleted, _score, _idx_set)

        recur_find({}, slot_signs, torch.Tensor([0]).to(self.device), set())

        # print("partition function", log_sum_exp(path_scores))

        # filter top k path
        if len(possible_paths) <= self.ALIGN_NUM_BOUND:
            top_k_paths = possible_paths
            top_k_scores = path_scores
        else:
            _all_score_v = torch.cat(path_scores, 0) 
            _, _top_k_idx = torch.topk(_all_score_v, self.ALIGN_NUM_BOUND, dim=0)
            top_k_paths = []
            top_k_scores = []
            for _id in _top_k_idx.cpu():
                top_k_paths.append(possible_paths[_id])
                top_k_scores.append(path_scores[_id])

        logger.info(f"{len(top_k_paths)} paths founded after filtering")

        assign_score_v = torch.cat(top_k_scores, 0)
        assign_prob = F.softmax(assign_score_v, dim=0)

        # for debug 
        if not self.training:
            if len(possible_paths) > self.ALIGN_NUM_BOUND:
                best_path = top_k_paths[0]
            else:
                _, max_ind = assign_score_v.max(0)
                max_ind = max_ind.cpu().item()
                best_path = possible_paths[max_ind]
            logger.info(f"Best alignment: {best_path}")

        # aggregate representations
        verized_dict = defaultdict(list)
        for _i, _path in enumerate(top_k_paths):
            for _slot_sign in _path:
                _slot_idx = _path[_slot_sign]
                if _slot_idx in span_rep_cache:
                    _rep = span_rep_cache[_slot_idx]
                else:
                    _rep = token_encodes[_slot_idx] 
                verized_dict[_slot_sign[0]].append(_rep)
        
        # sample a representation
        align_dict = dict()
        for _idx in verized_dict:
            v_list = verized_dict[_idx]
            _rep_v = torch.stack(v_list, 1)
            _rep = torch.mm(_rep_v, assign_prob.unsqueeze(1))
            align_dict[_idx] = _rep.squeeze(1)

        return align_dict
    
    def dp_search(self,
            cache_dic_score: Dict,
            num_slot: int,
            num_tokens: int,
            slot_signs: List,
            span_rep_cache: torch.Tensor,
            token_encodes: torch.Tensor):
        """
        Computing the marginal probabilities for each slot-span alignment based on dp
        Return: align_dict, slot_id to vector
        """
        align_dict = dict()

        # if List[Row] only has None, then it's directly assigned
        lattice_weight = defaultdict(dict)
        new_slot_signs = []
        for slot_sign in cache_dic_score:
            if len(cache_dic_score[slot_sign]) == 1:
                assert None in cache_dic_score[slot_sign]
                align_dict[slot_sign[0]] = span_rep_cache[None]
            else:
                new_slot_signs.append(slot_sign)
                lattice_weight[slot_sign] = cache_dic_score[slot_sign]

        assert len(lattice_weight) == len(new_slot_signs)
        aligned_prob = self.lattice(lattice_weight, len(new_slot_signs), num_tokens, new_slot_signs)

        for slot_sign in aligned_prob:
            items = []
            for span in aligned_prob[slot_sign]:
                if isinstance(span, int):
                    items.append(aligned_prob[slot_sign][span] * token_encodes[span])
                else:
                    items.append(aligned_prob[slot_sign][span] * span_rep_cache[span])
            assert len(items) > 0
            align_dict[slot_sign[0]] = sum(items)
        
        assert len(align_dict) == len(slot_signs)
        return align_dict