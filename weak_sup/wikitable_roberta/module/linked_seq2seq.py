import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Dict, List
from collections import defaultdict
from nltk.tree import Tree

from sempar.util import get_right_side_parts, gen_slot2action_dic
from sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage
from allennlp.common.util import START_SYMBOL
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet


class LinkedSeq2Seq(nn.Module):
    """
    Grammar based model
    """
    def __init__(self,
                encode_hidden_size: int,
                sketch_embed: nn.Embedding,
                sketch_prod2id: Dict,
                sketch_decoder_rnn_size: int,
                col2action: nn.Linear,
                row2action: nn.Linear,
                col2feat: nn.Linear,
                row2feat: nn.Linear,
                col_feat2score: nn.Bilinear,
                row_feat2score: nn.Bilinear,
                device: torch.device):
        super(LinkedSeq2Seq, self).__init__()
        assert encode_hidden_size == sketch_decoder_rnn_size
        self.device = device

        self.encode_hidden_size = encode_hidden_size
        self.sketch_embed = sketch_embed
        self.sketch_prod_embed_size = sketch_embed.embedding_dim
        self.sketch_prod2id = sketch_prod2id
        self.sketch_id2prod = {v:k for k,v in sketch_prod2id.items()}
        self.sketch_decoder_rnn_size = sketch_decoder_rnn_size

        # decode
        self.first_action_embed = nn.Parameter(nn.init.normal_(torch.empty(self.sketch_prod_embed_size)).to(device))
        self.decoder_lstm = nn.LSTMCell(self.sketch_prod_embed_size, sketch_decoder_rnn_size)
        self._max_decoding_steps = 20

        # score for abstract actions
        self.score_action_mlp_1 = nn.Linear(sketch_decoder_rnn_size * 2, sketch_decoder_rnn_size)
        self.score_action_mlp_2 = nn.Linear(sketch_decoder_rnn_size, len(sketch_prod2id))
        self.score_action = lambda x: self.score_action_mlp_2(torch.tanh(self.score_action_mlp_1(x)))

        self.row2action = row2action
        self.col2action = col2action
        self.row2feat = row2feat
        self.col2feat = col2feat
        self.row_feat2score = row_feat2score
        self.col_feat2score = col_feat2score


    def row2score(self, x: torch.Tensor, y: torch.Tensor):
        return self.row_feat2score(F.relu(self.row2feat(x)), y) 
    

    def col2score(self, x: torch.Tensor, y: torch.Tensor):
        return self.col_feat2score(F.relu(self.col2feat(x)), y)
    

    def _get_initial_state(self,
                        token_rep: torch.Tensor) -> RnnStatelet:
        """
        The hidden state of the first hidden state is initialized by token_rep
        """
        memory_cell = torch.zeros(1, self.sketch_decoder_rnn_size).to(self.device)
        initial_rnn_state = RnnStatelet(token_rep.unsqueeze(0), memory_cell, 
                self.first_action_embed.unsqueeze(0), 
                None, None, None)
        return initial_rnn_state


    def forward(self, 
                world: WikiTableAbstractLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor,
                candidate_rep_dic: torch.Tensor,
                sketch_actions: List,
                program_actions: List):
        """
        Input: a sequence of sketch actions
        """
        prod_action_dict = world.get_nonterminal_productions()
        sketch_action_dict = world._get_sketch_productions(prod_action_dict)
        initial_rnn_state = self._get_initial_state(token_rep)
        slot2action_dic = gen_slot2action_dic(world, prod_action_dict,
                sketch_actions, program_actions)

        program_ref_actions = program_actions[:]
        seq_likeli = []
        rnn_state = initial_rnn_state
        for i, prod in enumerate(sketch_actions):
            left_side, right_side = prod.split(" -> ")

            if right_side != "#PH#":
                candidates = sketch_action_dict[left_side]
                candidate_ids = [self.sketch_prod2id[ac] for ac in candidates]

                cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
                next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                            (cur_hidden, cur_memory))
                hidden_tran = next_hidden.transpose(0, 1)
                att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
                att_v = F.softmax(att_feat_v, dim=0)
                att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
                
                score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
                score_v = self.score_action(score_feat_v).squeeze(0)
                filter_score_v_list = [score_v[_id] for _id in candidate_ids]
                filter_score_v = torch.stack(filter_score_v_list, 0)
                log_likeli = F.log_softmax(filter_score_v, dim=0)

                gold_id = candidate_ids.index(self.sketch_prod2id[prod])
                seq_likeli.append(log_likeli[gold_id])
                next_action_embed = self.sketch_embed.weight[self.sketch_prod2id[prod]].unsqueeze(0)

            else:
                assert left_side == "List[Row]" or "Column" in left_side
                assert i in slot2action_dic

                candidate_v, candidate_actions = candidate_rep_dic[left_side]
                try:
                    gold_id = candidate_actions.index(slot2action_dic[i])
                except:
                    # not included, e.g and/or are order-invariant
                    return None

                # fit the memory for some extreme case
                if len(candidate_actions) > 256:
                    _s = max(0, gold_id - 128)
                    _e = min(gold_id + 128, len(candidate_actions))
                    candidate_v = candidate_v[_s: _e]
                    candidate_actions = candidate_actions[_s: _e]
                    gold_id = candidate_actions.index(slot2action_dic[i])
                    assert gold_id >= 0

                cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
                next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                            (cur_hidden, cur_memory))
                hidden_tran = next_hidden.transpose(0, 1)
                att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
                att_v = F.softmax(att_feat_v, dim=0)
                att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
                score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
                num_candidate = candidate_v.size()[0]

                if left_side == "List[Row]":
                    score_feat_v = score_feat_v.expand(num_candidate, -1)
                    att_over_sel = self.row2score(candidate_v, score_feat_v).squeeze(1)
                    att_over_sel = F.log_softmax(att_over_sel, dim=0)

                    seq_likeli.append(att_over_sel[gold_id])
                    next_action_embed = self.row2action(candidate_v[gold_id]).unsqueeze(0)
                else:
                    score_feat_v = score_feat_v.expand(num_candidate, -1)
                    att_over_col = self.col2score(candidate_v, score_feat_v).squeeze(1)
                    att_over_col = F.log_softmax(att_over_col, dim=0)

                    seq_likeli.append(att_over_col[gold_id])
                    next_action_embed = self.col2action(candidate_v[gold_id]).unsqueeze(0)
                
            rnn_state = RnnStatelet(next_hidden, next_memory, next_action_embed, None, None, None)

        return sum(seq_likeli) 


    def decode(self, 
                world: WikiTableAbstractLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor,
                candidate_rep_dic: torch.Tensor):
        """
        Input: a sequence of sketch actions
        Output: the most probable sequence
        """
        action_dict = world._get_sketch_productions(
                world.get_nonterminal_productions())
        initial_rnn_state = self._get_initial_state(token_rep)

        stack = [START_SYMBOL]
        history = []
        sketch_history = []
        rnn_state = initial_rnn_state
        for i in range(self._max_decoding_steps):
            if len(stack) == 0: break

            cur_non_terminal = stack.pop()
            if cur_non_terminal not in action_dict: continue

            if cur_non_terminal == "List[Row]":
                candidate_v, candidate_actions = candidate_rep_dic[cur_non_terminal]
                cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
                next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                            (cur_hidden, cur_memory))
                hidden_tran = next_hidden.transpose(0, 1)
                att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
                att_v = F.softmax(att_feat_v, dim=0)
                att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
                score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
                num_candidate = candidate_v.size()[0]

                score_feat_v = score_feat_v.expand(num_candidate, -1)
                att_over_sel = self.row2score(candidate_v, score_feat_v).squeeze(1)
                att_over_sel = F.softmax(att_over_sel, dim=0)
                _, pred_id = torch.max(att_over_sel, dim=0)
                pred_id = pred_id.cpu().item()

                next_action_embed = self.row2action(candidate_v[pred_id]).unsqueeze(0)
                history += candidate_actions[pred_id]
                sketch_history.append("List[Row] -> #PH#")
            elif "Column" == cur_non_terminal[-6:]:
                candidate_v, candidate_actions = candidate_rep_dic[cur_non_terminal]
                cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
                next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                            (cur_hidden, cur_memory))
                hidden_tran = next_hidden.transpose(0, 1)
                att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
                att_v = F.softmax(att_feat_v, dim=0)
                att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
                score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
                num_candidate = candidate_v.size()[0]

                score_feat_v = score_feat_v.expand(num_candidate, -1)
                att_over_col = self.col2score(candidate_v, score_feat_v).squeeze(1)
                att_over_col = F.softmax(att_over_col, dim=0)
                _, pred_id = torch.max(att_over_col, dim=0)
                pred_id = pred_id.cpu().item()

                next_action_embed = self.col2action(candidate_v[pred_id]).unsqueeze(0)
                history.append(candidate_actions[pred_id])
                sketch_history.append(f"{cur_non_terminal} -> #PH#")
            else:
                candidates = action_dict[cur_non_terminal]
                candidate_ids = [self.sketch_prod2id[ac] for ac in candidates]

                cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
                next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                            (cur_hidden, cur_memory))
                hidden_tran = next_hidden.transpose(0, 1)
                att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
                att_v = F.softmax(att_feat_v, dim=0)
                att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
                
                score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
                score_v = self.score_action(score_feat_v).squeeze(0)
                filter_score_v_list = [score_v[_id] for _id in candidate_ids]
                filter_score_v = torch.stack(filter_score_v_list, 0)
                prob_v = F.softmax(filter_score_v, dim=0)

                _, pred_id = torch.max(prob_v, dim=0)
                pred_id = pred_id.cpu().item()

                next_action_embed = self.sketch_embed.weight[candidate_ids[pred_id]].unsqueeze(0)

                prod = candidates[pred_id]
                history.append(prod)
                sketch_history.append(prod)
                non_terminals = get_right_side_parts(prod) 
                for _a in reversed(non_terminals):
                    if _a in action_dict:
                        stack.append(_a)

            rnn_state = RnnStatelet(next_hidden, next_memory, next_action_embed, None, None, None)

        return tuple(sketch_history), tuple(history)

    def beam_decode(self, 
                world: WikiTableAbstractLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor,
                candidate_rep_dic: torch.Tensor):
        """
        Input: a sequence of sketch actions
        Output: the most probable sequence
        """
        # TODO!!
        pass