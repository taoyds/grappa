import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Dict, List
from collections import defaultdict

from sempar.domain_languages.wikisql_language import WikiSQLLanguage
from allennlp.common.util import START_SYMBOL
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet


class Seq2Seq(nn.Module):
    """
    Grammar based model
    """
    def __init__(self,
                encode_hidden_size: int,
                sketch_embed: nn.Embedding,
                sketch_prod2id: Dict,
                sketch_decoder_rnn_size: int,
                device: torch.device):
        super(Seq2Seq, self).__init__()
        assert encode_hidden_size == sketch_decoder_rnn_size
        self.device = device

        self.encode_hidden_size = encode_hidden_size
        self.sketch_embed = sketch_embed
        self.sketch_prod_embed_size = sketch_embed.embedding_dim
        self.sketch_prod2id = sketch_prod2id
        self.sketch_decoder_rnn_size = sketch_decoder_rnn_size

        # decode
        # self.first_action_embed = nn.Parameter(nn.init.normal_(torch.empty(self.sketch_prod_embed_size)).to(device))
        self.decoder_lstm = nn.LSTMCell(self.sketch_prod_embed_size, sketch_decoder_rnn_size)
        self._max_decoding_steps = 20

        # score
        self.score_action_mlp_1 = nn.Linear(sketch_decoder_rnn_size * 2, sketch_decoder_rnn_size)
        self.score_action_mlp_2 = nn.Linear(sketch_decoder_rnn_size, len(sketch_prod2id))
        self.score_action = lambda x: self.score_action_mlp_2(torch.tanh(self.score_action_mlp_1(x)))

    
    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        if "[" == right_side[0]:
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts = [right_side]
        return right_side_parts
    

    def _get_initial_state(self,
                        token_rep: torch.Tensor) -> RnnStatelet:
        """
        The hidden state of the first hidden state is initialized by token_rep
        It consume the encoder output as the first action
        """
        hidden_state = torch.zeros(1, self.sketch_decoder_rnn_size).to(self.device)
        memory_cell = torch.zeros(1, self.sketch_decoder_rnn_size).to(self.device)
        initial_rnn_state = RnnStatelet(hidden_state, memory_cell, 
                token_rep.unsqueeze(0), 
                None, None, None)
        return initial_rnn_state
        

    def forward(self, 
                world: WikiSQLLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor,
                sketch_actions: List):
        """
        Input: a sequence of sketch actions
        """
        action_dict = world._get_sketch_productions(
                world.get_nonterminal_productions())
        initial_rnn_state = self._get_initial_state(token_rep)

        seq_likeli = []
        rnn_state = initial_rnn_state
        for i, prod in enumerate(sketch_actions):
            left_side, _ = prod.split(" -> ")
            candidates = action_dict[left_side]
            candidate_ids = [self.sketch_prod2id[ac] for ac in candidates]

            cur_hidden, cur_memory = rnn_state.hidden_state, rnn_state.memory_cell
            next_hidden, next_memory = self.decoder_lstm(rnn_state.previous_action_embedding,
                                                        (cur_hidden, cur_memory))
            hidden_tran = next_hidden.transpose(0, 1)
            att_feat_v = torch.mm(token_encodes, hidden_tran) # sent_len * 1
            att_v = F.softmax(att_feat_v, dim=0)
            att_ret_v = torch.mm(att_v.transpose(0, 1), token_encodes)
            
            score_feat_v = torch.cat([next_hidden, att_ret_v], 1)
            score_v = self.score_action(score_feat_v).squeeze()
            filter_score_v_list = [score_v[_id] for _id in candidate_ids]
            filter_score_v = torch.stack(filter_score_v_list, 0)
            log_likeli = F.log_softmax(filter_score_v, dim=0)

            gold_id = candidate_ids.index(self.sketch_prod2id[prod])
            seq_likeli.append(log_likeli[gold_id])

            next_action_embed = self.sketch_embed.weight[self.sketch_prod2id[prod]].unsqueeze(0)
            rnn_state = RnnStatelet(next_hidden, next_memory, next_action_embed, None, None, None)

        return sum(seq_likeli) 


    def decode(self, 
                world: WikiSQLLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor):
        """
        Input: a sequence of sketch actions
        Output: the most probable sequence
        """
        action_dict = world._get_sketch_productions(
                world.get_nonterminal_productions())
        initial_rnn_state = self._get_initial_state(token_rep)

        stack = [START_SYMBOL]
        history = []
        rnn_state = initial_rnn_state
        for i in range(self._max_decoding_steps):
            if len(stack) == 0: break

            cur_non_terminal = stack.pop()
            if cur_non_terminal not in action_dict: continue
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
            score_v = self.score_action(score_feat_v).squeeze()
            filter_score_v_list = [score_v[_id] for _id in candidate_ids]
            filter_score_v = torch.stack(filter_score_v_list, 0)
            prob_v = F.softmax(filter_score_v, dim=0)

            _, pred_id = torch.max(prob_v, dim=0)
            pred_id = pred_id.cpu().item()

            next_action_embed = self.sketch_embed.weight[candidate_ids[pred_id]].unsqueeze(0)
            rnn_state = RnnStatelet(next_hidden, next_memory, next_action_embed, None, None, None)

            prod = candidates[pred_id]
            history.append(prod)
            non_terminals = self._get_right_side_parts(prod) 
            stack += list(reversed(non_terminals))

        return tuple(history) 


    def beam_decode(self, 
                world: WikiSQLLanguage, 
                token_rep: torch.Tensor,
                token_encodes: torch.Tensor,
                beam_size: int):
        """
        Input: a sequence of sketch actions
        Output: output top-k most probable sequence
        """
        action_dict = world._get_sketch_productions(
                world.get_nonterminal_productions())
        initial_rnn_state = self._get_initial_state(token_rep)

        incomplete = [ ([START_SYMBOL], [], initial_rnn_state, None) ] # stack,history,rnn_state
        completed = []
    
        for i in range(self._max_decoding_steps):
            next_paths = []
            for stack, history, rnn_state, seq_score in incomplete:
                cur_non_terminal = stack.pop()
                if cur_non_terminal not in action_dict: continue
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
                score_v = self.score_action(score_feat_v).squeeze()
                filter_score_v_list = [score_v[_id] for _id in candidate_ids]
                filter_score_v = torch.stack(filter_score_v_list, 0)
                prob_v = F.log_softmax(filter_score_v, dim=0)

                pred_logits, pred_ids = torch.topk(prob_v, 
                        min(beam_size, prob_v.size()[0]), dim=0)

                for _logits, _idx in zip(pred_logits, pred_ids):
                    next_action_embed = self.sketch_embed.weight[candidate_ids[_idx]].unsqueeze(0)
                    rnn_state = RnnStatelet(next_hidden, next_memory, next_action_embed, None, None, None)

                    prod = candidates[_idx]
                    _history = history[:]
                    _history.append(prod)
                    non_terminals = self._get_right_side_parts(prod) 
                    _stack = stack[:]
                    for ac in reversed(non_terminals):
                        if ac in action_dict:
                            _stack.append(ac)
                    if seq_score is None:
                        _score = _logits
                    else:
                        _score = _logits + seq_score

                    next_paths.append((_stack, _history, rnn_state, _score))

            incomplete = []
            for stack, history, rnn_state, seq_score in next_paths: 
                if len(stack) == 0:
                    if world.action_sequence_to_logical_form(history) != "#PH#":
                        completed.append((history, seq_score))
                else:
                    incomplete.append((stack, history, rnn_state, seq_score))
                
            if len(completed) > beam_size:
                completed = sorted(completed, key=lambda x:-x[1])
                completed = completed[:beam_size]
                break

            if len(incomplete) > beam_size:
                incomplete = sorted(incomplete, key=lambda x: -x[3])
                incomplete = incomplete[:beam_size]

        return completed