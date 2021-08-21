import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Dict, List
from collections import defaultdict


def log_sum_exp(score_list: List):
    if isinstance(score_list, list):
        score_v = torch.stack(score_list, 0)
    else:
        score_v = score_list
    ret_v = score_v - F.log_softmax(score_v, dim=0)
    ret_scalar = ret_v.mean(0)
    return ret_scalar

def log_sum_exp_2d(score_v: torch.Tensor):
    ret_v = score_v - F.log_softmax(score_v, dim=0)
    ret_scalar = ret_v.mean(0)
    return ret_scalar

class Lattice(nn.Module):
    """
    Argumented semi-crf with uniqueness constraint
    """
    def __init__(self, device):
        super(Lattice, self).__init__()
        self.device = device

    def get_states(self, 
            idx: int,
            nums: int,
            flag: int):
        idxs = []
        for i in range(2 ** nums):
            _i = i >> idx
            if bin(_i)[-1] == str(flag):
                idxs.append(i)
        return idxs
                
    
    def _forward_alg(self,
            lattice_weight: Dict,
            num_slot: int,
            num_tokens: int,
            slot_signs: List):
        """
        lattice_weight: (slot_id, slot_type) to dict of span/token to a score
        slot_signs: a list of (slot_id, slot_type)
        """
        # special symbol of none
        init_alphas = torch.Tensor(num_tokens + 2, 2 ** num_slot).\
            fill_(float("nan")).to(self.device)
        
        for i in range(num_tokens + 1):
            _incoming_paths = defaultdict(list)
            for j, slot_sign in enumerate(slot_signs):
                slot_id, slot_type = slot_sign
                for k in self.get_states(j, num_slot, 0):
                    if slot_type == "List[Row]":
                        for span in lattice_weight[slot_sign]:
                            if span is not None and span[1] == i:
                                if torch.isnan(init_alphas[span[0], k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span])
                                elif not torch.isnan(init_alphas[span[0],k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span] + 
                                        init_alphas[span[0], k] )
                            elif i == num_tokens and span is None:
                                if torch.isnan(init_alphas[num_tokens, k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span])
                                elif not torch.isnan(init_alphas[num_tokens,k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span] + 
                                        init_alphas[num_tokens, k] )
                    else:
                        assert "Column" in slot_type
                        for t_id in lattice_weight[slot_sign]:
                            if t_id == i:
                                if torch.isnan(init_alphas[i,k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][t_id])
                                elif not torch.isnan(init_alphas[i,k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][t_id] + 
                                        init_alphas[i, k] )

            activated_k = set()
            for k in _incoming_paths:
                scores = _incoming_paths[k]
                if not torch.isnan(init_alphas[i, k]):
                    scores.append(init_alphas[i, k])

                init_alphas[i+1, k] = log_sum_exp(scores)
                activated_k.add(k)
                
            for k in range(2**num_slot):
                if k not in activated_k and not torch.isnan(init_alphas[i, k]):
                    init_alphas[i+1, k] = init_alphas[i, k]

        alpha = init_alphas[-1, 2**num_slot - 1]
        # alpha = log_sum_exp(init_alphas[-1])
        return alpha, init_alphas

    
    def _backward_alg(self,
            lattice_weight: Dict,
            num_slot: int,
            num_tokens: int,
            slot_signs: List):
        init_betas = torch.Tensor(num_tokens + 2, 2 ** num_slot).\
            fill_(float("nan")).to(self.device)
        
        for i in reversed(range(num_tokens + 1)):
            _incoming_paths = defaultdict(list)
            for j, slot_sign in enumerate(slot_signs):
                slot_id, slot_type = slot_sign
                for k in self.get_states(j, num_slot, 1):
                    if slot_type == "List[Row]":
                        for span in lattice_weight[slot_sign]:
                            if span is not None and span[0] == i:
                                if torch.isnan(init_betas[span[1] + 1, k]) and k == 2 ** num_slot - 1:
                                    _incoming_paths[k - 2**j].append(lattice_weight[slot_sign][span])
                                elif not torch.isnan(init_betas[span[1] + 1, k]):
                                    _incoming_paths[k - 2**j].append(lattice_weight[slot_sign][span] + 
                                        init_betas[span[1] + 1, k] )
                            elif i == num_tokens and span is None and k == 2 ** num_slot - 1:
                                assert torch.isnan(init_betas[num_tokens + 1, k])
                                _incoming_paths[k - 2**j].append(lattice_weight[slot_sign][span])
                        
                    else:
                        assert "Column" in slot_type
                        for t_id in lattice_weight[slot_sign]:
                            if t_id == i:
                                if torch.isnan(init_betas[i + 1, k]) and k == 2 ** num_slot - 1:
                                    _incoming_paths[k - 2**j].append(lattice_weight[slot_sign][t_id])
                                elif not torch.isnan(init_betas[i + 1, k]):
                                    _incoming_paths[k - 2**j].append(lattice_weight[slot_sign][t_id] + 
                                        init_betas[i + 1, k] )

            activated_k = set()
            for k in _incoming_paths:
                scores = _incoming_paths[k]
                if not torch.isnan(init_betas[i+1, k]):
                    scores.append(init_betas[i+1, k])

                init_betas[i, k] = log_sum_exp(scores)
                activated_k.add(k)
                
            for k in range(2**num_slot):
                if k not in activated_k and not torch.isnan(init_betas[i+1, k]):
                    init_betas[i, k] = init_betas[i+1, k]
        
        beta = init_betas[0,0]
        # beta = log_sum_exp(init_betas[0])
        return beta, init_betas


    def MAP(self,
            lattice_weight: Dict,
            num_slot: int,
            num_tokens: int,
            slot_signs: List) -> Dict:
        """
        TODO: 
        lattice_weight: (slot_id, slot_type) to dict of span/token to a score
        slot_signs: a list of (slot_id, slot_type)
        ouput: discrete alignment
        """
        # special symbol of none
        init_alphas = torch.Tensor(num_tokens + 2, 2 ** num_slot).\
            fill_(float("nan")).to(self.device)
        backpointers = torch.Tensor(num_tokens + 2, 2 ** num_slot).\
            fill_(float("nan")).to(self.device)
        
        for i in range(num_tokens + 1):
            _incoming_paths = defaultdict(list)
            for j, slot_sign in enumerate(slot_signs):
                slot_id, slot_type = slot_sign
                for k in self.get_states(j, num_slot, 0):
                    if slot_type == "List[Row]":
                        for span in lattice_weight[slot_sign]:
                            if span is not None and span[1] == i:
                                if torch.isnan(init_alphas[span[0], k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span])
                                elif not torch.isnan(init_alphas[span[0],k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span] + 
                                        init_alphas[span[0], k] )
                            elif i == num_tokens and span is None:
                                if torch.isnan(init_alphas[num_tokens, k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span])
                                elif not torch.isnan(init_alphas[num_tokens,k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][span] + 
                                        init_alphas[num_tokens, k] )
                    else:
                        assert "Column" in slot_type
                        for t_id in lattice_weight[slot_sign]:
                            if t_id == i:
                                if torch.isnan(init_alphas[i,k]) and k == 0:
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][t_id])
                                elif not torch.isnan(init_alphas[i,k]):
                                    _incoming_paths[k + 2**j].append(lattice_weight[slot_sign][t_id] + 
                                        init_alphas[i, k] )

            activated_k = set()
            for k in _incoming_paths:
                scores = _incoming_paths[k]
                if not torch.isnan(init_alphas[i, k]):
                    scores.append(init_alphas[i, k])

                max_val = max(scores)
                max_id = scores.index(max_val)
                init_alphas[i+1, k] = max_val
                backpointers[i+1, k] = max_id
                activated_k.add(k)
                
            for k in range(2**num_slot):
                if k not in activated_k and not torch.isnan(init_alphas[i, k]):
                    init_alphas[i+1, k] = init_alphas[i, k]

        alpha = init_alphas[-1, 2**num_slot - 1]
        # alpha = log_sum_exp(init_alphas[-1])
        return alpha, init_alphas

    
    def forward(self, 
            lattice_weight: Dict,
            num_slot: int,
            num_tokens: int,
            slot_signs: List):
        """
        Computing the marginal probabilities for each slot-span alignment based on dp
        Return: align_dict, slot_id to vector
        """
        alpha, alpha_mat = self._forward_alg(lattice_weight, num_slot, num_tokens, slot_signs)
        beta, beta_mat = self._backward_alg(lattice_weight, num_slot, num_tokens, slot_signs)
        # assert alpha == beta
        
        align_prob = defaultdict(dict)
        for slot_sign in lattice_weight:
            slot_idx = slot_signs.index(slot_sign)
            for span in lattice_weight[slot_sign]:
                if isinstance(span, int):
                    s, e = span, span
                elif span is not None:
                    s, e = span
                else:
                    s, e = num_tokens, num_tokens

                pot = []
                for k in self.get_states(slot_idx, num_slot, 0):
                    _p_no_nan = []
                    if not torch.isnan(alpha_mat[s,k]):
                        _p_no_nan.append(alpha_mat[s,k]) 
                    if not torch.isnan(beta_mat[e+1, k + 2**slot_idx]):
                        _p_no_nan.append(beta_mat[e+1, k + 2**slot_idx]) 

                    if len(_p_no_nan) > 0:
                        _p_no_nan.append(lattice_weight[slot_sign][span])
                        pot.append(sum(_p_no_nan))
                    elif num_slot == 1:
                        pot.append(lattice_weight[slot_sign][span])

                if len(pot) == 0:
                    continue
                potential = log_sum_exp(pot)
                prob = torch.exp(potential - alpha)
                align_prob[slot_sign][span] = prob

        return align_prob
