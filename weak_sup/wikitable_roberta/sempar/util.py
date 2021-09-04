from sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage
from typing import List, Dict


def get_left_side_part(action: str) -> str:
    left_side, _ = action.split(" -> ")
    return left_side


def get_right_side_parts(action: str) -> List[str]:
    _, right_side = action.split(" -> ")
    if "[" == right_side[0]:
        right_side_parts = right_side[1:-1].split(", ")
    else:
        right_side_parts = [right_side]
    return right_side_parts


def gen_slot2action_dic(world: WikiTableAbstractLanguage,
                    prod_dict: Dict,
                    sketch_actions: List,
                    program_actions: List):
    ret_dict = dict()
    ref_actions = program_actions[:]
    for i in range(len(sketch_actions)):
        sk_ac = sketch_actions[i]
        left_side, right_side = sk_ac.split(" -> ")
        if right_side == "#PH#":
            if "Column" in left_side:
                ret_dict[i] = ref_actions[0]
                ref_actions = ref_actions[1:]
            else:
                # extract the sub-tree
                tmp_ac_seq = []
                unresovled = [left_side]
                while unresovled:
                    _a = ref_actions[0]
                    _l, _ = _a.split(" -> ")
                    assert _l == unresovled.pop()
                    tmp_ac_seq.append(_a)
                    _r = reversed(get_right_side_parts(_a))
                    for _non_term in _r:
                        if _non_term in prod_dict:
                            unresovled.append(_non_term)
                    ref_actions = ref_actions[1:]
                ret_dict[i] = tmp_ac_seq
        else:
            ref_actions = ref_actions[1:]
    return ret_dict


def check_multi_col(world: WikiTableAbstractLanguage,
                    sketch_actions: List,
                    program_actions: List) -> bool:
    prod_dic = world.get_nonterminal_productions()
    slot_dic = gen_slot2action_dic(world, prod_dic, sketch_actions, program_actions)
    row_slot_acs = []
    col_slot_acs = []
    for idx in slot_dic:
        slot_type = get_left_side_part(sketch_actions[idx])
        if slot_type == "List[Row]":
            row_slot_acs.append(slot_dic[idx])
        else:
            col_slot_acs.append(slot_dic[idx])
    
    if len(row_slot_acs) == 0 or len(col_slot_acs) == 0:
        return False

    for col_slot_ac in col_slot_acs:
        col_name = get_right_side_parts(col_slot_ac)[0]
        for row_slot_ac in row_slot_acs:
            if col_name not in "_".join(row_slot_ac):
                return True
    return False
    