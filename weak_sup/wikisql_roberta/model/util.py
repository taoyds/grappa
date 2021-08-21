from typing import List, Any

def construct_row_selections(sel_type:str, operator:str, column_action:str, ent_action: str):
    date_non = 'List[Row] -> [<List[Row],DateColumn,Date:List[Row]>, List[Row], DateColumn, Date]'
    num_non = 'List[Row] -> [<List[Row],NumberColumn,Number:List[Row]>, List[Row], NumberColumn, Number]'
    str_non = 'List[Row] -> [<List[Row],StringColumn,str:List[Row]>, List[Row], StringColumn, str]' 

    num_op_dict = { ">": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater",
                    "<": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser",
                    ">=": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_greater_equals",
                    "<=": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_lesser_equals",
                    "=": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_equals",
                    "!=": "<List[Row],NumberColumn,Number:List[Row]> -> filter_number_not_equals"
                    }

    date_op_dict = { ">": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_greater",
                    "<": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_lesser",
                    ">=": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_greater_equals",
                    "<=": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_lesser_equals",
                    "=": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_equals",
                    "!=": "<List[Row],DateColumn,Date:List[Row]> -> filter_date_not_equals"
                    }

    str_op_dict = { "=": "<List[Row],StringColumn,str:List[Row]> -> filter_in",
                    "!=": "<List[Row],StringColumn,str:List[Row]> -> filter_not_in"
                    }

    all_rows_sel = "List[Row] -> all_rows"

    actions = []
    if sel_type == "number":
        actions.append(num_non)
        actions.append(num_op_dict[operator])
        actions.append(all_rows_sel)
        actions.append(column_action)
        actions.append(ent_action)
    elif sel_type == "date":
        actions.append(date_non)
        actions.append(date_op_dict[operator])
        actions.append(all_rows_sel)
        actions.append(column_action)
        actions.append(ent_action)
    elif sel_type == "string":
        actions.append(str_non)
        actions.append(str_op_dict[operator])
        actions.append(all_rows_sel)
        actions.append(column_action)
        actions.append(ent_action)
    else:
        raise NotImplementedError

    return actions

def construct_junction(sel_type:str, pair_1: tuple, pair_2:tuple):
    junction_non = 'List[Row] -> [<List[Row],List[Row]:List[Row]>, List[Row], List[Row]]'
    or_op = "<List[Row],List[Row]:List[Row]> -> disjunction"
    and_op = "<List[Row],List[Row]:List[Row]> -> conjunction"

    action_seq_1 = pair_1
    action_seq_2 = pair_2

    if sel_type == "or":
        return [junction_non, or_op] + action_seq_1 + action_seq_2
    elif sel_type == "and":
        return [junction_non, and_op] + action_seq_1 + action_seq_2
    else:
        raise NotImplementedError

def construct_same(str_col_ac:str, ent_ac:str, col_ac:str):
    same_non = 'List[Row] -> [<StringColumn,str,Column:List[Row]>, StringColumn, str, Column]'
    same_op = "<StringColumn,str,Column:List[Row]> -> same_as"
    return [same_non, same_op, str_col_ac, ent_ac, col_ac]