from collections import defaultdict
from numbers import Number
from typing import Dict, List, NamedTuple, Set, Tuple
import logging
import re

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                PredicateType, predicate)
from sempar.context.wikisql_context import WikiSQLContext

logger = logging.getLogger("root")  # pylint: disable=invalid-name


class Row(NamedTuple):
    values: Dict[str, str]

class Column(NamedTuple):
    name: str

class StringColumn(Column):
    pass


class NumberColumn(Column):
    pass



class WikiSQLLanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    similar to wikitable language
    """
    def __init__(self, table_context: WikiSQLContext) -> None:
        super().__init__(start_types={Number, List[str]})
        self.table_context = table_context
        self.table_data = [Row(row) for row in table_context.table_data]

        column_types = table_context.column_types
        if "string" in column_types:
            self.add_predicate('filter_in', self.filter_in)
        if "number" in column_types:
            self.add_predicate('filter_number_greater', self.filter_number_greater)
            self.add_predicate('filter_number_lesser', self.filter_number_lesser)
            self.add_predicate('filter_number_equals', self.filter_number_equals)
            self.add_predicate('max', self.max)
            self.add_predicate('min', self.min)
            self.add_predicate('average', self.average)
            self.add_predicate('sum', self.sum)

        # Adding entities and numbers seen in questions as constants.
        for entity in table_context._entity2id:
            self.add_constant(entity, entity)
        for number in table_context._num2id:
            self.add_constant(str(number), float(number), type_=Number)
        
        # Adding column names as constants.  Each column gets added once for every
        # type in the hierarchy going from its concrete class to the base Column.  String columns
        # get added as StringColumn and Column, and date and number columns get added as DateColumn
        for typed_column_name, column_type in table_context.column2types.items():
            column = None
            if column_type == 'string':
                column = StringColumn(typed_column_name)
            elif column_type == "number":
                column = NumberColumn(typed_column_name)
            else:
                raise NotImplementedError
            self.add_constant(typed_column_name, column)
            self.add_constant(typed_column_name, column, type_=Column)
    
    
    def get_nonterminal_productions(self) -> Dict[str, List[str]]:
        """
        grammar induction may fail in cases where certain non-terminals are not present
        but we still need these nonterminals during search
        TODO: this function includes some hotfixes
        """
        production_dict = super(WikiSQLLanguage, self).get_nonterminal_productions()
        # TODO: hotfix add str entry if is missing
        if "str" not in production_dict:
            production_dict["str"] = []

        if "Number" not in production_dict:
            production_dict["Number"] = []

        return production_dict
    
    @staticmethod 
    def _get_sketch_productions(actions: Dict) -> Dict:
        """
        v1: sketch operations do not include filter, date; columns are replaced with placeholder
        v2: sketch only replace columns, entities like numbers, date function
        """
        new_action_dic = defaultdict(list)
        pruned = []
        for non_terminal, productions in actions.items():
            if non_terminal in ["Column", "StringColumn", "NumberColumn", 
                "DateColumn", "ComparableColumn", "Date", "str"]:
                place_holder_prod = f"{non_terminal} -> #PH#"
                new_action_dic[non_terminal].append(place_holder_prod)
            elif non_terminal in  ["Number"]:
                new_prod_list = []
                for prod in actions[non_terminal]:
                    _, rhs = prod.split(" -> ")
                    try: # if this is a terminal
                        float(rhs)
                        place_holder_prod = f"{non_terminal} -> #PH#"
                        new_prod_list.append(place_holder_prod)
                    except:
                        new_prod_list.append(prod)
                new_action_dic[non_terminal] = new_prod_list
            elif non_terminal == "List[Row]":
                new_prod_list = []
                place_holder_prod = f"{non_terminal} -> #PH#"
                new_prod_list.append(place_holder_prod)
                # same as
                # new_prod_list.append("List[Row] -> [<Row,Column:List[Row]>, Row, Column]")
                new_action_dic[non_terminal] = new_prod_list 
            else:
                new_action_dic[non_terminal] = actions[non_terminal]
        
        return new_action_dic
    
    @staticmethod 
    def _get_slot_productions(actions: Dict) -> Dict:
        """
        filling slots of sketches 
        """
        new_action_dic = defaultdict(list)
        pruned = []
        for non_terminal, productions in actions.items():
            if non_terminal in ["Column", "StringColumn", "NumberColumn", "str", "Date"]:
                new_action_dic[non_terminal] = actions[non_terminal]
            elif non_terminal in  ["Number"]:
                new_prod_list = []
                for prod in actions[non_terminal]:
                    _, rhs = prod.split(" -> ")
                    try: # if this is a terminal
                        float(rhs)
                        new_prod_list.append(prod)
                    except:
                        pass
                new_action_dic[non_terminal] = new_prod_list
            elif non_terminal == "List[Row]":
                new_action_dic[non_terminal] = actions[non_terminal][:]

                for prod in new_action_dic[non_terminal]:
                   _, rhs = prod.split(" -> ")
                   _t = rhs[1:-1].split(", ")[0]
                   if _t in actions:
                       new_action_dic[_t] = actions[_t][:]
        return new_action_dic
    
    @staticmethod
    def get_slot_dict(actions: List) -> Dict:
        """
        Slot_dict: action_index to its type
        """
        slot_dict = dict()
        for action_ind, action in enumerate(actions):
            lhs, rhs =  action.split(" -> ")
            if lhs in ["Column", "StringColumn", "NumberColumn", "ComparableColumn", 
                        "DateColumn", "str", "Number", "Date"] and rhs == "#PH#":
                slot_dict[action_ind] = lhs
            elif lhs == "List[Row]" and rhs == "#PH#":
                slot_dict[action_ind] = lhs
        return slot_dict

    def evaluate_logical_form(self, 
                            logical_form: str, 
                            target_value: List[str]) -> bool:
        try:
            denotation = self.execute(logical_form)
        except ExecutionError as ex:
            logger.warning(f'{ex.message}, Failed to execute: {logical_form}')
            return False
        except Exception as ex:
            err_template = "Exception of type {0} occurred. Arguments:\n{1!r}"
            message = err_template.format(type(ex).__name__, ex.args)
            logger.warning(f'{message}')
            return False

        assert isinstance(target_value, list)
        if not isinstance(denotation, list):
            denotation = [denotation]
        elif isinstance(denotation, list) and len(denotation) == 0:
            denotation = [None]

        normalized_target = [self.table_context.normalize_string(v) if isinstance(v, str) else v for v in target_value]
        return denotation == normalized_target

    # Things below here are language predicates, until you get to private methods.  We start with
    # general predicates that are always included in the language, then move to
    # column-type-specific predicates, which only get added if we see columns of particular types
    # in the table.

    @predicate
    def all_rows(self) -> List[Row]:
        return self.table_data

    @predicate
    def select(self, rows: List[Row], column: Column) -> List[str]:
        """
        Select function takes a list of rows and a column and returns a list of cell values as
        strings.
        """
        return [row.values[column.name] for row in rows]
    
        
    @predicate
    def conjunction(self, row_1: List[Row], row_2: List[Row]) -> List[Row]:
        if len(row_1) == 0 or len(row_2) == 0:
            raise ExecutionError("AND gets empty lists")
        elif row_1 == row_2:
            raise ExecutionError("AND gets the same rows")
        else:
            ret_row = []
            for row in row_1:
                if row in row_2:
                    ret_row.append(row)
            return ret_row


    @predicate
    def count(self, rows: List[Row]) -> Number:
        return len(rows)  # type: ignore


    # These six methods take a list of rows, a column, and a numerical value and return all the
    # rows where the value in that column is [comparator] than the given value.  They only get
    # added to the language if we see a number column in the table.

    def filter_number_greater(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]  # type: ignore

    def filter_number_lesser(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]  # type: ignore

    def filter_number_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]  # type: ignore


    def filter_in(self, rows: List[Row], column: StringColumn, filter_value: str) -> List[Row]:
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.  Also, we need to remove the
        # "string:" that was prepended to the entity name in the language.
        # filter_value = filter_value.lstrip('string:')
        assert filter_value[:7] == "string:"
        filter_value = filter_value[7:]
        exact_match = [row for row in rows if filter_value == row.values[column.name]]
        if len(exact_match) > 0:
            return exact_match
        else:
            return [row for row in rows if filter_value in row.values[column.name]]


    def max(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the max of the values under that column in
        those rows.
        """
        if len(rows) <= 1:
            raise ExecutionError("max recieves too few row")
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return max([value for value, _ in cell_row_pairs])  # type: ignore

    def min(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the min of the values under that column in
        those rows.
        """
        if len(rows) <= 1:
            raise ExecutionError("min recieves too few row")
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return min([value for value, _ in cell_row_pairs])  # type: ignore

    def sum(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the sum of the values under that column in
        those rows.
        """
        if len(rows) <= 1:
            raise ExecutionError("sum recieves too few row")
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return sum([value for value, _ in cell_row_pairs])  # type: ignore

    def average(self, rows: List[Row], column: NumberColumn) -> Number:
        """
        Takes a list of rows and a column and returns the mean of the values under that column in
        those rows.
        """
        if len(rows) <= 1:
            raise ExecutionError("average recieves too few row")
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        if not cell_row_pairs:
            return 0.0  # type: ignore
        return sum([value for value, _ in cell_row_pairs]) / len(cell_row_pairs)  # type: ignore


    # End of language predicates.  Stuff below here is for private use, helping to implement the
    # functions above.

    def __eq__(self, other):
        if not isinstance(other, WikiSQLLanguage):
            return False
        return self.table_data == other.table_data

    @staticmethod
    def _get_number_row_pairs_to_filter(rows: List[Row],
                                        column_name: str) -> List[Tuple[float, Row]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a number taken from that column, and the corresponding row
        as the second element. The output can be used to compare rows based on the numbers.
        """
        if not rows:
            return []
        try:
            # cell_row_pairs = [(float(row.values[column_name].replace('_', '')), row) for row in rows]
            cell_row_pairs = []
            for row in rows:
                _cell_value = row.values[column_name]
                if isinstance(_cell_value, float) or isinstance(_cell_value, int):
                    cell_row_pairs.append((_cell_value, row))

        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        return cell_row_pairs
