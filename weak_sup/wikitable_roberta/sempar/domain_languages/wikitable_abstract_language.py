from collections import defaultdict
# We use "Number" in a bunch of places throughout to try to generalize ints and floats.
# Unfortunately, mypy doesn't like this very much, so we have to "type: ignore" a bunch of things.
# But it makes for a nicer induced grammar, so it's worth it.
from numbers import Number
from typing import Dict, List, NamedTuple, Set, Tuple
import logging
import re

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, ExecutionError,
                                                                PredicateType, predicate)
from allennlp.semparse.contexts.table_question_knowledge_graph import MONTH_NUMBERS
from allennlp.semparse.contexts import TableQuestionContext
from sempar.context.table_question_context import Date
from sempar.evaluator import target_values_map, check_denotation, to_value_list, tsv_unescape_list

logger = logging.getLogger("root")  # pylint: disable=invalid-name


class Row(NamedTuple):
    # Maps column names to cell values
    values: Dict[str, str]


class Column(NamedTuple):
    name: str


class StringColumn(Column):
    pass


class ComparableColumn(Column):
    pass


class DateColumn(ComparableColumn):
    pass


class NumberColumn(ComparableColumn):
    pass



class WikiTableAbstractLanguage(DomainLanguage):
    # pylint: disable=too-many-public-methods,no-self-use
    """
    Derive from wikitable language, but support extract sketches 
    """
    def __init__(self, table_context: TableQuestionContext) -> None:
        super().__init__(start_types={Number, Date, List[str]})
        self.table_context = table_context
        self.table_data = [Row(row) for row in table_context.table_data]

        # if the last colum is total, remove it
        _name = f"string_column:{table_context.column_index_to_name[0]}"
        if _name in table_context.table_data[-1] and  "total" in table_context.table_data[-1][_name]:
            self.table_data.pop()

        column_types = table_context.column_types
        if "string" in column_types:
            self.add_predicate('filter_in', self.filter_in)
            self.add_predicate('filter_not_in', self.filter_not_in)
        if "date" in column_types:
            self.add_predicate('filter_date_greater', self.filter_date_greater)
            self.add_predicate('filter_date_greater_equals', self.filter_date_greater_equals)
            self.add_predicate('filter_date_lesser', self.filter_date_lesser)
            self.add_predicate('filter_date_lesser_equals', self.filter_date_lesser_equals)
            self.add_predicate('filter_date_equals', self.filter_date_equals)
            self.add_predicate('filter_date_not_equals', self.filter_date_not_equals)
        if "number" in column_types or "num2" in column_types:
            self.add_predicate('filter_number_greater', self.filter_number_greater)
            self.add_predicate('filter_number_greater_equals', self.filter_number_greater_equals)
            self.add_predicate('filter_number_lesser', self.filter_number_lesser)
            self.add_predicate('filter_number_lesser_equals', self.filter_number_lesser_equals)
            self.add_predicate('filter_number_equals', self.filter_number_equals)
            self.add_predicate('filter_number_not_equals', self.filter_number_not_equals)
            self.add_predicate('max', self.max)
            self.add_predicate('min', self.min)
            self.add_predicate('average', self.average)
            self.add_predicate('sum', self.sum)
            self.add_predicate('diff', self.diff)
        if "date" in column_types or "number" in column_types or "num2" in column_types:
            self.add_predicate('argmax', self.argmax)
            self.add_predicate('argmin', self.argmin)

        # Adding entities and numbers seen in questions as constants.
        for entity in table_context._entity2id:
            self.add_constant(entity, entity)
        for number in table_context._num2id:
            self.add_constant(str(number), float(number), type_=Number)
        for date_str in table_context._date2id:
            date_obj = Date.make_date(date_str)
            self.add_constant(date_str, date_obj, type_=Date)
        
        self.table_graph = table_context.get_table_knowledge_graph()

        # Adding column names as constants.  Each column gets added once for every
        # type in the hierarchy going from its concrete class to the base Column.  String columns
        # get added as StringColumn and Column, and date and number columns get added as DateColumn
        # (or NumberColumn), ComparableColumn, and Column.
        for column_name, column_types in table_context.column2types.items():
            for column_type in column_types:
                typed_column_name = f"{column_type}_column:{column_name}"
                column: Column = None
                if column_type == 'string':
                    column = StringColumn(typed_column_name)
                elif column_type == 'date':
                    column = DateColumn(typed_column_name)
                    self.add_constant(typed_column_name, column, type_=ComparableColumn)
                elif column_type in ['number', 'num2']:
                    column = NumberColumn(typed_column_name)
                    self.add_constant(typed_column_name, column, type_=ComparableColumn)
                self.add_constant(typed_column_name, column, type_=Column)
                self.add_constant(typed_column_name, column)
                column_type_name = str(PredicateType.get_type(type(column)))
    
    
    def get_nonterminal_productions(self) -> Dict[str, List[str]]:
        """
        grammar induction may fail in cases where certain non-terminals are not present
        but we still need these nonterminals during search
        TODO: this function includes some hotfixes
        """
        production_dict = super(WikiTableAbstractLanguage, self).get_nonterminal_productions()
        # TODO: hotfix add str entry if is missing
        if "str" not in production_dict:
            production_dict["str"] = []

        if "Date" not in production_dict:
            production_dict["Date"] = []

        if "Number" not in production_dict:
            production_dict["Number"] = []

        _a = "List[str] -> [<Row,Column:List[str]>, Row, Column]"
        if _a not in production_dict["List[str]"]: 
            production_dict["List[str]"].append(_a)

        _b = "<Row,Column:List[str]> -> select"
        if "<Row,Column:List[str]>" not in production_dict:
            production_dict["<Row,Column:List[str]>"] = [_b] 

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
            if non_terminal in ["Column", "StringColumn", "NumberColumn", 
                "DateColumn", "ComparableColumn", "str", "Date"]:
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
                # same as
                # new_action_dic[non_terminal].remove('List[Row] -> [<Row,Column:List[Row]>, Row, Column]')

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
                            target_value: List[str], 
                            target_canon: List[str]) -> bool:
        """
        Taken from Chen's script
        """
        target_value_strings = tsv_unescape_list(target_value)
        normalized_target_value_strings = [ TableQuestionContext.normalize_string(value) 
                    for value in target_value_strings]
        canon_value_strings = tsv_unescape_list(target_canon)
        target_value_list = to_value_list(normalized_target_value_strings, canon_value_strings)

        try:
            denotation = self.execute(logical_form)
        except ExecutionError:
            logger.warning(f'Failed to execute: {logical_form}')
            return False
        except Exception as ex:
            err_template = "Exception of type {0} occurred. Arguments:\n{1!r}"
            message = err_template.format(type(ex).__name__, ex.args)
            logger.warning(f'{message}')
        
        if isinstance(denotation, list):
            denotation_list = [str(denotation_item) for denotation_item in denotation]
        else: 
            denotation_list = [str(denotation)]
        denotation_value_list = to_value_list(denotation_list)
        return check_denotation(target_value_list, denotation_value_list)

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
        if isinstance(rows, list):
            return [row.values[column.name] for row in rows]
        else:
            # could also be a row, adhoc added
            return rows.values[column.name]


    # remove it for now, same as returns a list of rows;
    # but most of the time, it's not helping
    # @predicate 
    def same_as(self, src_column: StringColumn, filter_value: str, column: Column) -> List[Row]:
        """
        Takes a row and a column and returns a list of rows from the full set of rows that contain
        the same value under the given column as the given row.
        """
        rows = self.filter_in(self.all_rows(), src_column, filter_value)
        if len(rows) == 0:
            raise ExecutionError("same as gets zero rows")
        row = rows[0]
        cell_value = row.values[column.name]
        return_list = []
        for table_row in self.table_data:
            if table_row.values[column.name] == cell_value:
                return_list.append(table_row)
        return return_list

    # @predicate
    # remove it for now, use corenlp date annotations
    def date(self, year: Number, month: Number, day: Number) -> Date:
        """
        Takes three numbers and returns a ``Date`` object whose year, month, and day are the three
        numbers in that order.
        """
        return Date(year, month, day)  # type: ignore

    @predicate
    def first(self, rows: List[Row]) -> Row:
        """
        Takes an expression that evaluates to a list of rows, and returns the first one in that
        list.
        """
        if not rows:
            # logger.warning("Trying to get first row from an empty list")
            raise ExecutionError("first gets no rows")
        return rows[0]

    @predicate
    def last(self, rows: List[Row]) -> Row:
        """
        Takes an expression that evaluates to a list of rows, and returns the last one in that
        list.
        """
        if not rows:
            # logger.warning("Trying to get first row from an empty list")
            raise ExecutionError("last gets no rows")
        elif len(rows) == 1:
            raise ExecutionError("use first instead!")
        return rows[-1]

    @predicate
    def previous(self, row: Row) -> Row:
        """
        Takes an expression that evaluates to a single row, and returns the row that occurs before
        the input row in the original set of rows. If the input row happens to be the top row, we
        will return an empty list.
        """
        if not row:
            raise ExecutionError("previous gets no rows")
        input_row_index = self._get_row_index(row)
        if input_row_index > 0:
            return self.table_data[input_row_index - 1]
        else:
            raise ExecutionError("preivous already the first line")

    @predicate
    def next(self, row: Row) -> Row:
        """
        Takes an expression that evaluates to a single row, and returns the row that occurs after
        the input row in the original set of rows. If the input row happens to be the last row, we
        will return an empty list.
        """
        if not row:
            raise ExecutionError("next gets no rows")
        input_row_index = self._get_row_index(row)
        if input_row_index < len(self.table_data) - 1 and input_row_index != -1:
            return self.table_data[input_row_index + 1]
        else:
            raise ExecutionError("already the last line")

    @predicate
    def count(self, rows: List[Row]) -> Number:
        return len(rows)  # type: ignore

    @predicate
    def mode(self, rows: List[Row], column: Column) -> List[str]:
        """
        Takes a list of rows and a column and returns the most frequent values (one or more) under
        that column in those rows.
        """
        if len(rows) <= 1:
            raise ExecutionError("mode recieves too few row")
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in rows:
            cell_value = row.values[column.name]
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    # These get added to the language (using `add_predicate()`) if we see a date or number column
    # in the table.

    def argmax(self, rows: List[Row], column: ComparableColumn) -> Row:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            raise ExecutionError("argmax recieves no rows")
        elif len(rows) == 1:
            raise ExecutionError("argmax recieves only one row")
        # We just check whether the first cell value is a date or number and assume that the rest
        # are the same kind of values.
        first_cell_value = rows[0].values[column.name]
        if column.name.startswith("date_column"):
            value_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        else:
            assert column.name.startswith("number_column:") or column.name.startswith("num2_column:") 
            value_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)  # type: ignore
        if not value_row_pairs:
            raise ExecutionError("argmax fail to retrieve value")
        # Returns a list containing the row with the max cell value.
        return sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][1]

    def argmin(self, rows: List[Row], column: ComparableColumn) -> Row:
        """
        Takes a list of rows and a column and returns a list containing a single row (dict from
        columns to cells) that has the minimum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of ``select`` and
        ``all_rows``.
        """
        if not rows:
            raise ExecutionError("argmin recieves no rows")
        elif len(rows) == 1:
            raise ExecutionError("argmin recieves only one row")
        # We just check whether the first cell value is a date or number and assume that the rest
        # are the same kind of values.
        first_cell_value = rows[0].values[column.name]
        if column.name.startswith("date_column"):
            value_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        else:
            assert column.name.startswith("number_column:") or column.name.startswith("num2_column:") 
            value_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)  # type: ignore
        if not value_row_pairs:
            raise ExecutionError("argmin fail to retrieve value")
        # Returns a list containing the row with the max cell value.
        return sorted(value_row_pairs, key=lambda x: x[0])[0][1]

    # These six methods take a list of rows, a column, and a numerical value and return all the
    # rows where the value in that column is [comparator] than the given value.  They only get
    # added to the language if we see a number column in the table.

    def filter_number_greater(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]  # type: ignore

    def filter_number_greater_equals(self,
                                     rows: List[Row],
                                     column: NumberColumn,
                                     filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]  # type: ignore

    def filter_number_lesser(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]  # type: ignore

    def filter_number_lesser_equals(self,
                                    rows: List[Row],
                                    column: NumberColumn,
                                    filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]  # type: ignore

    def filter_number_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]  # type: ignore

    def filter_number_not_equals(self, rows: List[Row], column: NumberColumn, filter_value: Number) -> List[Row]:
        cell_row_pairs = self._get_number_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]  # type: ignore

    # These six methods are the same as the six above, but for dates.  They only get added to the
    # language if we see a date column in the table.

    def filter_date_greater(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value > filter_value]

    def filter_date_greater_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value >= filter_value]

    def filter_date_lesser(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value < filter_value]

    def filter_date_lesser_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value <= filter_value]

    def filter_date_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value == filter_value]

    def filter_date_not_equals(self, rows: List[Row], column: DateColumn, filter_value: Date) -> List[Row]:
        cell_row_pairs = self._get_date_row_pairs_to_filter(rows, column.name)
        return [row for cell_value, row in cell_row_pairs if cell_value != filter_value]

    # These two are similar to the filter methods above, but operate on strings obtained from the
    # question, instead of dates or numbers.  So they check for whether the string value is present
    # in the cell or not, instead of using a numerical / date comparator.  We only add them to the
    # language if we see a string column in the table (which is basically always).

    def filter_in(self, rows: List[Row], column: StringColumn, filter_value: str) -> List[Row]:
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.  Also, we need to remove the
        # "string:" that was prepended to the entity name in the language.
        # filter_value = filter_value.lstrip('string:')
        assert filter_value[:7] == "string:"
        filter_value = filter_value[7:]
        return [row for row in rows if filter_value in row.values[column.name]]

    def filter_not_in(self, rows: List[Row], column: StringColumn, filter_value: str) -> List[Row]:
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.  Also, we need to remove the
        # "string:" that was prepended to the entity name in the language.
        # filter_value = filter_value.lstrip('string:')
        assert filter_value[:7] == "string:"
        filter_value = filter_value[7:]
        return [row for row in rows if filter_value not in row.values[column.name]]

    # These are some more number-column-specific functions, which only get added if we see a number
    # column.

        
    @predicate
    def disjunction(self, row_1: List[Row], row_2: List[Row]) -> List[Row]:
        if len(row_1) == 0 or len(row_2) == 0:
            raise ExecutionError("OR gets empty lists")
        elif row_1 == row_2:
            raise ExecutionError("OR gets the same rows")
        else:
            ret_row = []
            # order perversing!
            for row in self.all_rows():
                if row in row_1 or row in row_2:
                    ret_row.append(row)
            return ret_row
    
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

    def diff(self, first_row: Row, second_row: Row, column: NumberColumn) -> Number:
        """
        Takes a two rows and a number column and returns the difference between the values under
        that column in those two rows.
        """
        if not first_row or not second_row:
            return 0.0  # type: ignore
        elif first_row == second_row:
            raise ExecutionError("diff got the same rows")
        try:
            first_value = float(first_row.values[column.name])
            second_value = float(second_row.values[column.name])
            return first_value - second_value  # type: ignore
        except (ValueError, TypeError) as e:
            raise ExecutionError(f"Invalid column for diff: {column.name}")

    # End of language predicates.  Stuff below here is for private use, helping to implement the
    # functions above.

    def __eq__(self, other):
        if not isinstance(other, WikiTableAbstractLanguage):
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
                if isinstance(_cell_value, float):
                    cell_row_pairs.append((_cell_value, row))

        except ValueError:
            # This means that at least one of the cells is not numerical.
            return []
        return cell_row_pairs

    def _get_date_row_pairs_to_filter(self,
                                      rows: List[Row],
                                      column_name: str) -> List[Tuple[Date, Row]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a date taken from that column, and the corresponding row as
        the second element. The output can be used to compare rows based on the dates.
        """
        if not rows:
            return []

        # cell_row_pairs = [(self._make_date(row.values[column_name]), row) for row in rows]
        cell_row_pairs = []
        for row in rows:
            _cell_value = row.values[column_name]
            if isinstance(_cell_value, Date):
                cell_row_pairs.append((_cell_value, row))

        return cell_row_pairs

    @staticmethod
    def _make_date(cell_string: str) -> Date:
        string_parts = cell_string.split("_")
        year = -1
        month = -1
        day = -1
        for part in string_parts:
            if part.isdigit():
                if len(part) == 4:
                    year = int(part)
                else:
                    day = int(part)
            elif part in MONTH_NUMBERS:
                month = MONTH_NUMBERS[part]
        return Date(year, month, day)

    @staticmethod
    def _value_looks_like_date(cell_value: str) -> bool:
        # We try to figure out if the values being compared are simple numbers or dates. We use
        # simple rules here: that the string contains less than 4 parts, and one of the parts is a
        # month name. Note that this will not consider strings with just years as dates. That's fine
        # because we can compare them as numbers.
        values_are_dates = False
        cell_value_parts = cell_value.split('_')
        # Check if the number of parts in the string are 3 or fewer. If not, it's probably neither a
        # date nor a number.
        if len(cell_value_parts) <= 3:
            for part in cell_value_parts:
                if part in MONTH_NUMBERS:
                    values_are_dates = True
        return values_are_dates

    def _get_row_index(self, row: Row) -> int:
        """
        Takes a row and returns its index in the full list of rows. If the row does not occur in the
        table (which should never happen because this function will only be called with a row that
        is the result of applying one or more functions on the table rows), the method returns -1.
        """
        row_index = -1
        for index, table_row in enumerate(self.table_data):
            if table_row.values == row.values:
                row_index = index
                break
        return row_index

    def check_action_sequence(self, action_seq: List) -> List:
        """
        check if the current action sequence meet the knowledge graph
        implemented in action_walker
        """
        raise NotImplementedError

