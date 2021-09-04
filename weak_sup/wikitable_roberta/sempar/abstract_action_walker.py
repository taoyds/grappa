import logging
from typing import List, Dict, Set
from collections import defaultdict

from allennlp.common.util import START_SYMBOL

from allennlp.semparse.worlds.world import World
from allennlp.semparse.type_declarations import type_declaration as types
from allennlp.semparse.action_space_walker import ActionSpaceWalker

class AbstractWalker(ActionSpaceWalker):

    def _walk(self) -> None:
        """
        search in the action space without data selection operations like lookup.
        the operations used reflect the semantics of a question, it is more abstract and the space would be much smaller
        """
        # Buffer of NTs to expand, previous actions
        incomplete_paths = [([str(type_)], [f"{START_SYMBOL} -> {type_}"]) for type_ in
                            self._world.get_valid_starting_types()]

        self._completed_paths = []
        actions = self._world.get_valid_actions()
        actions = self._filter_abstract(actions)
        multi_match_substitutions = self._world.get_multi_match_mapping()

        while incomplete_paths:
            next_paths = []
            for nonterminal_buffer, history in incomplete_paths:
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
                    # assume repeated use of "previous" is invalid
                    if action not in ["e -> mul_rows_select", "r -> one_row_select"] and action in history: continue
                    new_history = history + [action]
                    new_nonterminal_buffer = nonterminal_buffer[:]
                    # Since we expand the last action added to the buffer, the left child should be
                    # added after the right child.
                    for right_side_part in reversed(self._get_right_side_parts(action)):
                        if types.is_nonterminal(right_side_part):
                            new_nonterminal_buffer.append(right_side_part)
                    next_paths.append((new_nonterminal_buffer, new_history))
            incomplete_paths = []
            for nonterminal_buffer, path in next_paths:
                # An empty buffer means that we've completed this path.
                if not nonterminal_buffer:
                    # paths like s -> string: is not acceptable 
                    if len(path) < 3: continue
                    # Indexing completed paths by the nonterminals they contain.
                    next_path_index = len(self._completed_paths)
                    for action in path:
                        for value in self._get_right_side_parts(action):
                            if not types.is_nonterminal(value):
                                self._terminal_path_index[action].add(next_path_index)
                    self._completed_paths.append(path)
                # We're adding to incomplete_paths for the next iteration, only those paths that are
                # shorter than the max_path_length. The remaining paths will be discarded.
                elif len(path) <= self._max_path_length:
                    incomplete_paths.append((nonterminal_buffer, path))
        # self._filter_path()

    @DeprecationWarning 
    def _filter_path(self):
        """
        filter invalid paths
        """
        fine_path = []
        for path in self._completed_paths:
            # directly to terminals
            if len(path) != 2:
                fine_path.append(path)
        self._completed_paths = fine_path
        
    
    def _filter_abstract(self, actions: List[str]) -> List[str]:
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
                if "string:" in rhs or is_float(rhs):
                        pruned.append(rhs)
                elif rhs in ["date", "-1"]:
                    pruned.append(rhs)
                else:
                    new_action_dic[key_type].append(action)

        # add placeholder for each type    
        # new_action_dic["s"].append("s -> string:#PH#")
        # new_action_dic["n"].append("n -> num:#PH#")
        # new_action_dic["d"].append("d -> date:#PH#")


        # print(f"Pruned actions: {' '.join(pruned)}")
        return new_action_dic