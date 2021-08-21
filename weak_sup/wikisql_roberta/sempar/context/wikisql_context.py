import re
import csv
from typing import Union, Dict, List, Tuple, Set
from collections import defaultdict

from unidecode import unidecode
from allennlp.data.tokenizers import Token
CellValueType = Union[str, float]

STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}

class WikiSQLContext:
    """
    Represent wikisql tabl
    """
    def __init__(self,
                 table_data: List[Dict[str, CellValueType]],
                 column2types: Dict[str, str],
                 column_index_to_name: Dict[str, str],
                 question_tokens: List[Token]) -> None:
        self.table_data = table_data
        self.column2types = column2types
        self.column_index_to_name = column_index_to_name
        self.question_tokens = question_tokens

        self.column_types: Set[str] = set()
        for types in column2types.values():
            self.column_types.add(types)

    def __eq__(self, other):
        if not isinstance(other, WikiSQLContext):
            return False
        return self.table_data == other.table_data
    
    def take_features(self, example: Dict):
        """
        1. extract entities
        2. indicator feature of entities and column names
        """
        # entities
        self._entity2id = {}
        self._num2id = {}

        # add entity and add to kg
        covered_set = set()
        self._knowledge_graph = defaultdict(set)
        for entity in example["entities"]:
            s, e = entity["token_start"], entity["token_end"]
            assert len(entity["value"]) == 1
            ent_l = entity["value"]
            if entity["type"] == "string_list":
                normalized_ent_str = self.normalize_string(ent_l[0])
                if normalized_ent_str == "": continue
                cols = self.string_in_table(normalized_ent_str)
                if len(cols) == 0:
                    # renormalize the tokens, discrepency occurs
                    normalized_ent_str = self.normalize_string(" ".join(example["tokens"][s: e]))
                    cols = self.string_in_table(normalized_ent_str)
                    if len(cols) == 0:
                        # import pdb; pdb.set_trace()
                        # print("ENTITY WARNING")
                        continue
                    
                entity = f"string:{normalized_ent_str}"
                for col in cols:
                    self._knowledge_graph[entity].add(col)
                    self._knowledge_graph[col].add(entity)
                        
                self._entity2id[entity] = (s, e)
                covered_set = covered_set.union(range(s, e))
            else:
                assert entity["type"] == "num_list"
                assert s == e - 1

                num = ent_l[0]
                for col, col_type in self.column2types.items():
                    if col_type == "number":
                        self._knowledge_graph[col].add(num)
                        self._knowledge_graph[num].add(col)

                self._num2id[num] = s 
                covered_set = covered_set.union(range(s,e))
        
        # in table feature
        in_table_feature = example["in_table"]
        self.question_in_table_feat = in_table_feature

        # column feature
        column_feature = example["prop_features"]
        self.column_feat = dict()
        for column_name in column_feature:
            _num = column_feature[column_name]

            real_col = column_name[2:]
            splits = real_col.split("-") # remove type
            real_col, col_type = "-".join(splits[:-1]), splits[-1]
            real_col_str = self.normalize_string(real_col)
            typed_column_name = f"{col_type}_column:{real_col_str}"

            if _num[0] > 0:
                self.column_feat[typed_column_name] = 1
            else:
                self.column_feat[typed_column_name] = 0

        # double check 
        for i in set(range(len(example["tokens"]))) - covered_set:
            _t = example["tokens"][i]
            normalized_t_str = self.normalize_string(_t)
            if normalized_t_str in STOP_WORDS: continue
            cols = self.string_in_table(normalized_t_str)
            if len(cols) > 0:
                prefixed_str = f"string:{normalized_t_str}"
                self._entity2id[prefixed_str] = (i, i+1)
                for col in cols:
                    self._knowledge_graph[prefixed_str].add(col)
                    self._knowledge_graph[col].add(prefixed_str) 

                    assert col in self.column_feat
                    self.column_feat[col] = 1
            
                # add to feat
                # assert self.question_in_table_feat[i] == 0
                self.question_in_table_feat[i] = 1

    
    def string_in_table(self, normalized_str: str):
        cols = []
        for row in self.table_data:
            for col, val in row.items():
                if isinstance(val, str) and normalized_str in val:
                    cols.append(col)
        return cols
    

    def get_entity_spans(self):
        """
        Get all spans that have entities
        """
        ret_spans = set()
        for _entity, (_s, _e) in self._entity2id.items():
            ret_spans.add((_s, _e - 1)) # inclusive
        for _num, _s in self._num2id.items():
            ret_spans.add((_s, _s))
        return list(ret_spans)

    @classmethod
    def read_from_json(cls, annotaed_example: Dict, annoated_table: Dict):
        """
        Read from processed table
        """
        # token
        tokens = [Token(t) for t in annotaed_example["tokens"]]

        # table_data
        table_data = []
        num_row = len(annoated_table["row_ents"])
        for i in range(num_row):
            row_ent = f"row_{i}"
            row_dic = annoated_table["kg"][row_ent]
            new_row_dict = {}
            for _col, _val in row_dic.items():
                real_col = _col[2:] # r.
                splits = real_col.split("-") # remove type
                real_col, col_type = "-".join(splits[:-1]), splits[-1]

                real_col_str = cls.normalize_string(real_col)
                typed_column_name = f"{col_type}_column:{real_col_str}"
                assert len(_val) == 1
                if isinstance(_val[0], str):
                    new_row_dict[typed_column_name] = cls.normalize_string(_val[0])
                    # new_row_dict[typed_column_name] = _val[0]
                else:
                    assert isinstance(_val[0], float) or isinstance(_val[0], int)
                    new_row_dict[typed_column_name] = _val[0]
            table_data.append(new_row_dict)
        
        # column_types
        column_types = set()
        column2types = dict()
        column_index_to_name = {}
        for i, column in enumerate(annoated_table["props"]):
            real_col = column[2:] # r.
            splits = real_col.split("-") # remove type
            real_col, col_type = "-".join(splits[:-1]), splits[-1]
            real_col_str = cls.normalize_string(real_col)
            column_index_to_name[i] = real_col_str
            typed_column_name = f"{col_type}_column:{real_col_str}"
            column_types.add(col_type)
            column2types[typed_column_name] = col_type
        
        return cls(table_data, column2types, column_index_to_name, tokens)


    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = unidecode(string.lower())
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return string