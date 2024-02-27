from tokenizers import Tokenizer
from typing import Dict, List


def get_classification_options(tokenizer: Tokenizer) -> List[int]:
    vocab = tokenizer.get_vocab()
    try:
        column_names = set([x.split('_')[0] for x, _ in vocab.items()])
        column_names = [int(x) for x in column_names if x.isnumeric()]
        end_col = str(max(column_names))
        return get_tokens_by_columns(tokenizer, [end_col,])[end_col]
    except:
        print("ERROR")
        return []


def get_tokens_by_columns(tokenizer: Tokenizer, columns: List[str] = None) -> Dict[str, List[int]]:
    vocab = tokenizer.get_vocab()
    column_names = set([x.split('_')[0] for x, _ in vocab.items()])
    mapping = {}
    for col_name in column_names:
        ids = []
        for k, v in vocab.items():
            x = k.split('_')[0]
            if col_name == x:
                ids.append(v)

        mapping[col_name] = ids
    if columns is None:
        return mapping
    else:
        return {x: y for x, y in mapping.items() if x in columns}
