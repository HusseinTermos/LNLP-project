import os
import json
from typing import Any

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def limit_df(df, max_rows):
    if max_rows is None:
        return df

    return df.iloc[:max_rows].copy()
