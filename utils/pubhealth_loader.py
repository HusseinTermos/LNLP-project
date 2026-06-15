import json
import logging
import os
from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)

def load_pubhealth_from_config(cfg, splits=None):
    if splits is None:
        splits = ["test", "train", "validation"]

    if isinstance(splits, str):
        splits = [splits]

    base_dir = Path(cfg["data"]["local_dir"])  # should now be "data" or "/data"

    pairs_dir = base_dir / "pubhealth_bigbio_pairs"
    source_dir = base_dir / "pubhealth_source"

    dataframes = {}

    for split in splits:
        pairs_path = pairs_dir / split / "0000.parquet"
        source_path = source_dir / split / "0000.parquet"

        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        pairs_df = pd.read_parquet(pairs_path)
        source_df = pd.read_parquet(source_path)

        pairs_df = pairs_df.rename(
            columns={
                "text_1": "claim",
                "text_2": "evidence",
            }
        )

        if "document_id" not in pairs_df.columns:
            raise ValueError(
                f"'document_id' column not found in pairs file for split={split}. "
                f"Available columns: {list(pairs_df.columns)}"
            )

        if "claim_id" not in source_df.columns:
            raise ValueError(
                f"'claim_id' column not found in source file for split={split}. "
                f"Available columns: {list(source_df.columns)}"
            )

        if "main_text" not in source_df.columns:
            raise ValueError(
                f"'maintext' column not found in source file for split={split}. "
                f"Available columns: {list(source_df.columns)}"
            )

        source_lookup = (
            source_df[["claim_id", "main_text"]]
            .drop_duplicates(subset=["claim_id"])
            .copy()
        )

        merged_df = pairs_df.merge(
            source_lookup,
            left_on="document_id",
            right_on="claim_id",
            how="left",
        )

        missing_maintext = merged_df["main_text"].isna().sum()

        if missing_maintext > 0:
            print(
                f"Warning: {missing_maintext} rows in split={split} "
                f"did not match a maintext entry."
            )

        dataframes[split] = merged_df

    return dataframes



def load_pubhealth(local_dir):
    return {
        "train": pd.read_parquet(os.path.join(local_dir, "train", "0000.parquet")),
        "validation": pd.read_parquet(os.path.join(local_dir, "validation", "0000.parquet")),
        "test": pd.read_parquet(os.path.join(local_dir, "test", "0000.parquet")),
    }

def load_processed_dataset(path):
    rows = []

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    grouped = {
        label: group_df.reset_index(drop=True)
        for label, group_df in df.groupby("split")
    }

    return grouped

