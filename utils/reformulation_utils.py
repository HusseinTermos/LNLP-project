
import os
from pathlib import Path

from query_reformulation.query_reformulation import Query_Reformulater


def build_reformulater_from_config(cfg):
    reform_cfg = cfg["reformulation"]

    if not reform_cfg.get("enabled", True):
        return None

    return Query_Reformulater(
        model_name=reform_cfg["model_name"],
        model_load_mode=reform_cfg["model_load_mode"],
        HF_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        batch_size=reform_cfg["batch_size"],
        temperature=reform_cfg["temperature"],
        max_new_tokens=reform_cfg["max_new_tokens"],
        cache_from=str(Path(reform_cfg["cache_path"])),
        cache_to=str(Path(reform_cfg["cache_path"])),
    )
