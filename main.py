import uuid
from datetime import datetime
from pathlib import Path

from utils import load_config
from rag_dataset.build_rag_dataset import build_dataset_from_config
from train_classifier import train_classifier_from_config

CONFIG_PATH = "configs/config1.json"


if __name__ == "__main__":
    cfg = load_config(CONFIG_PATH)

    # print("=== Step 1: Build RAG dataset (PubMed + Wikipedia) ===")
    # build_dataset_from_config(cfg)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())
    cfg["training"]["output_dir"] = f"{cfg['training']['output_dir']}_{run_id}"
    print(f"Output dir: {cfg['training']['output_dir']}")

    Path(cfg["training"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    train_classifier_from_config(cfg)
    print("\nDone.")