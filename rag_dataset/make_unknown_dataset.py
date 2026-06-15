import json
from pathlib import Path
from collections import Counter

INPUT_JSONL = Path("data/processed/FULL_no_reform.jsonl")
OUTPUT_JSONL = Path("data/processed/FULL_no_reform_3class_unknown.jsonl")

MERGE_MAP = {
    "mixture": "unknown",
    "unproven": "unknown",
}

def normalize_label(label):
    label = str(label).strip().lower()
    return MERGE_MAP.get(label, label)

def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    old_counts = Counter()
    new_counts = Counter()
    split_counts = Counter()

    n = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue

            row = json.loads(line)

            old_label = str(row.get("label", "")).strip().lower()
            new_label = normalize_label(old_label)

            row["original_label_4class"] = old_label
            row["label"] = new_label

            old_counts[old_label] += 1
            new_counts[new_label] += 1
            split_counts[(row.get("split", "unknown"), new_label)] += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Saved: {OUTPUT_JSONL}")
    print(f"Rows written: {n}")

    print("\nOld label distribution:")
    for k, v in old_counts.most_common():
        print(f"  {k}: {v}")

    print("\nNew label distribution:")
    for k, v in new_counts.most_common():
        print(f"  {k}: {v}")

    print("\nNew label distribution by split:")
    for (split, label), count in sorted(split_counts.items()):
        print(f"  {split:12s} {label:10s}: {count}")

if __name__ == "__main__":
    main()