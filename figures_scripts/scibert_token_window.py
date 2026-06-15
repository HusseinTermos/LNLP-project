"""
Generate Section 6 token-window EDA using the actual SciBERT tokenizer.

Run from the project root:
    python generate_section6_scibert_token_window.py

No command-line arguments are used. Edit the config block below if paths differ.

Outputs are saved under:
    EDA/section6_token_window/
"""

print("ok")

import json
print("ok")
import math
print("ok")
import traceback
print("ok")
from datetime import datetime
print("ok")
from pathlib import Path
print("ok")
from typing import Iterable

print("ok")
import pandas as pd
print("ok")

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required: pip install matplotlib") from exc
print("ok")

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("transformers is required: pip install transformers") from exc
print("ok")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x
print("ok")

# =============================================================================
# CONFIG: edit these if your project paths differ
# =============================================================================
DATA_DIR = Path("data")
SOURCE_DATASET_DIR = DATA_DIR / "pubhealth_source"
PROCESSED_RAG_JSONL = DATA_DIR / "processed" / "FULL_no_reform.jsonl"
SCIBERT_MODEL_DIR = Path("models") / "scibert_20260605_152114_fa6345e0-e3af-4933-bc23-0dbce6a63b12"

OUTPUT_DIR = Path("EDA") / "section6_token_window"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

SPLITS = ["train", "validation", "test"]
MAX_LENGTH = 512
RANDOM_STATE = 42

# Set to None for full data. Keep None unless the tokenizer run is too slow.
TOKEN_SAMPLE_SIZE = None

# Same formatting style used in the previous EDA script for RAG classifier input.
def build_rag_input_for_model(row: dict) -> str:
    claim = str(row.get("claim", "")).strip()
    query = str(row.get("rag_query", row.get("reformulated_query", claim))).strip()
    context = str(row.get("rag_context", "")).strip() or "No evidence retrieved."
    return (
        f"Original claim:\n{claim}\n\n"
        f"Search query:\n{query}\n\n"
        f"Retrieved evidence:\n{context}"
    )


def ensure_dirs() -> None:
    for d in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {exc}") from exc
    return pd.DataFrame(rows)


def load_source_dataset() -> pd.DataFrame:
    parts = []
    for split in SPLITS:
        p = SOURCE_DATASET_DIR / split / "0000.parquet"
        if not p.exists():
            print(f"WARNING: missing {p}")
            continue
        df = pd.read_parquet(p)
        df = df.copy()
        df["split"] = split
        parts.append(df)
        print(f"Loaded {p}: {len(df):,} rows, {len(df.columns)} columns")
    if not parts:
        raise FileNotFoundError(f"No parquet files found under {SOURCE_DATASET_DIR}")
    return pd.concat(parts, ignore_index=True)


def safe_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def count_tokens(texts: Iterable[str], tokenizer) -> list[int]:
    counts = []
    for t in tqdm(list(texts), desc="SciBERT tokenizing", leave=False):
        ids = tokenizer(str(t), add_special_tokens=True, truncation=False)["input_ids"]
        counts.append(len(ids))
    return counts


def summarize_counts(name: str, split: str, counts: list[int], source: str) -> dict:
    s = pd.Series(counts, dtype="float64")
    return {
        "input_format": name,
        "split": split,
        "source": source,
        "rows_analyzed": int(s.count()),
        "mean_tokens": round(float(s.mean()), 2),
        "median_tokens": round(float(s.median()), 2),
        "p75_tokens": round(float(s.quantile(0.75)), 2),
        "p90_tokens": round(float(s.quantile(0.90)), 2),
        "p95_tokens": round(float(s.quantile(0.95)), 2),
        "max_tokens": int(s.max()) if len(s) else 0,
        "pct_over_512": round(float((s > MAX_LENGTH).mean() * 100), 2),
        "pct_fits_512": round(float((s <= MAX_LENGTH).mean() * 100), 2),
        "tokenizer": "SciBERT tokenizer from local model",
    }


def save_csv_and_md(df: pd.DataFrame, stem: str) -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


def clean_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s).strip("_").lower()


def plot_hist(counts: list[int], name: str, split: str) -> None:
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(counts, bins=60)
    ax.axvline(MAX_LENGTH, linestyle="--", linewidth=1.5, label=f"{MAX_LENGTH}-token limit")
    ax.set_title(f"SciBERT token length: {name} ({split})")
    ax.set_xlabel("SciBERT tokens")
    ax.set_ylabel("Number of examples")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / f"hist_tokens_{clean_filename(name)}_{clean_filename(split)}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def plot_percent_over_512(summary_overall: pd.DataFrame) -> None:
    df = summary_overall.copy()
    df = df.sort_values("pct_over_512", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.45 * len(df))))
    ax.barh(df["input_format"], df["pct_over_512"])
    ax.set_title("Percent of examples exceeding SciBERT 512-token window")
    ax.set_xlabel("% over 512 tokens")
    ax.set_ylabel("Input format")
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIG_DIR / "percent_over_512_by_input_format.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ensure_dirs()
    log_path = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        print("=" * 88)
        print("Section 6 SciBERT token-window EDA")
        print("=" * 88)
        print(f"Output directory: {OUTPUT_DIR.resolve()}")
        print(f"Model/tokenizer: {SCIBERT_MODEL_DIR}")

        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_DIR)
        source_df = load_source_dataset()
        rag_df = read_jsonl(PROCESSED_RAG_JSONL)
        print(f"Loaded processed RAG JSONL: {len(rag_df):,} rows, {len(rag_df.columns)} columns")

        # Optional sample, usually disabled.
        if TOKEN_SAMPLE_SIZE is not None:
            if len(source_df) > TOKEN_SAMPLE_SIZE:
                source_df = source_df.sample(TOKEN_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
            if len(rag_df) > TOKEN_SAMPLE_SIZE:
                rag_df = rag_df.sample(TOKEN_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

        # Build variants. These are deliberately separated by source because source_df has
        # main_text/explanation, while rag_df has the exact RAG context used for training.
        source_variants = {
            "claim_only": safe_text_series(source_df, "claim"),
            "main_text_only": safe_text_series(source_df, "main_text"),
            "explanation_only": safe_text_series(source_df, "explanation"),
            "claim_plus_main_text": safe_text_series(source_df, "claim") + "\n\n" + safe_text_series(source_df, "main_text"),
            "claim_plus_explanation": safe_text_series(source_df, "claim") + "\n\n" + safe_text_series(source_df, "explanation"),
        }
        rag_variants = {
            "rag_context_only": safe_text_series(rag_df, "rag_context"),
            "classifier_input_claim_plus_rag": pd.Series([build_rag_input_for_model(r) for r in rag_df.to_dict("records")]),
        }

        rows = []
        long_rows = []

        def process_variant(name: str, df: pd.DataFrame, texts: pd.Series, source_name: str) -> None:
            for split_name, group_idx in [("all", df.index)]:
                counts = count_tokens(texts.loc[group_idx].tolist(), tokenizer)
                rows.append(summarize_counts(name, split_name, counts, source_name))
                plot_hist(counts, name, split_name)
                for i, c in zip(group_idx, counts):
                    long_rows.append({
                        "input_format": name,
                        "split": "all",
                        "source": source_name,
                        "row_index": int(i),
                        "token_count": int(c),
                        "over_512": bool(c > MAX_LENGTH),
                    })
            if "split" in df.columns:
                for split_name in SPLITS:
                    mask = df["split"].astype(str).eq(split_name)
                    if not mask.any():
                        continue
                    idx = df.index[mask]
                    counts = count_tokens(texts.loc[idx].tolist(), tokenizer)
                    rows.append(summarize_counts(name, split_name, counts, source_name))
                    # Only plot split-level histograms for the most important formats to avoid clutter.
                    if name in {"claim_only", "classifier_input_claim_plus_rag"}:
                        plot_hist(counts, name, split_name)

        for name, texts in source_variants.items():
            process_variant(name, source_df, texts, "pubhealth_source")
        for name, texts in rag_variants.items():
            process_variant(name, rag_df, texts, "processed_rag_jsonl")

        summary = pd.DataFrame(rows)
        order = [
            "claim_only",
            "explanation_only",
            "claim_plus_explanation",
            "main_text_only",
            "claim_plus_main_text",
            "rag_context_only",
            "classifier_input_claim_plus_rag",
        ]
        summary["_order"] = summary["input_format"].map({v: i for i, v in enumerate(order)}).fillna(999)
        summary = summary.sort_values(["split", "_order", "input_format"]).drop(columns=["_order"])
        save_csv_and_md(summary, "scibert_token_window_summary_by_split")

        overall = summary[summary["split"].eq("all")].copy()
        overall = overall.sort_values("input_format", key=lambda x: x.map({v: i for i, v in enumerate(order)}).fillna(999))
        save_csv_and_md(overall, "scibert_token_window_summary_overall")
        plot_percent_over_512(overall)

        # A compact version meant to paste into the report.
        report_cols = ["input_format", "rows_analyzed", "mean_tokens", "median_tokens", "p95_tokens", "max_tokens", "pct_over_512"]
        report_table = overall[report_cols].copy()
        report_table["input_format"] = report_table["input_format"].replace({
            "claim_only": "Claim only",
            "explanation_only": "Explanation only",
            "claim_plus_explanation": "Claim + explanation",
            "main_text_only": "Main text only",
            "claim_plus_main_text": "Claim + main text",
            "rag_context_only": "RAG context only",
            "classifier_input_claim_plus_rag": "Classifier input: claim + RAG",
        })
        save_csv_and_md(report_table, "section6_report_ready_token_table")

        # Keep long per-example counts for later analysis but not too large.
        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(TABLE_DIR / "per_example_token_counts_long.csv", index=False)
        print(f"Saved {TABLE_DIR / 'per_example_token_counts_long.csv'}")

        notes = f"""# Section 6 Token-Window Notes

Generated: {datetime.now().isoformat(timespec='seconds')}

This EDA uses the actual SciBERT tokenizer loaded from:

`{SCIBERT_MODEL_DIR}`

The most report-ready table is:

`tables/section6_report_ready_token_table.md`

Use this table for Section 6 instead of word-count tables. Word counts can remain as background EDA, but the 512-token context-window discussion should be based on SciBERT token counts.

Key interpretation to check after running:

- `Claim only` should usually fit inside 512 tokens.
- `Main text only` and `Claim + main text` likely exceed 512 often, which justifies retrieval/chunking.
- `Classifier input: claim + RAG` measures the exact formatted RAG input used in earlier EDA. If it exceeds 512 often, the classifier is seeing a truncated version, so top-ranked evidence and chunk ordering matter.
"""
        (OUTPUT_DIR / "SECTION6_TOKEN_WINDOW_NOTES.md").write_text(notes, encoding="utf-8")
        print(f"Saved {OUTPUT_DIR / 'SECTION6_TOKEN_WINDOW_NOTES.md'}")

        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "max_length": MAX_LENGTH,
            "tokenizer_model_dir": str(SCIBERT_MODEL_DIR),
            "source_rows": int(len(source_df)),
            "rag_rows": int(len(rag_df)),
            "outputs": [
                str(TABLE_DIR / "section6_report_ready_token_table.csv"),
                str(TABLE_DIR / "scibert_token_window_summary_overall.csv"),
                str(TABLE_DIR / "scibert_token_window_summary_by_split.csv"),
                str(FIG_DIR / "percent_over_512_by_input_format.png"),
            ],
        }
        (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Done.")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        log_path.write_text(tb, encoding="utf-8")
        raise


if __name__ == "__main__":
    print("AAA")
    main()
