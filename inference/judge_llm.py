import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_LLM_MODEL  = "Qwen/Qwen3-8B"
DEFAULT_BATCH_SIZE = 8

REFORMULATOR_SYSTEM = (
    "You are a scientific evidence summarizer. "
    "Given retrieved evidence chunks about a health claim, "
    "synthesize them into a single coherent paragraph. Be factual and concise."
)

JUDGE_SYSTEM = (
    "You are a fact-checking judge evaluating health claims.\n\n"
    "Use exactly these two judgments:\n"
    "- SUPPORTED: the evidence confirms or is consistent with the verdict.\n"
    "- NOT_SUPPORTED: the evidence contradicts the verdict, is irrelevant, "
    "incomplete, or provides no support at all.\n\n"
    "You MUST choose exactly one. Output only the two required lines — no other text."
)


class QwenLLM:
    def __init__(self, model_name: str, device: torch.device):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.model.to(device).eval()
        self.device = device

    def _apply_chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs   = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=1024)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        return [
            self.tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
            for ids in out_ids
        ]

    def run(self, system: str, user_prompts: list[str],
            batch_size: int, max_new_tokens: int, desc: str) -> list[str]:
        formatted = [self._apply_chat(system, u) for u in user_prompts]
        outputs   = []
        for start in tqdm(range(0, len(formatted), batch_size), desc=desc):
            outputs.extend(self._generate_batch(formatted[start: start + batch_size], max_new_tokens))
        return outputs


def _reformulator_prompt(claim: str, rag_context: str) -> str:
    return (
        f"Claim: {claim}\n\n"
        f"Retrieved chunks:\n{rag_context}\n\n"
        "Provide a single coherent evidence paragraph (2-4 sentences) "
        "summarizing what the evidence says about this claim."
    )


def _judge_prompt(claim: str, predicted_label: str, reformulated_evidence: str) -> str:
    return (
        f"Claim: {claim}\n"
        f"Verdict: {predicted_label}\n"
        f"Evidence: {reformulated_evidence}\n\n"
        "Does the evidence support the verdict?\n"
        "Respond in exactly this format (two lines, nothing else):\n"
        "JUDGMENT: SUPPORTED\n"
        "REASON: <one sentence>\n\n"
        "or\n\n"
        "JUDGMENT: NOT_SUPPORTED\n"
        "REASON: <one sentence>"
    )


def _parse_judge_output(text: str) -> tuple[str, str]:
    verdict, reason = "UNKNOWN", ""
    for line in text.split("\n"):
        line  = line.strip()
        upper = line.upper()
        if upper.startswith("JUDGMENT:"):
            raw = line.split(":", 1)[1].strip().upper()
            verdict = "NOT_SUPPORTED" if "NOT" in raw else "SUPPORTED" if "SUPPORTED" in raw else "UNKNOWN"
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


def run_judge(
    predictions_csv: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    print(f"Loaded {len(df)} rows from {predictions_csv}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_path) if output_path else Path(predictions_csv).with_name(
        Path(predictions_csv).stem + f"_judged_{ts}.csv"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    llm  = QwenLLM(llm_model, device)
    rows = df.to_dict("records")

    ref_prompts = [
        _reformulator_prompt(
            str(r.get("claim", "")).strip(),
            str(r.get("rag_context", "No evidence retrieved.")).strip() or "No evidence retrieved.",
        )
        for r in rows
    ]
    reformulated = llm.run(REFORMULATOR_SYSTEM, ref_prompts, batch_size,
                           max_new_tokens=128, desc="Reformulating evidence")

    judge_prompts = [
        _judge_prompt(str(r.get("claim", "")).strip(), str(r.get("predicted_label", "")), ev)
        for r, ev in zip(rows, reformulated)
    ]
    raw_outputs = llm.run(JUDGE_SYSTEM, judge_prompts, batch_size,
                          max_new_tokens=80, desc="Judging")

    verdicts, reasons = zip(*[_parse_judge_output(o) for o in raw_outputs])
    df["reformulated_evidence"] = reformulated
    df["judge_verdict"]         = list(verdicts)
    df["judge_explanation"]     = list(reasons)

    print("\nJudge verdict distribution:")
    print(df["judge_verdict"].value_counts().to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Qwen judge pipeline for PubHealth predictions.")
    parser.add_argument("--predictions", required=True, help="CSV from run_inference.py")
    parser.add_argument("--llm-model",   default=DEFAULT_LLM_MODEL)
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    run_judge(
        predictions_csv = args.predictions,
        llm_model       = args.llm_model,
        batch_size      = args.batch_size,
        output_path     = args.output,
    )


if __name__ == "__main__":
    main()
