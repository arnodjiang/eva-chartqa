#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
#  Minimal VQA runner for Qwen‑VL 2.0 / 2.5  (7B, 14B, 72B …)
#  Keeps library‑exact behaviour: pixel limits, VQA prompt, token cap,
#  auto‑split device‑map, 72‑B layer placement, 2.5 / 2.0 processor.
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
from typing import Any, Dict, List
from functools import lru_cache   # ← add this
from tqdm.auto import tqdm   # add at top of file

import torch
from PIL import Image


def vqa(image_path: str,
        question: str,
        dataset: str | None = None,
        model_path: str = MODEL_ID) -> str:

    from qwen_vl_utils import process_vision_info   # make sure pip install qwen-vl-utils

    # 1. load model / processor
    processor, model = load_qwen(model_path)

    # 2. build the VQA prompt
    user_prompt = f"{question}\nPlease try to answer the question with short words or phrases if possible."

    # 3. build chat messages with *multimodal* content
    img = _resize(Image.open(image_path).convert("RGB"))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": user_prompt},
        ],
    }]

    # 4. convert messages → text with <|image|> placeholder
    text = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 5. collect the vision inputs exactly like the library
    images, videos = process_vision_info([messages])

    # 6. tokenise + feature‑extract
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 7. decide max tokens (100 for ChartQA)
    gen_kwargs = {"max_new_tokens": 100}

    # 8. generate  – run the model
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, **gen_kwargs)

    # 9. slice away the prompt tokens (keep only newly generated ones)
    new_tokens = gen_ids[0][len(inputs.input_ids[0]):]

    # 10. decode → clean answer
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return answer

# ─────────────────────────────────────────────────────────── #
#  Relaxed‑accuracy helper  (safe when gold answer == 0)      #
# ─────────────────────────────────────────────────────────── #
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """
    5 % numeric tolerance.  Exact match for non‑numeric.
    Implementation identical to pix2struct (avoids /0).
    """
    def _to_float(text: str):
        try:
            return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
        except ValueError:
            return None

    prediction, target = str(prediction).strip(), str(target).strip()
    p_float, t_float = _to_float(prediction), _to_float(target)

    # NB: the "and t_float" check is what prevents ZeroDivisionError
    if p_float is not None and t_float:
        rel_change = abs(p_float - t_float) / abs(t_float)
        return rel_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# ------------------------------------------------------------------ #
#  Batch driver                                                      #
# ------------------------------------------------------------------ #
def run_split(entries, img_root, split_name, model_id):
    """Run VQA over one split, show live progress & predictions."""
    results = []
    for ex in tqdm(entries, desc=f"Infer {split_name}", ncols=80):
        img_path = os.path.join(img_root, ex["imgname"])
        pred     = vqa(img_path, ex["query"], model_path=model_id)

        # live print
        print(f"[{split_name}] Q: {ex['query']}  →  {pred}")

        results.append({
            "imgname":    ex["imgname"],
            "query":      ex["query"],
            "prediction": pred,
            "answer":     ex["label"],
            "split":      split_name,
        })
    return results

# ─────────────────────────────────────────────────────────── #
#  Accuracy for a single split                                #
# ─────────────────────────────────────────────────────────── #
def compute_accuracy(recs: List[Dict[str, Any]]) -> float:
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Batch VQA for ChartQA splits")
    ap.add_argument("--img_root",       required=True)
    ap.add_argument("--result_file",            required=True,
                    help="Output predictions JSONL (full path or filename)")
    args = ap.parse_args()

    t0 = time.time()

    # read the two input JSONs
    results = []
    with open(args.result_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            results.append(ex)

    acc = compute_accuracy(results)

    # save evaluation
    eval_json = {
        "test_human":     round(acc_h * 100, 2),
        "test_augmented": round(acc_a * 100, 2),
        "overall":        round(acc_o * 100, 2),
    }
    eval_path = os.path.join(os.path.dirname(args.out), "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_json, f, indent=2)

    # print summary
    print("\n────────  Finished inference  ────────")
    for k, v in eval_json.items():
        print(f"{k:>15}: {v:.2f}%")
    print(f"Predictions saved to : {args.out}")
    print(f"Evaluation  saved to : {eval_path}")
    print(f"Elapsed time         : {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()