#!/usr/bin/env python3
"""
Test masked approach vs single_crop on single_crop failure cases.

Hypothesis: Masking irrelevant regions (instead of adding confusing multi-crop)
will help the model focus on the relevant area.
"""

import os
import sys
import json
import torch
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from run import vicrop_qa
from info import model_to_fullname


def evaluate_answer(pred, labels):
    """Check if prediction matches any label."""
    if pred is None:
        return False
    pred = pred.lower().strip()
    for label in labels:
        if label.lower() in pred or pred in label.lower():
            return True
    return False


def main(num_samples=50):
    device = "cuda:1"  # Use GPU 1
    print(f"Loading LLaVA model on {device}...")

    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load single_crop failures (path relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    failures_path = os.path.join(project_root, "data/single_crop_failures.json")
    with open(failures_path) as f:
        failures = json.load(f)[:num_samples]

    print(f"\n{'='*70}")
    print(f"Testing MASKED approach vs SINGLE_CROP on {len(failures)} failure cases")
    print(f"{'='*70}\n")

    results = {
        "single_crop": {"correct": 0, "total": 0},
        "masked_black": {"correct": 0, "total": 0},
        "masked_blur": {"correct": 0, "total": 0},
    }

    fixed_by_masked = []

    for i, item in enumerate(tqdm(failures, desc="Testing")):
        image_path = item['image_path']
        question = item['question']
        labels = item['labels']
        short_question = item.get('short_question', question)

        for mode in ["single_crop", "masked_black", "masked_blur"]:
            try:
                result = vicrop_qa(
                    "llava", "rel_att", image_path, question,
                    model, processor, short_question, crop_mode=mode
                )
                answer = result.get("crop_answer", "")
                correct = evaluate_answer(answer, labels)

                results[mode]["total"] += 1
                if correct:
                    results[mode]["correct"] += 1

                    # Track if masked fixed a single_crop failure
                    if mode.startswith("masked") and not results["single_crop"].get("last_correct", False):
                        fixed_by_masked.append({
                            "question": question[:60],
                            "labels": labels[:2],
                            "single_answer": item.get('crop_answer', 'N/A'),
                            "masked_answer": answer,
                            "mode": mode
                        })

                if mode == "single_crop":
                    results["single_crop"]["last_correct"] = correct

            except Exception as e:
                print(f"\nError with {mode} on sample {i}: {e}")
                results[mode]["total"] += 1

            gc.collect()
            torch.cuda.empty_cache()

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"\n--- Progress {i+1}/{len(failures)} ---")
            for mode in results:
                if mode != "single_crop" or "last_correct" not in results[mode]:
                    acc = results[mode]["correct"] / max(results[mode]["total"], 1) * 100
                    print(f"  {mode}: {results[mode]['correct']}/{results[mode]['total']} ({acc:.1f}%)")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Testing on {len(failures)} SINGLE_CROP FAILURE cases")
    print(f"(These are cases where single_crop originally failed)")
    print()

    for mode in ["single_crop", "masked_black", "masked_blur"]:
        total = results[mode]["total"]
        correct = results[mode]["correct"]
        acc = correct / max(total, 1) * 100
        print(f"{mode:15}: {correct:3}/{total:3} fixed ({acc:.1f}%)")

    # Show cases fixed by masking
    print(f"\n{'='*70}")
    print("CASES FIXED BY MASKING (single_crop failed, masked succeeded)")
    print(f"{'='*70}")

    for case in fixed_by_masked[:5]:
        print(f"\nQ: {case['question']}...")
        print(f"Labels: {case['labels']}")
        print(f"Single-crop (WRONG): '{case['single_answer']}'")
        print(f"Masked ({case['mode']}) (CORRECT): '{case['masked_answer']}'")

    print(f"\nTotal fixed by masking: {len(fixed_by_masked)}")

    # Save results
    with open("masked_approach_results.json", "w") as f:
        json.dump({
            "summary": {mode: {"correct": results[mode]["correct"], "total": results[mode]["total"]}
                       for mode in results if "last_correct" not in results.get(mode, {})},
            "fixed_by_masked": fixed_by_masked
        }, f, indent=2)
    print(f"\nResults saved to masked_approach_results.json")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(num_samples=n)
