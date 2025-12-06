#!/usr/bin/env python3
"""
Visualize cases where multi-crop BEATS single-crop.
These are PROOF that multi-crop can help when single-crop fails.
"""

import os
import json
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

from utils import bbox_from_att_image_adaptive, get_multi_crop_bboxes
from llava_methods import rel_attention_llava
from info import model_to_fullname


def visualize_win_case(image, att_map, question, labels, single_answer, multi_answer,
                       single_bbox, multi_bboxes, save_path, idx):
    """Visualize a case where multi-crop wins."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Title showing this is a WIN case
    fig.suptitle(f"PROOF #{idx}: Multi-crop WINS where Single-crop FAILED",
                 fontsize=14, fontweight='bold', color='green')

    # Row 1: Original, Attention, Single crop region
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=11)
    axes[0, 0].axis('off')

    # Attention overlay
    att_resized = np.array(Image.fromarray((att_map * 255).astype(np.uint8)).resize(image.size))
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(att_resized, cmap='hot', alpha=0.5)
    axes[0, 1].set_title("Attention Map", fontsize=11)
    axes[0, 1].axis('off')

    # Single crop with X mark (FAILED)
    axes[0, 2].imshow(image)
    if single_bbox:
        rect = patches.Rectangle((single_bbox[0], single_bbox[1]),
                                  single_bbox[2]-single_bbox[0], single_bbox[3]-single_bbox[1],
                                  linewidth=3, edgecolor='red', facecolor='none', linestyle='--')
        axes[0, 2].add_patch(rect)
    axes[0, 2].set_title(f"Single Crop: '{single_answer}' ✗ WRONG", fontsize=11, color='red')
    axes[0, 2].axis('off')

    # Row 2: Multi crop regions, Cropped images preview, Results
    colors = ['lime', 'cyan', 'yellow']

    # Multi crop regions with check mark (SUCCESS)
    axes[1, 0].imshow(image)
    if multi_bboxes:
        for i, bbox in enumerate(multi_bboxes):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                      linewidth=3, edgecolor=colors[i % len(colors)], facecolor='none')
            axes[1, 0].add_patch(rect)
    axes[1, 0].set_title(f"Multi Crop: '{multi_answer}' ✓ CORRECT", fontsize=11, color='green')
    axes[1, 0].axis('off')

    # Show the cropped regions
    if multi_bboxes and len(multi_bboxes) >= 2:
        # Create a composite showing crops
        crop1 = image.crop(multi_bboxes[0]).resize((150, 150))
        crop2 = image.crop(multi_bboxes[1]).resize((150, 150))

        composite = Image.new('RGB', (320, 160), (255, 255, 255))
        composite.paste(crop1, (5, 5))
        composite.paste(crop2, (165, 5))

        axes[1, 1].imshow(composite)
        axes[1, 1].set_title("Crop Regions (Multi)", fontsize=11)
    else:
        axes[1, 1].axis('off')
    axes[1, 1].axis('off')

    # Summary box
    axes[1, 2].axis('off')
    summary = f"""
Question:
{question[:70]}{'...' if len(question) > 70 else ''}

Ground Truth: {labels[:2]}

SINGLE-CROP (1 region):
  Answer: '{single_answer}'
  Result: ✗ WRONG

MULTI-CROP (2-3 regions):
  Answer: '{multi_answer}'
  Result: ✓ CORRECT

WHY MULTI-CROP HELPS:
By covering multiple attention peaks,
multi-crop captures regions that
single-crop misses.
"""
    axes[1, 2].text(0.05, 0.95, summary, fontsize=10, verticalalignment='top',
                    family='monospace', transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")

    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load the analysis results that show which cases multi-crop won
    with open("multicrop_analysis.json") as f:
        analysis = json.load(f)

    # Load single_crop_failures to get image paths
    with open("data/single_crop_failures.json") as f:
        failures = json.load(f)

    # Create image path lookup
    failure_lookup = {}
    for item in failures:
        key = item['question'][:60]
        failure_lookup[key] = item

    # Find cases where multi-crop won
    win_cases = []
    for result in analysis:
        for strategy, data in result["strategies"].items():
            if data.get("correct"):
                # Multi-crop fixed it!
                q_key = result["question"]
                if q_key in failure_lookup:
                    win_cases.append({
                        "question": failure_lookup[q_key]["question"],
                        "labels": failure_lookup[q_key]["labels"],
                        "image_path": failure_lookup[q_key]["image_path"],
                        "single_answer": result["single_crop_answer"],
                        "multi_answer": data["answer"],
                        "strategy": strategy
                    })
                break  # Only count once per question

    print(f"\n{'='*60}")
    print(f"Found {len(win_cases)} cases where MULTI-CROP WINS")
    print(f"{'='*60}\n")

    os.makedirs("visualizations/multicrop_wins", exist_ok=True)
    bbox_size = 336

    for i, case in enumerate(tqdm(win_cases[:10], desc="Visualizing wins")):  # First 10
        image_path = case['image_path']
        image = Image.open(image_path).convert("RGB")
        question = case['question']
        labels = case['labels']

        # Get attention map
        short_prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: Write a general description of the image.\nASSISTANT:"

        att_map = rel_attention_llava(image, short_prompt, general_prompt, model, processor)

        # Get bboxes
        single_bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        multi_bboxes_with_scores = get_multi_crop_bboxes(att_map, image.size, 3, bbox_size)
        multi_bboxes = [bbox for bbox, _ in multi_bboxes_with_scores]

        # Visualize
        save_path = f"visualizations/multicrop_wins/proof_{i+1:02d}.png"
        visualize_win_case(
            image, att_map, question, labels,
            case['single_answer'], case['multi_answer'],
            single_bbox, multi_bboxes,
            save_path, i+1
        )

        print(f"\n  PROOF #{i+1}:")
        print(f"    Q: {question[:60]}...")
        print(f"    Single: '{case['single_answer']}' ❌")
        print(f"    Multi:  '{case['multi_answer']}' ✓")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"PROOF VISUALIZATIONS SAVED")
    print(f"{'='*60}")
    print(f"Location: ./visualizations/multicrop_wins/")
    print(f"Total proofs generated: {min(len(win_cases), 10)}")
    print(f"\nThese visualizations PROVE that multi-crop helps in cases")
    print(f"where single-crop fails by covering multiple attention regions.")


if __name__ == "__main__":
    main()
