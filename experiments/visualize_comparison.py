#!/usr/bin/env python3
"""
Visualize comparison between no_crop, single_crop, and multi_crop.
Shows attention maps, crop regions, and answers for each method.
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


def evaluate_answer(pred, labels):
    if pred is None:
        return False
    pred = pred.lower().strip()
    for label in labels:
        if label.lower() in pred or pred in label.lower():
            return True
    return False


def get_answer(image, crops, question, model, processor):
    """Get answer using original image + optional crops."""
    if not crops:
        all_images = [image]
        prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
    else:
        all_images = [image] + crops
        prompt = "<image>" * len(all_images) + f"\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"

    inputs = processor(images=all_images, text=prompt, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    answer = response.split('ASSISTANT:')[1].strip() if 'ASSISTANT:' in response else response.strip()

    del inputs, gen_ids
    torch.cuda.empty_cache()
    return answer


def visualize_sample(image, att_map, question, labels, answers, bboxes_dict, save_path):
    """Create visualization for one sample."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original image, Attention map, Attention overlay
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(att_map, cmap='hot')
    axes[0, 1].set_title("Attention Map", fontsize=12)
    axes[0, 1].axis('off')

    # Overlay attention on image
    att_resized = np.array(Image.fromarray((att_map * 255).astype(np.uint8)).resize(image.size))
    axes[0, 2].imshow(image)
    axes[0, 2].imshow(att_resized, cmap='hot', alpha=0.5)
    axes[0, 2].set_title("Attention Overlay", fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: Single crop region, Multi crop regions, Results summary
    colors = ['lime', 'cyan', 'magenta', 'yellow']

    # Single crop
    axes[1, 0].imshow(image)
    if bboxes_dict.get('single'):
        bbox = bboxes_dict['single']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  linewidth=3, edgecolor='lime', facecolor='none')
        axes[1, 0].add_patch(rect)
    correct = "✓" if answers['single']['correct'] else "✗"
    axes[1, 0].set_title(f"Single Crop: '{answers['single']['answer']}' {correct}", fontsize=11)
    axes[1, 0].axis('off')

    # Multi crop
    axes[1, 1].imshow(image)
    if bboxes_dict.get('multi'):
        for i, bbox in enumerate(bboxes_dict['multi']):
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                      linewidth=3, edgecolor=colors[i % len(colors)], facecolor='none')
            axes[1, 1].add_patch(rect)
    correct = "✓" if answers['multi']['correct'] else "✗"
    axes[1, 1].set_title(f"Multi Crop (3): '{answers['multi']['answer']}' {correct}", fontsize=11)
    axes[1, 1].axis('off')

    # Results summary
    axes[1, 2].axis('off')
    summary = f"""Question:
{question[:80]}{'...' if len(question) > 80 else ''}

Ground Truth: {labels[:2]}

Results:
  No Crop:     '{answers['no_crop']['answer'][:30]}' {"✓ CORRECT" if answers['no_crop']['correct'] else "✗ WRONG"}
  Single Crop: '{answers['single']['answer'][:30]}' {"✓ CORRECT" if answers['single']['correct'] else "✗ WRONG"}
  Multi Crop:  '{answers['multi']['answer'][:30]}' {"✓ CORRECT" if answers['multi']['correct'] else "✗ WRONG"}
"""
    axes[1, 2].text(0.1, 0.5, summary, fontsize=11, verticalalignment='center',
                    family='monospace', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(num_samples=20):
    device = "cuda:2"
    print(f"Loading LLaVA model on {device}...")

    model_id = model_to_fullname["llava"]
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    # Load data - mix of regular samples
    with open("./data/textvqa/data.json") as f:
        data = json.load(f)[:num_samples]

    os.makedirs("visualizations", exist_ok=True)
    bbox_size = 336

    results_summary = []

    for i, item in enumerate(tqdm(data, desc="Visualizing")):
        image_path = os.path.join("./data/textvqa/images", item['image_path'])
        image = Image.open(image_path).convert("RGB")
        question = item['question']
        labels = item['labels']

        # Get attention map
        short_prompt = f"<image>\nUSER: {question} Answer the question using a single word or phrase.\nASSISTANT:"
        general_prompt = f"<image>\nUSER: Write a general description of the image.\nASSISTANT:"

        att_map = rel_attention_llava(image, short_prompt, general_prompt, model, processor)

        # Get bboxes
        single_bbox = bbox_from_att_image_adaptive(att_map, image.size, bbox_size)
        multi_bboxes_with_scores = get_multi_crop_bboxes(att_map, image.size, 3, bbox_size)
        multi_bboxes = [bbox for bbox, _ in multi_bboxes_with_scores]

        # Get answers
        answers = {}

        # No crop
        no_crop_answer = get_answer(image, [], question, model, processor)
        answers['no_crop'] = {
            'answer': no_crop_answer,
            'correct': evaluate_answer(no_crop_answer, labels)
        }

        # Single crop
        single_crop = image.crop(single_bbox)
        single_answer = get_answer(image, [single_crop], question, model, processor)
        answers['single'] = {
            'answer': single_answer,
            'correct': evaluate_answer(single_answer, labels)
        }

        # Multi crop
        multi_crops = [image.crop(bbox) for bbox in multi_bboxes]
        multi_answer = get_answer(image, multi_crops, question, model, processor)
        answers['multi'] = {
            'answer': multi_answer,
            'correct': evaluate_answer(multi_answer, labels)
        }

        bboxes_dict = {
            'single': single_bbox,
            'multi': multi_bboxes
        }

        # Visualize
        save_path = f"visualizations/sample_{i:03d}.png"
        visualize_sample(image, att_map, question, labels, answers, bboxes_dict, save_path)

        results_summary.append({
            'idx': i,
            'question': question[:50],
            'no_crop': answers['no_crop']['correct'],
            'single': answers['single']['correct'],
            'multi': answers['multi']['correct']
        })

        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    no_crop_acc = sum(1 for r in results_summary if r['no_crop'])
    single_acc = sum(1 for r in results_summary if r['single'])
    multi_acc = sum(1 for r in results_summary if r['multi'])

    print(f"No Crop:     {no_crop_acc}/{len(results_summary)} ({100*no_crop_acc/len(results_summary):.1f}%)")
    print(f"Single Crop: {single_acc}/{len(results_summary)} ({100*single_acc/len(results_summary):.1f}%)")
    print(f"Multi Crop:  {multi_acc}/{len(results_summary)} ({100*multi_acc/len(results_summary):.1f}%)")

    # Show interesting cases
    print(f"\n{'='*60}")
    print("INTERESTING CASES")
    print(f"{'='*60}")

    print("\nMulti wins, Single loses:")
    for r in results_summary:
        if r['multi'] and not r['single']:
            print(f"  Sample {r['idx']}: {r['question']}...")

    print("\nSingle wins, Multi loses:")
    for r in results_summary:
        if r['single'] and not r['multi']:
            print(f"  Sample {r['idx']}: {r['question']}...")

    print(f"\nVisualizations saved to ./visualizations/")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(num_samples=n)
