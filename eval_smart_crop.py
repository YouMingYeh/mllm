import json
import os
import numpy as np
from get_score import evaluate_textvqa, get_acc

def analyze_smart_crop_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {file_path}")
    
    # Overall accuracy
    raw_acc, crop_acc = evaluate_textvqa(data)
    print(f"\nOverall Results:")
    print(f"Original Accuracy: {raw_acc:.2f}%")
    print(f"Smart Crop Accuracy: {crop_acc:.2f}%")
    print(f"Improvement: {crop_acc - raw_acc:.2f}%")
    
    # Analysis by number of crops
    by_num_crops = {}
    
    for item in data:
        # Determine number of crops used
        if 'bboxes' in item:
            num_crops = len(item['bboxes'])
        else:
            num_crops = 0
            
        if num_crops not in by_num_crops:
            by_num_crops[num_crops] = []
        by_num_crops[num_crops].append(item)
        
    print(f"\nBreakdown by Number of Crops:")
    print(f"{'# Crops':<10} {'Count':<10} {'Orig Acc':<10} {'Crop Acc':<10} {'Diff':<10}")
    print("-" * 55)
    
    sorted_counts = sorted(by_num_crops.keys())
    for count in sorted_counts:
        items = by_num_crops[count]
        subset_raw, subset_crop = evaluate_textvqa(items)
        diff = subset_crop - subset_raw
        print(f"{count:<10} {len(items):<10} {subset_raw:<10.2f} {subset_crop:<10.2f} {diff:<10.2f}")

    # Analysis by decision reason
    by_reason = {}
    for item in data:
        reason = item.get('analysis', {}).get('decision_reason', 'unknown')
        if reason not in by_reason:
            by_reason[reason] = []
        by_reason[reason].append(item)
        
    print(f"\nBreakdown by Decision Reason:")
    print(f"{'Reason':<25} {'Count':<10} {'Orig Acc':<10} {'Crop Acc':<10} {'Diff':<10}")
    print("-" * 70)
    
    for reason, items in by_reason.items():
        subset_raw, subset_crop = evaluate_textvqa(items)
        diff = subset_crop - subset_raw
        print(f"{reason:<25} {len(items):<10} {subset_raw:<10.2f} {subset_crop:<10.2f} {diff:<10.2f}")

if __name__ == "__main__":
    file_path = "./playground/data/results/llava-textvqa-grad_att-smart_multi_crop.json"
    if os.path.exists(file_path):
        analyze_smart_crop_results(file_path)
    else:
        print(f"File not found: {file_path}")
