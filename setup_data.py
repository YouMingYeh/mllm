#!/usr/bin/env python3
"""
Dataset preparation script for MLLMs Know Where to Look.
Run this after cloning the repo to download and prepare datasets.

Usage:
    python setup_data.py --dataset textvqa
    python setup_data.py --dataset aokvqa
    python setup_data.py --dataset all
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def setup_textvqa():
    """Download and prepare TextVQA dataset."""
    print("\n" + "="*50)
    print("Setting up TextVQA dataset...")
    print("="*50)

    data_dir = Path("data/textvqa")
    images_dir = data_dir / "images"

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    if (data_dir / "data.json").exists():
        print("TextVQA already prepared, skipping...")
        return

    # Download images
    zip_file = data_dir / "train_val_images.zip"
    if not any(images_dir.iterdir()) if images_dir.exists() else True:
        print("Downloading TextVQA images (~7GB, this may take a while)...")
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
            "-P", str(data_dir)
        ], check=True)

        print("Extracting images...")
        subprocess.run(["unzip", "-q", str(zip_file), "-d", str(images_dir)], check=True)

        # Move files from train_images subdirectory
        train_images = images_dir / "train_images"
        if train_images.exists():
            for f in train_images.iterdir():
                f.rename(images_dir / f.name)
            train_images.rmdir()

        # Clean up
        zip_file.unlink()

    # Download annotations
    ann_file = data_dir / "TextVQA_0.5.1_val.json"
    if not ann_file.exists():
        print("Downloading TextVQA annotations...")
        subprocess.run([
            "wget", "-q", "--show-progress",
            "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
            "-P", str(data_dir)
        ], check=True)

    # Convert to unified format
    print("Converting to unified format...")
    with open(ann_file) as f:
        datas = json.load(f)

    new_datas = []
    for data_id, data in enumerate(datas['data']):
        new_datas.append({
            'id': str(data_id).zfill(10),
            'question': data['question'],
            'labels': data['answers'],
            'image_path': f"{data['image_id']}.jpg"
        })

    with open(data_dir / "data.json", 'w') as f:
        json.dump(new_datas, f, indent=4)

    print(f"TextVQA: {len(new_datas)} samples prepared")


def setup_aokvqa():
    """Download and prepare AOKVQA dataset from HuggingFace."""
    print("\n" + "="*50)
    print("Setting up AOKVQA dataset...")
    print("="*50)

    data_dir = Path("data/aokvqa")
    images_dir = data_dir / "images"

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    if (data_dir / "data.json").exists():
        print("AOKVQA already prepared, skipping...")
        return

    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Run: pip install datasets tqdm pillow")
        sys.exit(1)

    print("Downloading AOKVQA from HuggingFace (validation split)...")
    ds = load_dataset('HuggingFaceM4/A-OKVQA', split='validation')

    data = []
    for item in tqdm(ds, desc='Processing AOKVQA'):
        img_filename = f'{item["question_id"]}.jpg'
        img_path = images_dir / img_filename

        # Save image
        item['image'].save(img_path)

        # Get answers
        labels = item['direct_answers'] if item['direct_answers'] else []

        data.append({
            'id': item['question_id'],
            'question': item['question'],
            'labels': labels,
            'image_path': img_filename
        })

    with open(data_dir / "data.json", 'w') as f:
        json.dump(data, f, indent=2)

    print(f"AOKVQA: {len(data)} samples prepared")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for MLLMs Know Where to Look")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["textvqa", "aokvqa", "all"],
        default="all",
        help="Which dataset to prepare (default: all)"
    )
    args = parser.parse_args()

    # Create results directory
    Path("data/results").mkdir(parents=True, exist_ok=True)

    if args.dataset in ["textvqa", "all"]:
        setup_textvqa()

    if args.dataset in ["aokvqa", "all"]:
        setup_aokvqa()

    print("\n" + "="*50)
    print("Dataset setup complete!")
    print("="*50)
    print("\nPrepared datasets:")
    for p in Path("data").glob("*/data.json"):
        with open(p) as f:
            count = len(json.load(f))
        print(f"  - {p.parent.name}: {count} samples")


if __name__ == "__main__":
    main()
