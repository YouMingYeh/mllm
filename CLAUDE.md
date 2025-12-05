# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the ICLR 2025 paper "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs". The project provides training-free methods to enhance multimodal large language models' (MLLMs) visual perception of small details using attention mechanisms and gradients.

### Key Research Findings

1. **Causal Size Sensitivity**: MLLMs' performance on visual questions is causally sensitive to the size of the visual subject. Accuracy can drop 11-24 percentage points between large and small objects on TextVQA (even for models trained on the dataset).

2. **MLLMs Know Where to Look**: Despite incorrect answers, MLLMs consistently attend to the correct image regions. The attention ratio (attention to ground-truth bbox / average attention to same-sized bboxes) is significantly >1 across most layers, showing this is a **perception limitation, not a localization limitation**.

3. **Visual Cropping as Solution**: Providing cropped images of small details significantly improves accuracy (+7-17pp on TextVQA for different models) without any training. This approach is scalable and complementary to high-resolution training methods.

## Installation & Setup

```bash
# Create and activate conda environment
conda create -n mllms_know python=3.10
conda activate mllms_know

# Install dependencies
pip install -r requirements.txt

# Install modified transformers library (REQUIRED)
cd transformers
pip install -e .
cd ..
```

**Important**: This project requires a modified version of the transformers library located in the `transformers/` directory. Always install it in editable mode.

## Running Experiments

### Quick Start
The `quick_start.ipynb` notebook demonstrates basic usage including loading images, applying methods, and visualizing attention maps.

### Benchmark Evaluation

1. **Dataset preparation**: Download datasets and place them under `data/[dataset_name]/`. Update paths in `info.py` to point to your local data directories.

2. **Run evaluation**:
```bash
# Format: bash run_all.sh [dataset] [model] [method]
bash run_all.sh textvqa llava rel_att
```

Available options:
- **Datasets**: textvqa, docvqa, gqa, aokvqa, pope, vstar, vqav2
- **Models**: llava (LLaVA-1.5), blip (InstructBLIP), qwen2_5 (Qwen-2.5-VL)
- **Methods**: rel_att, grad_att, pure_grad, rel_att_high, grad_att_high, pure_grad_high

3. **Get performance scores**:
```bash
python get_score.py --data_dir ./data/results --save_path ./
```

**Important Note on TextVQA Evaluation**: This codebase evaluates TextVQA **without external OCR tokens** to measure true visual perception (not confounded by external OCR model ability). The original LLaVA-1.5 paper reports higher TextVQA scores by providing externally extracted OCR tokens in the prompt, which this implementation does not do. With OCR tokens, LLaVA-1.5 achieves 59.8 accuracy; with rel-att it reaches 63.95.

### Multi-GPU Parallel Processing

`run_all.sh` automatically distributes work across 8 GPUs by default. Modify the `gpus` array in `run_all.sh` to use different GPUs:
```bash
declare -a gpus=(0 1 2 3)  # Use only 4 GPUs
```

### Adjusting Model Parameters

**For Qwen-2.5-VL**:
- Attention layers: Modify `ATT_LAYER` in `qwen2_5_methods.py:9` (currently 22)
- Model resolution: Modify `max_pixels` in `run.py:210` (currently 256 * 28 * 28)

**For LLaVA**:
- Attention layer: Modify `ATT_LAYER` in `llava_methods.py:14` (currently 14)

**For BLIP**:
- Q-Former layer: Modify `QFORMER_LAYER` in `blip_methods.py:14` (currently 2)
- Language model layer: Modify `LM_LAYER` in `blip_methods.py:15` (currently 15)

## Architecture & Code Structure

### Core Method Implementations

The repository implements three main visual cropping (ViCrop) methods that leverage MLLMs' internal knowledge:

#### 1. Relative Attention-based Visual Cropping (`rel_att`)

**Intuition**: Normalize attention by a general instruction to emphasize semantically relevant regions.

**Method**:
- Computes answer-to-image attention A_si(x,q) by chaining:
  - Answer-to-token attention: A_st from final answer token to all image tokens across LLM layers
  - Token-to-image attention: A_ti from image tokens to ViT patches across connector layers
  - Combined: A_si(x,q) = A_st(x,q) ⊗ A_ti(x)
- Normalizes by general description: A_rel(x,q) = A_si(x,q) / A_si(x,q') where q' = "Write a general description of the image."
- Requires **two forward passes** (one for question, one for general description)
- **Layer selection**: Uses held-out TextVQA samples to select best layer (LLaVA: layer 14, InstructBLIP: layer 15/2)
- Implementation: `rel_attention_llava()`, `rel_attention_blip()`, `rel_attention_qwen2_5()`

**Performance**: Most effective method overall. Robust to layer averaging if no validation data available.

#### 2. Gradient-Weighted Attention-based Visual Cropping (`grad_att`)

**Intuition**: Use gradients to weight attention without requiring a second forward pass.

**Method**:
- Defines decision as v(x,q) = log(max softmax(z(x,q))) where z is LLM output logit
- Computes gradient-weighted attention:
  - Ã_st(x,q) = A_st(x,q) ⊙ σ(∇_A_st v(x,q))
  - Ã_ti(x,q) = A_ti(x) ⊙ σ(∇_A_ti v(x,q))
  - Combined: Ã_si(x,q) = Ã_st(x,q) ⊗ Ã_ti(x,q)
- Uses ReLU (σ) to filter negative gradients (regions that decrease certainty)
- Requires **one forward pass + gradient computation**
- Implementation: `gradient_attention_llava()`, `gradient_attention_blip()`

**Performance**: Comparable to rel-att. More sensitive to layer selection (3.5pp drop with averaging).

#### 3. Input Gradient-based Visual Cropping (`pure_grad`)

**Intuition**: Directly use gradients w.r.t. input pixels, filtered to emphasize edges with visual details.

**Method**:
- Computes G(x,q) = ||∇_x v(x,q)||_2 (L2 norm over color channels)
- Applies edge-emphasis filtering:
  1. 3×3 Gaussian high-pass filter on image
  2. 3×3 median filter to reduce salt-and-pepper noise
  3. Threshold at spatial median to create binary mask
  4. Element-wise multiply G by mask
- Spatially average-pools to N×N patches
- Normalizes by general description: G_final = G(x,q) / G(x,q')
- Most versatile: **doesn't require Transformer architecture**
- Implementation: `pure_gradient_llava()`, `pure_gradient_blip()`

**Performance**: Lower than attention-based methods but still provides significant gains. Good for non-Transformer models.

#### High-Resolution Support (`_high` suffix)

For images >1024px (e.g., V* benchmark):
1. Split image into non-overlapping blocks <1024×1024 with aspect ratio ≈1
2. Compute importance map separately for each block
3. Merge importance maps back together
4. Apply standard bounding box selection on merged map

**Critical for V***: Without this, LLaVA-1.5 drops 14.66pp; with it, gains 19.89pp over baseline.

### Key Files

- **`run.py`**: Main entry point for running experiments. Contains `vicrop_qa()` which orchestrates the full pipeline: attention map generation → bounding box selection → image cropping → VQA inference on both original and cropped images.

- **`llava_methods.py`**: LLaVA-specific attention extraction methods. Key constants: 576 image tokens (24×24 patches), 14×14 patch size, 336px resolution.

- **`blip_methods.py`**: InstructBLIP-specific methods. Uses two-stage attention: Q-Former cross-attention (32 tokens) → Language Model attention (256 image tokens from 16×16 patches).

- **`qwen2_5_methods.py`**: Qwen-2.5-VL implementation. Uses base64-encoded images and dynamic attention shapes.

- **`utils.py`**: Shared utilities including:
  - `bbox_from_att_image_adaptive()`: **Adaptive bounding box selection algorithm**:
    1. Define window sizes as multiples {1.0, 1.2, 1.4, 1.6, 1.8, 2.0} of MLLM input resolution (336px for LLaVA, 224px for BLIP)
    2. For each window size, slide with stride=1 over importance map to find position maximizing sum of importance values
    3. Compute "sharpness" = (max_sum - mean(adjacent_sums)) / window_area for each window
    4. Select window with maximum sharpness (avoids too-small or too-large crops)
    5. Crop smallest square containing selected window (prevents deformation when resizing)
  - `high_res()`: Splits high-resolution images into patches, processes separately, then merges attention maps.
  - `high_pass_filter()`: Applies 3×3 Gaussian blur + 7×7 median filter to highlight edges.

- **`get_score.py`**: Evaluation metrics for different benchmarks. Each dataset has custom evaluation logic (e.g., VQAv2 uses soft accuracy with 0.3× multiplier per matching answer).

- **`info.py`**: Configuration mapping tasks to data paths and models to HuggingFace model IDs.

### MLLM Architecture Understanding

MLLMs process image-question pairs in four stages:

1. **ViT Encoding**: Image divided into N×N patches → ViT outputs N×N tokens
2. **Token Transformation**: ViT outputs → image tokens via:
   - LLaVA: MLP (maintains N×N structure)
   - InstructBLIP/Qwen-VL: Transformer connector (resamples to fixed T tokens via cross-attention)
3. **LLM Input**: Image tokens prepended to [question tokens + starting answer token]
4. **Autoregressive Generation**: LLM generates answer token-by-token

**Key insight for ViCrop**: MLLMs naturally support multiple images by concatenating their token sequences. This allows seamless addition of cropped image tokens without architectural changes.

### Inference Pipeline Flow

1. Load model and processor for chosen MLLM
2. For each question-image pair:
   - Generate attention map using chosen method (e.g., `rel_attention_llava()`)
   - Select adaptive bounding box from attention map (`bbox_from_att_image_adaptive()`)
   - Crop image at selected region, resize to MLLM input resolution
   - Run VQA on [original_image, cropped_image] as **multi-image input** (tokens concatenated)
3. Compare answers: original-only vs. original+cropped

**Multi-image input format**:
- LLaVA: `<image><image>\nUSER: {question} ASSISTANT:`
- InstructBLIP: Two images processed, same text prompt
- Qwen-2.5-VL: Base64-encoded images in message content array

### Modified Transformers Library

The `transformers/` directory contains a modified version of HuggingFace Transformers that exposes internal attention weights and supports gradient computation. This is essential for the attention-based methods. Always install this version rather than the standard PyPI package.

## Dataset Format

All datasets are converted to a unified JSON format in `data/[dataset]/data.json`:
```json
{
  "id": "0000000001",
  "question": "What text is shown?",
  "labels": ["answer1", "answer2", ...],
  "image_path": "image_id.jpg"
}
```

For VSTAR benchmark, include `short_question` field for attention computation.

## Expected Performance

### Accuracy Improvements on Detail-Sensitive Benchmarks (rel-att)

**LLaVA-1.5**:
- TextVQA: +7.37pp (47.80 → 55.17)
- V*: +19.89pp (42.41 → 62.30)
- DocVQA: +3.66pp (15.97 → 19.63)

**InstructBLIP**:
- TextVQA: +11.96pp (33.48 → 45.44)
- V*: +6.81pp (35.60 → 42.41)

**Performance on General Benchmarks**: Methods maintain or slightly improve accuracy on GQA, AOKVQA, VQAv2, and POPE (0-2pp gains), showing no degradation for larger visual concepts.

### Inference Time Overhead

- **rel-att**: ~1.2s on GPU (equivalent to generating 5-7 tokens)
- **grad-att**: ~0.9s on GPU
- **pure-grad**: ~2.4s on GPU
- Overhead is **constant** regardless of answer length (only needs starting answer token)

### Comparison with Training-Based Methods

ViCrop is complementary to high-resolution training:
- LLaVA-NeXT (higher-res trained): 65.17 TextVQA → 68.65 with rel-att (+3.48pp)
- SEAL (multi-agent fine-tuning): Generally outperformed by LLaVA-1.5+rel-att except on V* (SEAL's target benchmark)

## Known Limitations

1. **Relation and Counting Questions**: Methods don't enhance questions requiring multiple regions or global context. ViCrop focuses on single regions, making it ineffective for:
   - Spatial relations ("Is the truck left or right of the car?")
   - Counting ("How many dogs are in the image?")
   - Global reasoning ("What is the overall scene?")

2. **Model-Specific Effectiveness**: LLaVA-1.5 benefits more than InstructBLIP because:
   - InstructBLIP only trains connector, not LLM backbone
   - LLM doesn't adapt to use additional image tokens effectively
   - Both models still show significant improvements

3. **Layer Selection Dependency**:
   - `grad-att` is more sensitive to layer choice (3.5pp drop with averaging)
   - `rel-att` is robust (0.28pp difference between selective and average)
   - Without validation data, use layer averaging or default layers

4. **External Tools**: Internal ViCrop methods (using model's own attention/gradients) outperform external tools (SAM, YOLO, CLIP) by 5-17pp on TextVQA, showing the value of model-internal knowledge.

## Future Directions from Paper

- **Multi-region cropping**: Extend to focus on multiple regions simultaneously for relations/counting
- **Inference optimization**: Lower precision, weight quantization, Matryoshka Query Transformer
- **Method combination**: Ensemble different methods based on prediction uncertainty
- **Video extension**: Apply to video understanding (currently only images)
