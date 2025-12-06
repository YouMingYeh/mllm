"""
Utility functions for attention-based visual cropping and masking.

This module provides:
1. Core functions (original paper): bbox_from_att_image_adaptive, high_res, high_pass_filter
2. Attention-based masking: Mask out low-attention regions to reduce noise

Key Insight: "MLLMs are noise-sensitive, not information-starved"
- Multi-crop (adding more images) HURTS performance - adds noise/confusion
- Masking irrelevant regions HELPS - reduces visual noise
- Single focused crop WORKS - provides relevant detail without noise

Reference: "MLLMs Know Where to Look" (ICLR 2025)
"""

import numpy as np
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image as PILImage, ImageFilter
from scipy.ndimage import median_filter, maximum_filter, gaussian_filter
from scipy.stats import entropy as scipy_entropy
from skimage.measure import block_reduce
from qwen_vl_utils import process_vision_info

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs

def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336):
    """
    Generates an adaptive bounding box for original image from an attention map.
    
    This function finds the region with the highest attention in the attention map
    and creates a bounding box around it. It tries different crop ratios and selects
    the one that produces the sharpest attention difference.
    
    Args:
        att_map: A 2D numpy array representing the attention map (e.g., 24x24 for LLaVA or 16x16 for BLIP)
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    # the ratios corresponds to the bounding box we are going to crop the image
    ratios = [1, 1.2, 1.4, 1.6, 1.8, 2]

    max_att_poses = []
    differences = []
    block_nums = []

    for ratio in ratios:
        # perform a bbox_size*r width and bbox_size*r height crop, where bbox_size is the size of the model's original image input resolution. (336 for LLaVA, 224 for BLIP)

        # the size of each block in the attention map, in the original image
        block_size = image_size[0] / att_map.shape[1], image_size[1] / att_map.shape[0]

        # if I want a bbox_size*r width and bbox_size*r height crop from the original image, the number of blocks I need (x, y)
        block_num = min(int(bbox_size*ratio/block_size[0]), att_map.shape[1]), min(int(bbox_size*ratio/block_size[1]), att_map.shape[0])
        if att_map.shape[1]-block_num[0] < 1 and att_map.shape[0]-block_num[1] < 1:
            if ratio == 1:
                return 0, 0, image_size[0], image_size[1]
            else:
                continue
        block_nums.append((block_num[0], block_num[1]))
        
        # attention aggregation map
        sliding_att = np.zeros((att_map.shape[0]-block_num[1]+1, att_map.shape[1]-block_num[0]+1))
        max_att = -np.inf
        max_att_pos = (0, 0)

        # sliding window to find the block with the highest attention
        for x in range(att_map.shape[1]-block_num[0]+1): 
            for y in range(att_map.shape[0]-block_num[1]+1): 
                att = att_map[y:y+block_num[1], x:x+block_num[0]].sum()
                sliding_att[y, x] = att
                if att > max_att:
                    max_att = att
                    max_att_pos = (x, y)
        
        # we have the position of max attention, we can calculate the difference between the max attention and the average of its adjacent attentions, to see if it is sharp enough, the more difference, the sharper
        # we choose the best ratio r according to their attention difference
        adjcent_atts = []
        if max_att_pos[0] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]-1])
        if max_att_pos[0] < sliding_att.shape[1]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1], max_att_pos[0]+1])
        if max_att_pos[1] > 0:
            adjcent_atts.append(sliding_att[max_att_pos[1]-1, max_att_pos[0]])
        if max_att_pos[1] < sliding_att.shape[0]-1:
            adjcent_atts.append(sliding_att[max_att_pos[1]+1, max_att_pos[0]])
        difference = (max_att - np.mean(adjcent_atts)) / (block_num[0] * block_num[1])
        differences.append(difference)
        max_att_poses.append(max_att_pos)
    max_att_pos = max_att_poses[np.argmax(differences)]
    block_num = block_nums[np.argmax(differences)]
    selected_bbox_size = bbox_size * ratios[np.argmax(differences)]
    
    x_center = int(max_att_pos[0] * block_size[0] + block_size[0] * block_num[0] / 2)
    y_center = int(max_att_pos[1] * block_size[1] + block_size[1] * block_num[1] / 2)
    
    x_center = selected_bbox_size//2 if x_center < selected_bbox_size//2 else x_center
    y_center = selected_bbox_size//2 if y_center < selected_bbox_size//2 else y_center
    x_center = image_size[0] - selected_bbox_size//2 if x_center > image_size[0] - selected_bbox_size//2 else x_center
    y_center = image_size[1] - selected_bbox_size//2 if y_center > image_size[1] - selected_bbox_size//2 else y_center

    x1 = max(0, x_center - selected_bbox_size//2)
    y1 = max(0, y_center - selected_bbox_size//2)
    x2 = min(image_size[0], x_center + selected_bbox_size//2)
    y2 = min(image_size[1], y_center + selected_bbox_size//2)

    return x1, y1, x2, y2

def high_res_split_threshold(image, res_threshold=1024):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att


# =============================================================================
# Attention Analysis: Helper functions for attention map analysis
# =============================================================================

def compute_attention_entropy(att_map):
    """Compute normalized entropy of attention map (0=focused, 1=uniform)."""
    flat = att_map.flatten()
    flat = flat - flat.min()
    total = flat.sum()
    if total <= 0:
        return 1.0
    probs = flat / total
    probs = probs[probs > 0]
    max_entropy = np.log(len(flat))
    return scipy_entropy(probs) / max_entropy if max_entropy > 0 else 0.0


def compute_attention_concentration(att_map, top_k_percent=10):
    """Compute how much attention is concentrated in top-k% of regions."""
    flat = np.sort(att_map.flatten())[::-1]
    k = max(1, int(len(flat) * top_k_percent / 100))
    total = flat.sum()
    return flat[:k].sum() / total if total > 0 else 0.0


def count_attention_peaks(att_map, min_distance=3, threshold_rel=0.3):
    """Count significant local maxima in attention map."""
    threshold = att_map.max() * threshold_rel
    local_max = maximum_filter(att_map, size=min_distance) == att_map
    peaks = local_max & (att_map > threshold)
    return int(peaks.sum())


def get_confidence_from_logits(logits):
    """
    Extract confidence score from model output logits.

    Args:
        logits: Model output logits for the first generated token

    Returns:
        float: Confidence score (probability of top prediction)
    """
    # Get softmax probabilities
    probs = F.softmax(logits, dim=-1)
    # Confidence is the max probability
    confidence = probs.max().item()

    return confidence


# =============================================================================
# Attention-based Masking: Mask out low-attention regions instead of multi-crop
# =============================================================================

def create_attention_mask(att_map, threshold_percentile=50, soft_mask=True, blur_kernel=3):
    """
    Create a mask from attention map to highlight relevant regions.

    Args:
        att_map: 2D attention map (e.g., 24x24 for LLaVA)
        threshold_percentile: Percentile threshold for masking (regions below this are masked)
        soft_mask: If True, create a soft mask with gradual falloff. If False, binary mask.
        blur_kernel: Kernel size for Gaussian blur (soft mask smoothing)

    Returns:
        mask: 2D numpy array (same size as att_map), values in [0, 1]
              1 = keep (high attention), 0 = mask out (low attention)
    """
    # Normalize attention to [0, 1]
    att_norm = att_map - att_map.min()
    if att_norm.max() > 0:
        att_norm = att_norm / att_norm.max()

    if soft_mask:
        # Soft mask: use normalized attention directly with smoothing
        mask = gaussian_filter(att_norm, sigma=blur_kernel / 2)
        # Enhance contrast: stretch values above threshold
        threshold = np.percentile(mask, threshold_percentile)
        mask = np.clip((mask - threshold * 0.5) / (1 - threshold * 0.5), 0, 1)
    else:
        # Binary mask: threshold at percentile
        threshold = np.percentile(att_norm, threshold_percentile)
        mask = (att_norm > threshold).astype(np.float32)

    return mask


def apply_mask_to_image(image, mask, mask_color=(0, 0, 0), mask_alpha=0.8):
    """
    Apply attention mask to image, dimming/blacking out low-attention regions.

    Args:
        image: PIL Image
        mask: 2D numpy array (attention-based mask), values in [0, 1]
        mask_color: RGB tuple for masked regions (default: black)
        mask_alpha: How much to blend mask color (1.0 = full mask, 0.0 = no effect)

    Returns:
        masked_image: PIL Image with low-attention regions masked
    """
    # Resize mask to image size
    mask_resized = np.array(
        PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
            image.size, resample=PILImage.BILINEAR
        )
    ) / 255.0

    # Convert image to numpy
    img_array = np.array(image).astype(np.float32)

    # Create mask color array
    mask_color_array = np.array(mask_color, dtype=np.float32)

    # Apply mask: blend between original and mask_color based on mask value
    # High mask value (1) = keep original, low mask value (0) = show mask_color
    for c in range(3):
        img_array[:, :, c] = (
            mask_resized * img_array[:, :, c] +
            (1 - mask_resized) * mask_alpha * mask_color_array[c] +
            (1 - mask_resized) * (1 - mask_alpha) * img_array[:, :, c]
        )

    return PILImage.fromarray(img_array.astype(np.uint8))


def create_masked_image(image, att_map, threshold_percentile=40, soft_mask=True,
                        mask_style="black"):
    """
    Main function to create a masked version of image based on attention.

    This is used for the "original + masked" approach:
    - Instead of adding multiple crop images (confusing)
    - Add ONE masked image where irrelevant regions are dimmed/hidden

    Args:
        image: PIL Image (original)
        att_map: 2D attention map from rel_attention or similar
        threshold_percentile: Regions below this percentile get masked
        soft_mask: Use soft (gradual) or hard (binary) masking
        mask_style: "black" (black out), "blur" (blur out), or "dim" (darken)

    Returns:
        masked_image: PIL Image with irrelevant regions masked
    """
    # Create the attention mask
    mask = create_attention_mask(att_map, threshold_percentile, soft_mask)

    if mask_style == "black":
        # Black out low-attention regions
        masked_image = apply_mask_to_image(image, mask, mask_color=(0, 0, 0), mask_alpha=1.0)

    elif mask_style == "blur":
        # Blur low-attention regions (keep them visible but unfocused)
        blurred = image.filter(ImageFilter.GaussianBlur(radius=15))
        blurred_array = np.array(blurred).astype(np.float32)
        original_array = np.array(image).astype(np.float32)

        # Resize mask
        mask_resized = np.array(
            PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
                image.size, resample=PILImage.BILINEAR
            )
        ) / 255.0

        # Blend: high attention = original, low attention = blurred
        result = np.zeros_like(original_array)
        for c in range(3):
            result[:, :, c] = (
                mask_resized * original_array[:, :, c] +
                (1 - mask_resized) * blurred_array[:, :, c]
            )
        masked_image = PILImage.fromarray(result.astype(np.uint8))

    elif mask_style == "dim":
        # Dim/darken low-attention regions
        masked_image = apply_mask_to_image(image, mask, mask_color=(0, 0, 0), mask_alpha=0.7)

    else:
        raise ValueError(f"Unknown mask_style: {mask_style}")

    return masked_image


def should_use_masking(att_map, entropy_threshold=0.75):
    """
    Decide whether masking would be beneficial based on attention distribution.

    Returns True if attention is focused enough that masking would help.
    Returns False if attention is too diffuse (masking wouldn't help).

    Args:
        att_map: 2D attention map
        entropy_threshold: If entropy > this, attention too diffuse for masking

    Returns:
        tuple: (should_mask: bool, analysis: dict)
    """
    entropy = compute_attention_entropy(att_map)
    concentration = compute_attention_concentration(att_map, top_k_percent=20)
    num_peaks = count_attention_peaks(att_map)

    # Decision: mask if attention is reasonably focused
    should_mask = entropy < entropy_threshold and concentration > 0.25

    analysis = {
        "entropy": float(entropy),
        "concentration": float(concentration),
        "num_peaks": int(num_peaks),
        "should_mask": should_mask,
        "reason": "focused_attention" if should_mask else "diffuse_attention"
    }

    return should_mask, analysis