#!/usr/bin/env python3
"""
Weight Delta Analysis for Quantized Models

Load two models from folders and calculate the weight delta.
Handles both regular weights and quantization scales for .w8a8 models.
"""

import torch
import gc
import os
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoConfig
import numpy as np
from collections import defaultdict


def load_model_weights(model_path: str, postfix: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load model weights and scales from a directory.
    
    Args:
        model_path: Path to the model directory
        postfix: Postfix to add to parameter names
    
    Returns:
        Tuple of (weights_dict, scales_dict)
    """
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to match model
            device_map="cpu",  # Load on CPU first
            low_cpu_mem_usage=True,
        )
        
        weights_dict = {}
        scales_dict = {}
        
        # Track projection weights specifically
        proj_patterns = ['_proj.weight', 'down_proj', 'gate_proj', 'up_proj', 'k_proj', 'o_proj', 'q_proj', 'v_proj']
        
        # Extract weights and scales from parameters
        for name, param in model.named_parameters():
            key_name = f"{name}{postfix}"
            
            # Check if this is a scale parameter
            if name.endswith('_scale'):
                scales_dict[key_name] = param.detach().clone()
                print(f"  Scale parameter: {name} -> dtype: {param.dtype}, shape: {param.shape}")
            # Check if this is a projection weight
            elif any(pattern in name for pattern in proj_patterns):
                weights_dict[key_name] = param.detach().clone()
                print(f"  Proj weight parameter: {name} -> dtype: {param.dtype}, shape: {param.shape}")
            # Other weights (for completeness, but we'll focus on proj weights)
            elif 'weight' in name:
                weights_dict[key_name] = param.detach().clone()
                print(f"  Other weight parameter: {name} -> dtype: {param.dtype}, shape: {param.shape}")
        
        # Also check for named_buffers (scales might be stored as buffers)
        for name, buffer in model.named_buffers():
            key_name = f"{name}{postfix}"
            
            # Check if this is a scale buffer
            if name.endswith('_scale') or 'scale' in name.lower():
                scales_dict[key_name] = buffer.detach().clone()
                print(f"  Scale buffer: {name} -> dtype: {buffer.dtype}, shape: {buffer.shape}")
            # Check if this is a projection weight buffer
            elif any(pattern in name for pattern in proj_patterns):
                weights_dict[key_name] = buffer.detach().clone()
                print(f"  Proj weight buffer: {name} -> dtype: {buffer.dtype}, shape: {buffer.shape}")
            # Other weight buffers
            elif 'weight' in name.lower():
                weights_dict[key_name] = buffer.detach().clone()
                print(f"  Other weight buffer: {name} -> dtype: {buffer.dtype}, shape: {buffer.shape}")
        
        # Filter to focus on projection weights only
        proj_weights = {k: v for k, v in weights_dict.items() if any(pattern in k for pattern in proj_patterns)}
        proj_scales = {k: v for k, v in scales_dict.items() if any(pattern in k for pattern in proj_patterns)}
        
        print(f"Loaded {len(weights_dict)} total weight tensors ({len(proj_weights)} projection weights)")
        print(f"Loaded {len(scales_dict)} total scale tensors ({len(proj_scales)} projection scales)")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return proj_weights, proj_scales  # Return only projection weights/scales
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return {}, {}


def calculate_delta_statistics(delta_dict: Dict[str, torch.Tensor], dict_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for delta tensors.
    
    Args:
        delta_dict: Dictionary of delta tensors
        dict_name: Name for logging purposes
    
    Returns:
        Dictionary with statistics
    """
    if not delta_dict:
        print(f"No deltas found for {dict_name}")
        return {}
    
    print(f"\n=== {dict_name} Delta Statistics ===")
    
    all_deltas = []
    per_tensor_stats = {}
    
    for name, delta_tensor in delta_dict.items():
        # Convert to float for calculations
        delta_flat = delta_tensor.float().flatten()
        all_deltas.extend(delta_flat.tolist())
        
        # Per-tensor statistics
        tensor_mean = delta_flat.mean().item()
        tensor_min = delta_flat.min().item()
        tensor_max = delta_flat.max().item()
        tensor_std = delta_flat.std().item()
        tensor_abs_mean = delta_flat.abs().mean().item()
        
        per_tensor_stats[name] = {
            'mean': tensor_mean,
            'min': tensor_min,
            'max': tensor_max,
            'std': tensor_std,
            'abs_mean': tensor_abs_mean,
            'shape': delta_tensor.shape,
            'dtype': delta_tensor.dtype
        }
        
        print(f"  {name}:")
        print(f"    Shape: {delta_tensor.shape}, Dtype: {delta_tensor.dtype}")
        print(f"    Mean: {tensor_mean:.6f}, Std: {tensor_std:.6f}")
        print(f"    Min: {tensor_min:.6f}, Max: {tensor_max:.6f}")
        print(f"    Abs Mean: {tensor_abs_mean:.6f}")
    
    # Overall statistics
    all_deltas = np.array(all_deltas)
    overall_stats = {
        'mean': float(np.mean(all_deltas)),
        'min': float(np.min(all_deltas)),
        'max': float(np.max(all_deltas)),
        'std': float(np.std(all_deltas)),
        'abs_mean': float(np.mean(np.abs(all_deltas))),
        'num_tensors': len(delta_dict),
        'total_elements': len(all_deltas)
    }
    
    # Find tensors with extreme values
    min_tensor = min(per_tensor_stats.items(), key=lambda x: x[1]['min'])
    max_tensor = max(per_tensor_stats.items(), key=lambda x: x[1]['max'])
    max_abs_mean_tensor = max(per_tensor_stats.items(), key=lambda x: x[1]['abs_mean'])
    
    print(f"\n=== Overall {dict_name} Statistics ===")
    print(f"Total tensors: {overall_stats['num_tensors']}")
    print(f"Total elements: {overall_stats['total_elements']}")
    print(f"Overall mean: {overall_stats['mean']:.6f}")
    print(f"Overall std: {overall_stats['std']:.6f}")
    print(f"Overall min: {overall_stats['min']:.6f} (in {min_tensor[0]})")
    print(f"Overall max: {overall_stats['max']:.6f} (in {max_tensor[0]})")
    print(f"Overall abs mean: {overall_stats['abs_mean']:.6f}")
    print(f"Largest abs mean: {max_abs_mean_tensor[1]['abs_mean']:.6f} (in {max_abs_mean_tensor[0]})")
    
    return {
        'overall': overall_stats,
        'per_tensor': per_tensor_stats,
        'extremes': {
            'min_tensor': min_tensor,
            'max_tensor': max_tensor,
            'max_abs_mean_tensor': max_abs_mean_tensor
        }
    }


def compare_model_dtypes(weights_dict_a: Dict[str, torch.Tensor], weights_dict_b: Dict[str, torch.Tensor],
                        scales_dict_a: Dict[str, torch.Tensor], scales_dict_b: Dict[str, torch.Tensor]):
    """Compare dtypes between two models"""
    print("\n=== Model Dtype Comparison ===")
    
    # Get unique dtypes for each model
    dtypes_a_weights = set(tensor.dtype for tensor in weights_dict_a.values())
    dtypes_b_weights = set(tensor.dtype for tensor in weights_dict_b.values())
    dtypes_a_scales = set(tensor.dtype for tensor in scales_dict_a.values())
    dtypes_b_scales = set(tensor.dtype for tensor in scales_dict_b.values())
    
    print(f"Model A weight dtypes: {dtypes_a_weights}")
    print(f"Model B weight dtypes: {dtypes_b_weights}")
    print(f"Model A scale dtypes: {dtypes_a_scales}")
    print(f"Model B scale dtypes: {dtypes_b_scales}")
    
    # Check for dtype mismatches
    weight_dtype_match = dtypes_a_weights == dtypes_b_weights
    scale_dtype_match = dtypes_a_scales == dtypes_b_scales
    
    print(f"Weight dtypes match: {weight_dtype_match}")
    print(f"Scale dtypes match: {scale_dtype_match}")
    
    if not weight_dtype_match:
        print("WARNING: Weight dtypes differ between models!")
    if not scale_dtype_match:
        print("WARNING: Scale dtypes differ between models!")


def analyze_projection_deltas(weight_deltas: Dict[str, torch.Tensor], scale_deltas: Dict[str, torch.Tensor]):
    """
    Provide detailed analysis of projection weight deltas by layer and type.
    """
    print("\n=== Projection Weight Analysis ===")
    
    # Categorize deltas by layer and projection type
    mlp_weights = {}
    attn_weights = {}
    mlp_scales = {}
    attn_scales = {}
    
    # Process weight deltas
    for name, delta in weight_deltas.items():
        if 'mlp' in name:
            if 'down_proj' in name:
                mlp_weights.setdefault('down_proj', []).append((name, delta))
            elif 'gate_proj' in name:
                mlp_weights.setdefault('gate_proj', []).append((name, delta))
            elif 'up_proj' in name:
                mlp_weights.setdefault('up_proj', []).append((name, delta))
        elif 'self_attn' in name or 'attn' in name:
            if 'k_proj' in name:
                attn_weights.setdefault('k_proj', []).append((name, delta))
            elif 'q_proj' in name:
                attn_weights.setdefault('q_proj', []).append((name, delta))
            elif 'v_proj' in name:
                attn_weights.setdefault('v_proj', []).append((name, delta))
            elif 'o_proj' in name:
                attn_weights.setdefault('o_proj', []).append((name, delta))
    
    # Process scale deltas
    for name, delta in scale_deltas.items():
        if 'mlp' in name:
            if 'down_proj' in name:
                mlp_scales.setdefault('down_proj', []).append((name, delta))
            elif 'gate_proj' in name:
                mlp_scales.setdefault('gate_proj', []).append((name, delta))
            elif 'up_proj' in name:
                mlp_scales.setdefault('up_proj', []).append((name, delta))
        elif 'self_attn' in name or 'attn' in name:
            if 'k_proj' in name:
                attn_scales.setdefault('k_proj', []).append((name, delta))
            elif 'q_proj' in name:
                attn_scales.setdefault('q_proj', []).append((name, delta))
            elif 'v_proj' in name:
                attn_scales.setdefault('v_proj', []).append((name, delta))
            elif 'o_proj' in name:
                attn_scales.setdefault('o_proj', []).append((name, delta))
    
    # Analyze MLP weights
    print("\n--- MLP Weight Deltas ---")
    for proj_type, deltas in mlp_weights.items():
        print(f"\n{proj_type.upper()} ({len(deltas)} layers):")
        all_deltas = torch.cat([delta.flatten() for _, delta in deltas])
        print(f"  Overall mean: {all_deltas.mean().item():.6f}")
        print(f"  Overall std: {all_deltas.std().item():.6f}")
        print(f"  Overall min: {all_deltas.min().item():.6f}")
        print(f"  Overall max: {all_deltas.max().item():.6f}")
        print(f"  Overall abs mean: {all_deltas.abs().mean().item():.6f}")
        
        # Show per-layer stats for first few layers
        for i, (name, delta) in enumerate(deltas[:3]):
            layer_num = name.split('.')[2] if 'layers.' in name else 'unknown'
            print(f"    Layer {layer_num}: mean={delta.mean().item():.6f}, abs_mean={delta.abs().mean().item():.6f}")
    
    # Analyze Attention weights
    print("\n--- Attention Weight Deltas ---")
    for proj_type, deltas in attn_weights.items():
        print(f"\n{proj_type.upper()} ({len(deltas)} layers):")
        all_deltas = torch.cat([delta.flatten() for _, delta in deltas])
        print(f"  Overall mean: {all_deltas.mean().item():.6f}")
        print(f"  Overall std: {all_deltas.std().item():.6f}")
        print(f"  Overall min: {all_deltas.min().item():.6f}")
        print(f"  Overall max: {all_deltas.max().item():.6f}")
        print(f"  Overall abs mean: {all_deltas.abs().mean().item():.6f}")
        
        # Show per-layer stats for first few layers
        for i, (name, delta) in enumerate(deltas[:3]):
            layer_num = name.split('.')[2] if 'layers.' in name else 'unknown'
            print(f"    Layer {layer_num}: mean={delta.mean().item():.6f}, abs_mean={delta.abs().mean().item():.6f}")
    
    # Analyze MLP scales
    if mlp_scales:
        print("\n--- MLP Scale Deltas ---")
        for proj_type, deltas in mlp_scales.items():
            print(f"\n{proj_type.upper()}_scale ({len(deltas)} layers):")
            all_deltas = torch.cat([delta.flatten() for _, delta in deltas])
            print(f"  Overall mean: {all_deltas.mean().item():.6f}")
            print(f"  Overall std: {all_deltas.std().item():.6f}")
            print(f"  Overall min: {all_deltas.min().item():.6f}")
            print(f"  Overall max: {all_deltas.max().item():.6f}")
            print(f"  Overall abs mean: {all_deltas.abs().mean().item():.6f}")
    
    # Analyze Attention scales
    if attn_scales:
        print("\n--- Attention Scale Deltas ---")
        for proj_type, deltas in attn_scales.items():
            print(f"\n{proj_type.upper()}_scale ({len(deltas)} layers):")
            all_deltas = torch.cat([delta.flatten() for _, delta in deltas])
            print(f"  Overall mean: {all_deltas.mean().item():.6f}")
            print(f"  Overall std: {all_deltas.std().item():.6f}")
            print(f"  Overall min: {all_deltas.min().item():.6f}")
            print(f"  Overall max: {all_deltas.max().item():.6f}")
            print(f"  Overall abs mean: {all_deltas.abs().mean().item():.6f}")


def calculate_weight_deltas(model_path_a: str, model_path_b: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate weight deltas between two quantized models.
    
    Args:
        model_path_a: Path to first model (smoothquant)
        model_path_b: Path to second model (linear)
    
    Returns:
        Tuple of (weight_stats, scale_stats)
    """
    print("=== Loading Models ===")
    
    # Load both models
    weights_a, scales_a = load_model_weights(model_path_a, "_smoothquant")
    weights_b, scales_b = load_model_weights(model_path_b, "_linear")
    
    if not weights_a or not weights_b:
        print("ERROR: Failed to load models")
        return {}, {}
    
    # Compare dtypes
    compare_model_dtypes(weights_a, weights_b, scales_a, scales_b)
    
    print("\n=== Calculating Weight Deltas ===")
    
    # Calculate weight deltas (B - A)
    weight_deltas = {}
    scale_deltas = {}
    
    # Find common weight parameters
    common_weight_keys = set()
    for key_a in weights_a.keys():
        key_base = key_a.replace("_smoothquant", "")
        key_b = key_base + "_linear"
        if key_b in weights_b:
            common_weight_keys.add(key_base)
    
    print(f"Found {len(common_weight_keys)} common weight parameters")
    
    for key_base in common_weight_keys:
        key_a = key_base + "_smoothquant"
        key_b = key_base + "_linear"
        
        tensor_a = weights_a[key_a]
        tensor_b = weights_b[key_b]
        
        # Ensure same dtype for subtraction
        if tensor_a.dtype != tensor_b.dtype:
            print(f"  Converting dtypes for {key_base}: {tensor_a.dtype} vs {tensor_b.dtype}")
            tensor_a = tensor_a.float()
            tensor_b = tensor_b.float()
        
        # Ensure same shape
        if tensor_a.shape != tensor_b.shape:
            print(f"  WARNING: Shape mismatch for {key_base}: {tensor_a.shape} vs {tensor_b.shape}")
            continue
        
        # Calculate delta (B - A)
        delta = tensor_b - tensor_a
        weight_deltas[key_base] = delta
    
    # Find common scale parameters
    common_scale_keys = set()
    for key_a in scales_a.keys():
        key_base = key_a.replace("_smoothquant", "")
        key_b = key_base + "_linear"
        if key_b in scales_b:
            common_scale_keys.add(key_base)
    
    print(f"Found {len(common_scale_keys)} common scale parameters")
    
    for key_base in common_scale_keys:
        key_a = key_base + "_smoothquant"
        key_b = key_base + "_linear"
        
        tensor_a = scales_a[key_a]
        tensor_b = scales_b[key_b]
        
        # Ensure same dtype for subtraction
        if tensor_a.dtype != tensor_b.dtype:
            print(f"  Converting dtypes for scale {key_base}: {tensor_a.dtype} vs {tensor_b.dtype}")
            tensor_a = tensor_a.float()
            tensor_b = tensor_b.float()
        
        # Ensure same shape
        if tensor_a.shape != tensor_b.shape:
            print(f"  WARNING: Shape mismatch for scale {key_base}: {tensor_a.shape} vs {tensor_b.shape}")
            continue
        
        # Calculate delta (B - A)
        delta = tensor_b - tensor_a
        scale_deltas[key_base] = delta
    
    # Calculate statistics
    weight_stats = calculate_delta_statistics(weight_deltas, "Weight")
    scale_stats = calculate_delta_statistics(scale_deltas, "Scale")
    
    # Add detailed projection analysis
    analyze_projection_deltas(weight_deltas, scale_deltas)
    
    return weight_stats, scale_stats


def main():
    """Main function to run weight delta analysis"""
    # Model paths
    model_path_a = "/root/lm-evaluation-harness/Qwen3-14B-v0.2-deepresearch-no-think-100-step-quantized.w8a8"
    model_path_b = "/root/lm-evaluation-harness/Qwen3-14B-v0.2-deepresearch-no-think-100-step-linear-quantized.w8a8"
    
    print("=== Weight Delta Analysis ===")
    print(f"Model A (smoothquant): {model_path_a}")
    print(f"Model B (linear): {model_path_b}")
    
    # Check if paths exist
    if not os.path.exists(model_path_a):
        print(f"ERROR: Model A path does not exist: {model_path_a}")
        return
    
    if not os.path.exists(model_path_b):
        print(f"ERROR: Model B path does not exist: {model_path_b}")
        return
    
    try:
        # Calculate deltas
        weight_stats, scale_stats = calculate_weight_deltas(model_path_a, model_path_b)
        
        print("\n=== Analysis Complete ===")
        print("Weight and scale delta analysis has been completed.")
        print("Check the detailed statistics printed above.")
        
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

