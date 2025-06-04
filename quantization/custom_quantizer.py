import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class W8A16Linear(nn.Module):
    def __init__(self, linear: nn.Linear, group_size=None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        
        # Quantize weights once and store int8_weights as buffer, scales as parameters
        self._quantize_and_store_weights(linear.weight)
        
        # Store bias as parameter (keep in original precision)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.clone())
        else:
            self.bias = None
    
    def _quantize_and_store_weights(self, weights):
        """Quantize weights once: int8 weights as buffer, scales as parameters"""
        w_fp32 = weights.clone().to(torch.float32)
        
        # Calculate scales: map [-max, max] to [-127, 127] per output channel
        if self.group_size is None:
            scales = w_fp32.abs().max(dim=-1, keepdim=True).values / 127.0
        else:
            out_features, in_features = w_fp32.shape
    
            # Pad in_features to be divisible by group_size
            if in_features % self.group_size != 0:
                padding = self.group_size - (in_features % self.group_size)
                w_fp32 = torch.nn.functional.pad(w_fp32, (0, padding))
                
            # Reshape to (out_features, in_features // group_size, group_size)
            num_groups = w_fp32.shape[1] // self.group_size
            w_grouped = w_fp32.reshape(out_features, num_groups, self.group_size)
            
            # Calculate scales for each group
            scales = w_grouped.abs().max(dim=-1, keepdim=True).values / 127.0
            
            # Reshape scales back to (out_features, 1)
            scales = scales.reshape(out_features, 1)
        
        scales = torch.clamp(scales, min=1e-8)
        
        # Quantize: W_int8 = round(W_fp32 / scale) clamped to [-127, 127]
        int8_weights = torch.round(w_fp32 / scales).clamp(-127, 127).to(torch.int8)
        
        # Store int8 weights as buffer with standard .weight name
        self.register_buffer("weight", int8_weights)
        
        # Store scales as trainable parameters with shape [out_features, 1] and .weight_scale name
        self.weight_scale = nn.Parameter(scales.to(torch.bfloat16))

    def forward(self, x):
        """Forward pass: dequantize weights using current scale parameters"""
        # Dequantize: W_fp = W_int8 * scale (scales can be updated during training)
        if self.group_size is None:
            dequantized_weights = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        else:
            out_features = self.weight.shape[0]
            padded_in_features = self.weight.shape[1]
            num_groups = padded_in_features // self.group_size
            
            # Reshape weights to groups
            weight_grouped = self.weight.view(out_features, num_groups, self.group_size)
            
            # Apply scales (scales shape: [out_features, num_groups])
            scales_expanded = self.weight_scale.unsqueeze(-1).to(x.dtype)  # [out_features, num_groups, 1]
            dequantized_grouped = weight_grouped.to(x.dtype) * scales_expanded
            
            # Reshape back and remove padding
            dequantized_weights = dequantized_grouped.view(out_features, -1)
            if hasattr(self, 'original_in_features'):
                dequantized_weights = dequantized_weights[:, :self.original_in_features]
        
        
        # Standard linear operation
        return F.linear(x, dequantized_weights, self.bias)
    
    def get_quantization_info(self):
        """Get information about the quantization"""
        info = {
            'weight_shape': self.weight.shape,
            'weight_scale_shape': self.weight_scale.shape,
            'weight_scale_dtype': self.weight_scale.dtype,
            'weight_scale_requires_grad': self.weight_scale.requires_grad,
            'weight_range': (self.weight.min().item(), self.weight.max().item()),
            'scale_range': (self.weight_scale.min().item(), self.weight_scale.max().item()),
            'group_size': self.group_size
        }
        if self.group_size is not None:
            info['num_groups'] = self.weight_scale.shape[1]
            info['original_in_features'] = self.original_in_features.item() if hasattr(self, 'original_in_features') else self.in_features
            
        return info
    
    def get_memory_info(self):
        """Calculate memory usage breakdown"""
        int8_size = self.out_features * self.in_features * 1  # int8 = 1 byte
        scale_size = self.out_features * 2  # bf16 = 2 bytes  
        original_size = self.out_features * self.in_features * 2  # float32 = 4 bytes
        
        total_quantized = int8_size + scale_size
        savings = (original_size - total_quantized) / original_size * 100
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'quantized_mb': total_quantized / (1024 * 1024),
            'savings_percent': savings,
            'int8_weights_mb': int8_size / (1024 * 1024),
            'scales_mb': scale_size / (1024 * 1024),
            'group_size': self.group_size,
            'num_groups': self.weight_scale.shape[1] if self.group_size is not None else None,
        }


def replace_linear_with_quantized_linear(model, ignore_layers=None):
    """Replace all Linear layers with quantized versions, except those in ignore_layers"""
    if ignore_layers is None:
        ignore_layers = []
    
    # Fix: Collect modules to replace first, then replace them
    modules_to_replace = []
    ignored_modules = []
    
    def find_linear_modules(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if full_name not in ignore_layers:
                    modules_to_replace.append((full_name, child))
                else:
                    ignored_modules.append(full_name)
                    print(f"🚫 Ignoring layer: {full_name}")
            else:
                find_linear_modules(child, full_name)
    
    # First pass: find all linear modules
    find_linear_modules(model)
    
    print(f"📊 Found {len(modules_to_replace)} layers to quantize")
    print(f"🚫 Ignoring {len(ignored_modules)} layers: {ignored_modules}")
    
    # Second pass: replace them
    for name, linear_module in modules_to_replace:
        # Navigate to parent module
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        
        # Replace with quantized version
        quantized_linear = W8A16Linear(linear_module, group_size=None)
        print(f"Replaced {name}: {quantized_linear.get_memory_info()}")
        
        setattr(parent, attr_name, quantized_linear)
    
    return model

def count_parameters(model):
    """Count total parameters in model"""
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model.buffers())
    return f"{total:,} parameters"

def test_dynamic_quantization():
    """Test dynamic quantization with a simple example"""
    print("=== Testing Dynamic Quantization ===")
    
    # Create a simple linear layer
    original_linear = nn.Linear(4, 2)
    original_linear.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0], 
                                                [0.5, 1.5, 2.5, 3.5]])
    
    # Create dynamic quantized version
    dynamic_quant = W8A16Linear(original_linear)
    
    # Test input
    test_input = torch.randn(1, 4)
    
    print(f"Original weights:\n{original_linear.weight}")
    print(f"Original output: {original_linear(test_input)}")
    
    # Get current scales
    current_scales = dynamic_quant.get_quantization_info()
    print(f"Dynamic scales: {current_scales}")
    
    # Forward pass with dynamic quantization
    quant_output = dynamic_quant(test_input)
    print(f"Quantized output: {quant_output}")
    
    # Modify weights to show dynamic adaptation
    with torch.no_grad():
        dynamic_quant.weight_scale *= 2.0  # Double the scales
    
    new_scales = dynamic_quant.get_quantization_info()
    print(f"New scales after scale change: {new_scales}")
    print(f"Scale change ratio: {new_scales['scale_range'][1] / current_scales['scale_range'][1]}")

def inspect_quantized_model(model, max_layers=5):
    """Inspect quantized model structure using state_dict like Hugging Face"""
    print(f"\n=== Quantized Model Inspection (State Dict) ===")
    
    state_dict = model.state_dict()
    
    # Group tensors by layer
    layer_groups = {}
    for name, tensor in state_dict.items():
        if '.weight' in name or '.weight_scale' in name:
            # Extract layer name (everything before .weight or .weight_scale)
            if '.weight_scale' in name:
                layer_name = name.replace('.weight_scale', '')
            else:
                layer_name = name.replace('.weight', '')
            
            if layer_name not in layer_groups:
                layer_groups[layer_name] = {}
            
            if name.endswith('.weight'):
                layer_groups[layer_name]['weight'] = tensor
            elif name.endswith('.weight_scale'):
                layer_groups[layer_name]['weight_scale'] = tensor
    
    # Show quantized layers
    quantized_layers = {k: v for k, v in layer_groups.items() 
                       if 'weight' in v and 'weight_scale' in v and v['weight'].dtype == torch.int8}
    
    # Show regular layers  
    regular_layers = {k: v for k, v in layer_groups.items()
                     if 'weight' in v and v['weight'].dtype != torch.int8}
    
    print(f"Found {len(quantized_layers)} quantized layers:")
    
    # Show detailed info for first few quantized layers
    layer_count = 0
    for layer_name, tensors in quantized_layers.items():
        if layer_count >= max_layers:
            break
            
        weight_tensor = tensors['weight']
        scale_tensor = tensors['weight_scale']
        
        # Calculate memory info
        int8_size = weight_tensor.numel() * 1  # int8 = 1 byte
        scale_size = scale_tensor.numel() * 2  # bf16 = 2 bytes
        original_size = weight_tensor.numel() * 2  # assuming bf16 original
        total_quantized = int8_size + scale_size
        savings = (original_size - total_quantized) / original_size * 100
        
        print(f"\n{layer_name}:")
        print(f"  - {layer_name}.weight: {list(weight_tensor.shape)} ({weight_tensor.dtype})")
        print(f"  - {layer_name}.weight_scale: {list(scale_tensor.shape)} ({scale_tensor.dtype})")
        print(f"  - weight_range: ({weight_tensor.min().item()}, {weight_tensor.max().item()})")
        print(f"  - scale_range: ({scale_tensor.min().item():.6f}, {scale_tensor.max().item():.6f})")
        print(f"  - memory: {savings:.1f}% reduction ({total_quantized / 1024 / 1024:.2f}MB)")
        layer_count += 1
    
    if len(quantized_layers) > max_layers:
        print(f"  ... and {len(quantized_layers) - max_layers} more quantized layers")
    
    # Show non-quantized layers
    if regular_layers:
        print(f"\nFound {len(regular_layers)} non-quantized Linear layers:")
        for layer_name, tensors in list(regular_layers.items())[:5]:
            weight_tensor = tensors['weight']
            print(f"  - {layer_name}.weight: {list(weight_tensor.shape)} ({weight_tensor.dtype})")
        if len(regular_layers) > 5:
            print(f"  ... and {len(regular_layers) - 5} more non-quantized layers")
    
    # Summary statistics
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model.buffers())
    quantized_weight_params = sum(tensors['weight'].numel() for tensors in quantized_layers.values())
    scale_params = sum(tensors['weight_scale'].numel() for tensors in quantized_layers.values())
    
    print(f"\nSummary:")
    print(f"Total tensors in state_dict: {len(state_dict):,}")
    print(f"Quantized layers: {len(quantized_layers)}")
    print(f"Non-quantized linear layers: {len(regular_layers)}")
    print(f"Quantized weight elements: {quantized_weight_params:,} (int8)")
    print(f"Scale parameters: {scale_params:,} (bf16)")
    print(f"Total parameters + buffers: {total_params:,}")

def save_quantized_model(model, path="quantized_model.safetensors"):
    """Save quantized model using safetensors"""
    try:
        from safetensors.torch import save_file
        
        # Get state dict (includes both buffers and parameters)
        state_dict = model.state_dict()
        
        print(f"\nSaving model state dict with {len(state_dict)} tensors...")
        
        # Show some example tensor names to verify scales and weights are included
        scale_tensors = [k for k in state_dict.keys() if 'weight_scale' in k]
        int8_weight_tensors = [k for k in state_dict.keys() if k.endswith('.weight') and state_dict[k].dtype == torch.int8]
        
        print(f"Found {len(scale_tensors)} weight_scale parameter tensors")
        print(f"Found {len(int8_weight_tensors)} quantized int8 weight tensors")
        
        if scale_tensors:
            print(f"Example scale tensor: {scale_tensors[0]} -> {state_dict[scale_tensors[0]].shape} {state_dict[scale_tensors[0]].dtype}")
        if int8_weight_tensors:
            print(f"Example quantized weight: {int8_weight_tensors[0]} -> {state_dict[int8_weight_tensors[0]].shape} {state_dict[int8_weight_tensors[0]].dtype}")
        
        save_file(state_dict, path)
        print(f"Model saved to {path}")
        
        return True
    except ImportError:
        print("safetensors not available. Install with: pip install safetensors")
        
        # Alternative: show what would be saved
        state_dict = model.state_dict()
        print(f"\nModel state dict contains {len(state_dict)} tensors:")
        for name, tensor in list(state_dict.items())[:10]:  # Show first 10
            print(f"  {name}: {tensor.shape} ({tensor.dtype})")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more")
        
        return False

def save_and_push_to_hub(model, tokenizer, original_model_name, quantized_repo_name, push_to_hub=True):
    """Save quantized model and optionally push to Hugging Face Hub"""
    import os
    import tempfile
    from pathlib import Path
    
    try:
        from huggingface_hub import HfApi
        from safetensors.torch import save_file
        import json
        
        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"Saving quantized model to temporary directory: {temp_path}")
            
            # Save model weights in safetensors format
            state_dict = model.state_dict()
            
            weights_path = temp_path / "model.safetensors"
            save_file(state_dict, weights_path)
            print(f"✅ Saved model weights: {len(state_dict)} tensors")
            
            # Save tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(temp_path, max_shard_size="5GB")
                print("✅ Saved tokenizer")
            
            # Save config (copy from original model)
            try:
                original_config = model.config
                config_path = temp_path / "config.json"
                
                # Add quantization info to config
                config_dict = original_config.to_dict()
                config_dict["quantization_config"] = {
                    "config_groups": {
                        "group_0": {
                            "input_activations": {
                                "actorder": None,
                                "block_structure": None,
                                "dynamic": True,
                                "group_size": None,
                                "num_bits": 8,
                                "observer": "memoryless",
                                "observer_kwargs": {},
                                "strategy": "token",
                                "symmetric": True,
                                "type": "int"
                            },
                            "output_activations": None,
                            "targets": ["Linear"],
                            "weights": {
                                "actorder": None,
                                "block_structure": None,
                                "dynamic": False,
                                "group_size": None,
                                "num_bits": 8,
                                "observer": "minmax",
                                "observer_kwargs": {},
                                "strategy": "channel",
                                "symmetric": True,
                                "type": "int"
                            }
                        }
                    },
                    "format": "int-quantized",
                    "global_compression_ratio": 1.2389973594794181,
                    "ignore": ["lm_head"],
                    "kv_cache_scheme": None,
                    "quant_method": "compressed-tensors", 
                    "quantization_status": "compressed",
                    "version": "0.6.0"
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                print("✅ Saved config with quantization info")
                
            except Exception as e:
                print(f"⚠️  Could not save config: {e}")
            
            # Create README
            readme_content = f"""---
library_name: transformers
tags:
- quantization
- int8
- custom
base_model: {original_model_name}
---

# {quantized_repo_name}

This is a custom W8A16 quantized version of [{original_model_name}](https://huggingface.co/{original_model_name}).

## Quantization Details

- **Method**: Custom W8A16 (8-bit weights, 16-bit activations)
- **Weight precision**: INT8 
- **Scale precision**: BF16
- **Quantization**: Symmetric per-channel
- **Zero points**: None (symmetric)

## Model Structure

The quantized model contains:
- `.weight`: INT8 quantized weights
- `.weight_scale`: BF16 scale parameters (trainable)
- Standard embedding and normalization layers in original precision

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Note: This requires custom quantization code to load properly
model = AutoModelForCausalLM.from_pretrained("{quantized_repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{quantized_repo_name}")
```
"""
            
            readme_path = temp_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print("✅ Created README.md")
            
            if push_to_hub:
                try:
                    api = HfApi()
                    
                    print(f"\n🚀 Pushing to Hub: {quantized_repo_name}")
                    api.create_repo(repo_id=quantized_repo_name, exist_ok=True)
                    
                    api.upload_folder(
                        folder_path=temp_path,
                        repo_id=quantized_repo_name,
                        commit_message=f"Upload quantized {original_model_name}"
                    )
                    
                    print(f"✅ Successfully pushed to: https://huggingface.co/{quantized_repo_name}")
                    print(f"🔍 View model files: https://huggingface.co/{quantized_repo_name}/tree/main")
                    
                except Exception as e:
                    print(f"❌ Failed to push to hub: {e}")
                    print("💡 Make sure you're logged in: huggingface-cli login")
                    return False
            else:
                print(f"📁 Model saved locally to: {temp_path}")
                # Copy to a permanent location if not pushing
                import shutil
                local_path = f"./{quantized_repo_name.split('/')[-1]}"
                shutil.copytree(temp_path, local_path)
                print(f"📁 Copied to: {local_path}")
            
            return True
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install huggingface-hub safetensors")
        return False

def quick_hub_push(original_model_name="meta-llama/Llama-3.1-8B-Instruct", 
                   quantized_repo_name=None,
                   push_to_hub=True):
    """Quick function to quantize and push a model to hub"""
    
    if quantized_repo_name is None:
        # Generate a repo name based on original
        base_name = original_model_name.split('/')[-1]
        quantized_repo_name = f"your-username/{base_name}-w8a16-quantized"
        print(f"💡 Using repo name: {quantized_repo_name}")
        print("💡 Change 'your-username' to your actual HF username")
    
    print(f"Loading original model: {original_model_name}")
    model = AutoModelForCausalLM.from_pretrained(original_model_name, torch_dtype = torch.bfloat16)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    except:
        tokenizer = None
        print("⚠️  Could not load tokenizer")
    
    print("Quantizing model...")
    model = replace_linear_with_quantized_linear(model)
    
    print(f"Quantized model: {count_parameters(model)}")
    
    return save_and_push_to_hub(
        model=model,
        tokenizer=tokenizer, 
        original_model_name=original_model_name,
        quantized_repo_name=quantized_repo_name,
        push_to_hub=push_to_hub
    )

def verify_quantization(model, expected_ignored_layers):
    """Verify which layers were quantized vs ignored"""
    quantized_layers = []
    linear_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, W8A16Linear):
            quantized_layers.append(name)
        elif isinstance(module, nn.Linear):
            linear_layers.append(name)
    
    print(f"\n=== Quantization Verification ===")
    print(f"✅ Quantized layers: {len(quantized_layers)}")
    for layer in quantized_layers[:5]:  # Show first 5
        print(f"  - {layer}")
    if len(quantized_layers) > 5:
        print(f"  ... and {len(quantized_layers) - 5} more")
    
    print(f"\n🚫 Ignored/Remaining Linear layers: {len(linear_layers)}")
    for layer in linear_layers:
        print(f"  - {layer}")
        if layer in expected_ignored_layers:
            print(f"    ✅ Expected to be ignored")
        else:
            print(f"    ⚠️  Unexpected - should this be quantized?")
    
    return {
        'quantized': quantized_layers,
        'linear': linear_layers,
        'total_linear_modules': len(quantized_layers) + len(linear_layers)
    }

if __name__ == "__main__":
    # Test static quantization first
    test_dynamic_quantization()
    
    print("\n" + "="*60 + "\n")
    
    # Option 1: Quick test with smaller model (uncomment to use)
    # model_name = "microsoft/DialoGPT-small"  # Much smaller for testing
    
    # Option 2: Full Llama 2 7B (default)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
    
    # Load tokenizer too
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = None
        print("Could not load tokenizer")
    
    print(f"Original model: {count_parameters(model)}")
    
    print("\nQuantizing linear layers with static quantization + parameter scales...")
    model = replace_linear_with_quantized_linear(model, ignore_layers=["lm_head"])
    
    print(f"\nQuantized model: {count_parameters(model)}")
    
    # Verify that ignore_layers worked correctly
    verification = verify_quantization(model, expected_ignored_layers=["lm_head"])
    
    # Detailed inspection of quantized layers
    inspect_quantized_model(model, max_layers=3)
    
    # Option: Push to Hugging Face Hub for easy viewing
    push_to_hub = input("\n🤔 Push to Hugging Face Hub? (y/n): ").lower().strip() == 'y'
    
    if push_to_hub:
        # You'll need to change this to your username
        your_username = input("Enter your HF username: ").strip()
        base_name = model_name.split('/')[-1]
        repo_name = f"{your_username}/{base_name}-bf16-quantized.w8a8"
        
        print(f"\n🚀 Pushing quantized model to: {repo_name}")
        success = save_and_push_to_hub(
            model=model,
            tokenizer=tokenizer,
            original_model_name=model_name,
            quantized_repo_name=repo_name,
            push_to_hub=True
        )
        
        if success:
            print(f"\n🎉 Success! View your model at:")
            print(f"🔗 https://huggingface.co/{repo_name}")
            print(f"📊 Safetensors viewer: https://huggingface.co/{repo_name}/tree/main")
    else:
        # Just save locally
        save_quantized_model(model, f"{model_name.split('/')[-1]}_quantized.safetensors")
    
    print("\nModel structure (first few layers):")
    # Print first few layers to verify quantization
    for name, module in list(model.named_modules())[:10]:
        if isinstance(module, W8A16Linear):
            print(f"  {name}: W8A16Linear({module.in_features}, {module.out_features})")
        elif isinstance(module, nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features}) [NOT QUANTIZED]")