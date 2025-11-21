"""
GGML-aware operations for ComfyUI-Ovi lazy quantization.
Provides Linear layers that can handle GGMLTensor weights with on-the-fly dequantization.
"""
import torch
import torch.nn as nn
import logging
from .ggml_tensor import GGMLTensor, dequantize_ggml_tensor


class GGMLLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear that supports GGMLTensor weights.
    Dequantizes weights on-the-fly during forward pass.
    """
    
    def forward(self, input):
        # Check if weight is quantized
        if isinstance(self.weight, GGMLTensor):
            # Dequantize on-the-fly
            weight = dequantize_ggml_tensor(self.weight, dtype=input.dtype)
            bias = dequantize_ggml_tensor(self.bias, dtype=input.dtype) if self.bias is not None else None
            return torch.nn.functional.linear(input, weight, bias)
        else:
            # Regular forward pass
            return super().forward(input)


def try_use_comfyui_gguf_ops():
    """
    Try to use ComfyUI-GGUF's GGMLOps if available (preferred).
    Falls back to our simple implementation if not available.
    
    Returns:
        GGMLOps class or None if not available
    """
    try:
        import sys
        import os
        
        # Add ComfyUI-GGUF to path
        gguf_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "ComfyUI-GGUF"
        )
        
        if gguf_path not in sys.path:
            sys.path.insert(0, gguf_path)
        
        # Try to import GGMLOps
        from ops import GGMLOps
        logging.info("[OVI GGUF] Using ComfyUI-GGUF GGMLOps (preferred - full optimizations)")
        return GGMLOps
    except ImportError as e:
        logging.info(f"[OVI GGUF] ComfyUI-GGUF not available, using simple GGMLLinear fallback: {e}")
        return None


def get_ggml_ops():
    """
    Get GGML operations class.
    Tries ComfyUI-GGUF first, falls back to simple implementation.
    
    Returns:
        Ops class with Linear attribute
    """
    # Try ComfyUI-GGUF first
    ggml_ops = try_use_comfyui_gguf_ops()
    if ggml_ops is not None:
        return ggml_ops
    
    # Fallback: create simple ops class
    class SimpleGGMLOps:
        """Simple GGML operations using our GGMLLinear."""
        Linear = GGMLLinear
    
    logging.info("[OVI GGUF] Using simple GGMLLinear (fallback)")
    return SimpleGGMLOps


def patch_linear_layers(module, ops_class=None):
    """
    Recursively replace nn.Linear layers with GGML-aware Linear layers.
    
    Args:
        module: PyTorch module to patch
        ops_class: Operations class (default: auto-detect)
    
    Returns:
        int: Number of layers patched
    """
    if ops_class is None:
        ops_class = get_ggml_ops()
    
    patched_count = 0
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, GGMLLinear):
            # Get Linear class from ops
            if hasattr(ops_class, 'Linear'):
                LinearClass = ops_class.Linear
            else:
                LinearClass = GGMLLinear
            
            # Create new GGML-aware layer
            new_layer = LinearClass(
                child.in_features,
                child.out_features,
                bias=child.bias is not None
            )
            
            # Copy weights and bias (may be GGMLTensor)
            new_layer.weight = child.weight
            new_layer.bias = child.bias
            
            # Replace the layer
            setattr(module, name, new_layer)
            patched_count += 1
            
        else:
            # Recursively patch children
            patched_count += patch_linear_layers(child, ops_class)
    
    return patched_count


def check_quantized_weights(module):
    """
    Count how many parameters are GGMLTensor (quantized).
    
    Returns:
        tuple: (quantized_count, total_count, quantized_size_mb, dequant_size_mb)
    """
    quantized_count = 0
    total_count = 0
    quantized_bytes = 0
    dequant_bytes = 0
    
    for name, param in module.named_parameters():
        total_count += 1
        if isinstance(param, GGMLTensor):
            quantized_count += 1
            # Compressed size
            quantized_bytes += param.data.numel() * param.data.element_size()
            # Decompressed size (FP16)
            shape = getattr(param, 'tensor_shape', param.shape)
            dequant_bytes += shape.numel() * 2  # 2 bytes for FP16
        else:
            # Regular tensor
            dequant_bytes += param.numel() * param.element_size()
            quantized_bytes += param.numel() * param.element_size()
    
    return (
        quantized_count,
        total_count,
        quantized_bytes / (1024 * 1024),
        dequant_bytes / (1024 * 1024)
    )
