"""
GGMLTensor implementation for ComfyUI-Ovi lazy quantization support.
Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0
"""
import torch
import gguf
import logging


class GGMLTensor(torch.Tensor):
    """
    Custom tensor class for storing quantized weights.
    Stores raw quantized data and metadata for lazy dequantization.
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = []

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        """Override to() to preserve metadata."""
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        """Override clone() to avoid unnecessary cloning."""
        return self

    def detach(self, *args, **kwargs):
        """Override detach() to avoid unnecessary detaching."""
        return self

    def copy_(self, *args, **kwargs):
        """Override copy_() for compatibility."""
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Ignoring 'copy_' on GGMLTensor: {e}")
            return self

    def new_empty(self, size, *args, **kwargs):
        """Override new_empty() for Intel Arc compatibility."""
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            patches=getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        """Return the dequantized shape, not the compressed shape."""
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


def is_quantized_tensor(tensor):
    """Check if a tensor is a quantized GGMLTensor."""
    return isinstance(tensor, GGMLTensor)


def get_dequant_function():
    """
    Get dequantization function, preferring ComfyUI-GGUF if available,
    otherwise falling back to our embedded implementation.
    """
    try:
        # Try to import from ComfyUI-GGUF (preferred)
        import sys
        import os
        gguf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ComfyUI-GGUF")
        if gguf_path not in sys.path:
            sys.path.insert(0, gguf_path)
        
        from dequant import dequantize_tensor as gguf_dequant
        logging.info("[OVI GGUF] Using ComfyUI-GGUF dequantization (preferred)")
        return gguf_dequant
    except ImportError:
        # Fall back to our embedded implementation
        try:
            from .ggml_dequant import dequantize_tensor
            logging.info("[OVI GGUF] Using embedded dequantization (fallback)")
            return dequantize_tensor
        except ImportError:
            from ggml_dequant import dequantize_tensor
            logging.info("[OVI GGUF] Using embedded dequantization (fallback)")
            return dequantize_tensor


def dequantize_ggml_tensor(tensor, dtype=None):
    """
    Dequantize a GGMLTensor to a regular torch.Tensor.
    
    Args:
        tensor: GGMLTensor to dequantize
        dtype: Target dtype (default: torch.float16)
    
    Returns:
        torch.Tensor: Dequantized tensor
    """
    if not isinstance(tensor, GGMLTensor):
        return tensor
    
    if dtype is None:
        dtype = torch.float16
    
    # Check if it's an unquantized type (F32, F16, BF16)
    qtype = getattr(tensor, "tensor_type", None)
    if qtype in (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return tensor.to(dtype)
    
    # Get shape
    shape = getattr(tensor, "tensor_shape", tensor.shape)
    
    # Dequantize using available function
    dequant_fn = get_dequant_function()
    return dequant_fn(tensor.data, qtype, shape, dtype=dtype)
