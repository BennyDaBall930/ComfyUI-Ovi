"""
Standalone GGUF loader for OVI models.
Based on ComfyUI-GGUF loader but embedded directly in OVI to avoid dependencies.
(c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""
import warnings
import logging
import torch
import gguf

# Support both package and standalone imports
try:
    from .ggml_dequant import is_quantized, dequantize_tensor
    from .ggml_tensor import GGMLTensor
except ImportError:
    from ggml_dequant import is_quantized, dequantize_tensor
    from ggml_tensor import GGMLTensor


def get_orig_shape(reader, tensor_name):
    """Get original tensor shape from GGUF metadata."""
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_field(reader, field_name, field_type):
    """Get a field value from GGUF reader."""
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]])
    else:
        raise TypeError(f"Unknown field type {field_type}")


def gguf_sd_loader(path, handle_prefix=None, lazy_dequant=True):
    """
    Load GGUF file and return state dict compatible with PyTorch models.
    
    Args:
        path: Path to GGUF file
        handle_prefix: Optional prefix to strip from tensor names
        lazy_dequant: If True, return GGMLTensor for Q4_K (lazy), else eagerly dequantize
        
    Returns:
        dict: State dictionary with tensors
    """
    reader = gguf.GGUFReader(path)
    
    # Filter and strip prefix if specified
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
    
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix and handle_prefix is not None:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))
    
    # Detect architecture
    arch_str = get_field(reader, "general.architecture", str)
    if arch_str:
        logging.info(f"GGUF architecture detected: {arch_str}")
    
    # Load tensors
    state_dict = {}
    qtype_dict = {}
    
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        
        # Convert numpy array to torch tensor (suppress mmap warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw_data = torch.from_numpy(tensor.data)
        
        # Get target shape from GGUF tensor metadata (already correct)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        
        # Check for original shape metadata (from converter reshaping)
        orig_shape = get_orig_shape(reader, tensor_name)
        if orig_shape is not None:
            shape = orig_shape
        
        # Decode based on tensor type
        if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            # F32 - direct view
            torch_tensor = raw_data.view(torch.float32).view(*shape)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
            # F16 - direct view
            torch_tensor = raw_data.view(torch.float16).view(*shape)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            # BF16 - stored as uint16, need to reinterpret
            torch_tensor = raw_data.view(torch.uint16).view(torch.bfloat16).view(*shape)
        else:
            # Quantized types (Q4_K, Q5_K, etc.)
            if is_quantized(tensor.tensor_type):
                if lazy_dequant:
                    # Return GGMLTensor for lazy dequantization (saves VRAM!)
                    torch_tensor = GGMLTensor(
                        raw_data,
                        tensor_type=tensor.tensor_type,
                        tensor_shape=shape
                    )
                    logging.debug(f"Loaded lazy Q4_K {sd_key}: {tensor.tensor_type.name} -> {shape}")
                else:
                    # Eager dequantization (old behavior)
                    try:
                        torch_tensor = dequantize_tensor(
                            raw_data, 
                            tensor.tensor_type, 
                            shape,
                            dtype=torch.float16
                        )
                        logging.debug(f"Dequantized {sd_key}: {tensor.tensor_type.name} -> {torch_tensor.shape}")
                    except Exception as e:
                        logging.error(f"Failed to dequantize {sd_key} ({tensor.tensor_type.name}): {e}")
                        raise
            else:
                # Should not reach here, but fallback to raw data
                logging.warning(f"Unknown tensor type for {sd_key}: {tensor.tensor_type}")
                torch_tensor = raw_data
        
        # Fix shape for modulation tensors that lost first dimension during quantization
        # video_model/audio_model blocks need (1, 6, 3072), head needs (1, 2, 3072)
        if "modulation" in sd_key and torch_tensor.ndim == 2:
            # Add missing batch dimension
            torch_tensor = torch_tensor.unsqueeze(0)
            logging.debug(f"Fixed modulation shape for {sd_key}: {shape} -> {torch_tensor.shape}")
        
        state_dict[sd_key] = torch_tensor
        
        # Track tensor types for logging
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
    
    # Log loaded tensor types
    logging.info("GGUF tensor types: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))
    
    return state_dict
