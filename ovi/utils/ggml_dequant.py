"""
GGML Dequantization functions for OVI GGUF support.
Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0
"""
import gguf
import torch
import logging


TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


def is_quantized(tensor_type):
    """Check if a tensor type is quantized (not F32/F16)."""
    return tensor_type not in TORCH_COMPATIBLE_QTYPES


def dequantize_tensor(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype.
    
    Args:
        data: Raw quantized data as torch.Tensor
        qtype: gguf.GGMLQuantizationType
        oshape: Original/target shape
        dtype: Target dtype (default: torch.float16)
    
    Returns:
        torch.Tensor: Dequantized tensor with correct shape
    """
    if dtype is None:
        dtype = torch.float16
    
    # Handle unquantized types
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return data.to(dtype)
    
    # Check if we have a dequantize function for this type
    if qtype not in dequantize_functions:
        logging.warning(f"No fast dequantization for {qtype}, falling back to numpy")
        import numpy as np
        new = gguf.quants.dequantize(data.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(data.device, dtype=dtype)
    
    # Fast dequantization
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    
    # Reshape to blocks
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    
    # Dequantize blocks
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    
    # Reshape to target shape
    return blocks.reshape(oshape)


def to_uint32(x):
    """Convert bytes to uint32 (no uint32 in PyTorch)."""
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks, *args):
    """Split blocks into dimensions."""
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# BF16 Dequantization
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    """Dequantize BF16 blocks."""
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# Q8_0 Dequantization
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q8_0 blocks."""
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)


# Q4_K Dequantization (the one we need!)
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    """Extract scale and min from K-quant scales."""
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min_val = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min_val.reshape((n_blocks, 8)))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    """Dequantize Q4_K blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    """Dequantize Q5_K blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    """Dequantize Q6_K blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    """Dequantize Q3_K blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    """Dequantize Q2_K blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


# Legacy quant types
def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q4_0 blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    """Dequantize Q4_1 blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    """Dequantize Q5_0 blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    """Dequantize Q5_1 blocks."""
    if dtype is None:
        dtype = torch.float16
        
    n_blocks = blocks.shape[0]
    device = blocks.device

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m


# Dequantization function registry
dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}
