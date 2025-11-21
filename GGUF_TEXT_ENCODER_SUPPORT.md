# GGUF Text Encoder Support for ComfyUI-Ovi

## Overview

ComfyUI-Ovi now supports **GGUF format text encoders** (UMT5-XXL) in addition to the standard .pth and .safetensors formats. This enables significant disk space savings for the text encoder component.

## Supported Formats

### Text Encoders (UMT5-XXL)
- ✓ `.pth` - Original PyTorch format
- ✓ `.pt` - PyTorch format
- ✓ `.safetensors` - SafeTensors format
- ✓ **`.gguf`** - GGUF quantized format (NEW!)

### VAE
- ✓ `.pth` - PyTorch format
- ✓ `.safetensors` - SafeTensors format

## Usage

### In ComfyUI

1. **Place GGUF text encoder** in `models/text_encoders/` directory
   ```
   ComfyUI/
   └── models/
       └── text_encoders/
           ├── umt5-xxl-enc-bf16.safetensors  (existing)
           └── umt5-xxl-enc-q4k.gguf          (new!)
   ```

2. **Use OviWanComponentLoader node:**
   - Connect to `OviEngineLoader` output
   - Select VAE file (e.g., `wan2.2_vae.safetensors`)
   - Select text encoder file (now includes `.gguf` options!)
   - GGUF files will appear in the dropdown automatically

3. **Node will automatically detect format:**
   - `.gguf` → Uses OVI GGUF loader with proper dequantization
   - `.safetensors` → Uses SafeTensors loader
   - `.pth`/`.pt` → Uses PyTorch loader

## Benefits of GGUF Text Encoders

| Format | File Size | Quality | Use Case |
|--------|-----------|---------|----------|
| BF16 (.pth) | ~15GB | Best | Maximum quality, high VRAM |
| FP16 (.safetensors) | ~15GB | Excellent | Standard quality |
| FP8 (.safetensors) | ~7.5GB | Very Good | Low VRAM systems |
| **Q4_K (.gguf)** | **~4GB** | **Good** | **Minimum disk space** |
| **Q6_K (.gguf)** | **~5.5GB** | **Very Good** | **Balanced** |
| **Q8_0 (.gguf)** | **~7GB** | **Excellent** | **Near-original quality** |

### Advantages

✓ **Disk Space Savings:** 60-75% smaller than BF16/FP16  
✓ **Fast Loading:** Efficient binary format  
✓ **Quality Preservation:** Minimal loss with Q6_K or Q8_0  
✓ **Compatible:** Works with existing workflows  
✓ **Automatic Dequantization:** Handled transparently  

## Creating GGUF Text Encoders

### Using ComfyUI-GGUF Tools

If you have the original UMT5-XXL weights, you can convert to GGUF:

```bash
cd custom_nodes/ComfyUI-GGUF/tools

# Convert to F16 GGUF first
python convert.py \
  --src path/to/umt5-xxl-enc-bf16.pth \
  --dst path/to/umt5-xxl-enc-f16.gguf

# Quantize to Q4_K (smallest)
llama-quantize \
  path/to/umt5-xxl-enc-f16.gguf \
  path/to/umt5-xxl-enc-q4k.gguf \
  Q4_K

# Or Q6_K (better quality)
llama-quantize \
  path/to/umt5-xxl-enc-f16.gguf \
  path/to/umt5-xxl-enc-q6k.gguf \
  Q6_K
```

### Recommended Quantization Levels

**For T5 Text Encoder:**
- **Q8_0** - Best choice (7GB, minimal quality loss)
- **Q6_K** - Good balance (5.5GB, very good quality)
- **Q4_K** - Maximum savings (4GB, acceptable quality)

**Not recommended:**
- Q2_K/Q3_K - May degrade text understanding
- Lower precision might affect prompt comprehension

## Technical Details

### How It Works

1. **Registration:** `.gguf` extension registered for `text_encoders` folder
2. **Detection:** `T5EncoderModel._load_weights()` detects `.gguf` suffix
3. **Loading:** Uses `gguf_loader.gguf_sd_loader()` with eager dequantization
4. **Dequantization:** Q4_K/Q6_K quantized weights → FP16 tensors
5. **Conversion:** Model dtype cast to BF16 for inference

### Eager Dequantization

Text encoders use **eager dequantization** (not lazy):
- GGUF loaded → immediately dequantized to FP16
- Then cast to BF16 for model
- **Why:** Text encoder parameters must support gradients
- **VRAM:** Same as FP16 during inference (~15GB)
- **Benefit:** Disk space savings only

### VRAM Usage Example

```
Model Component          | Disk Size | VRAM Usage
-------------------------|-----------|------------
Fusion Model (Q4_K GGUF) | 7.3GB     | ~23GB (eager dequant)
Text Encoder (Q4_K GGUF) | 4GB       | ~15GB (eager dequant)
VAE (SafeTensors)        | 200MB     | ~500MB
-------------------------|-----------|------------
Total                    | ~11.5GB   | ~38GB
```

**vs Original:**
```
Fusion Model (SafeTensors) | 23GB    | ~23GB
Text Encoder (BF16)        | 15GB    | ~15GB
VAE (PyTorch)              | 200MB   | ~500MB
----------------------------|---------|--------
Total                      | ~38GB   | ~38GB
```

**Savings:** ~70% disk space, same VRAM usage

## Troubleshooting

### Issue: GGUF files not showing in dropdown

**Solution 1:** Restart ComfyUI or reload custom nodes
**Solution 2:** Check file is in correct directory:
```
models/text_encoders/your-file.gguf
```

**Solution 3:** Check permissions - file must be readable

### Issue: "gguf module not found"

**Solution:** Install gguf package:
```bash
pip install gguf
```

### Issue: Loading fails with shape mismatch

**Problem:** GGUF file might be incorrectly quantized or corrupted

**Solutions:**
1. Re-download or re-create the GGUF file
2. Use a different quantization level (try Q8_0 or Q6_K)
3. Fall back to SafeTensors format

### Issue: Poor text understanding

**Problem:** Quantization level too aggressive (Q2_K/Q3_K)

**Solutions:**
1. Use higher precision: Q6_K or Q8_0
2. Fall back to FP8 SafeTensors
3. Use original BF16 format

## Logging

### Successful GGUF Load

```
Loading GGUF text encoder from models/text_encoders/umt5-xxl-enc-q4k.gguf
GGUF architecture detected: t5
GGUF tensor types: Q4_K (728), F32 (50)
GGUF text encoder loaded successfully: 778 tensors
```

### Format Auto-Detection

The loader automatically detects file format by extension:
- `.gguf` → GGUF loader (with dequantization)
- `.safetensors` → SafeTensors loader
- `.pth`/`.pt` → PyTorch loader

## File Locations

### Input Directory
```
ComfyUI/models/text_encoders/
```

Place your GGUF text encoder files here. They will automatically appear in the `OviWanComponentLoader` dropdown.

### Supported Naming Patterns

Any filename works, but for clarity consider:
- `umt5-xxl-enc-q4k.gguf`
- `umt5-xxl-enc-q6k.gguf`
- `umt5-xxl-enc-q8_0.gguf`
- `umt5-xxl-enc-f16.gguf`

## Code Changes Summary

### Modified Files

1. **`ovi/modules/t5.py`**
   - Updated `_load_weights()` to support `.gguf` format
   - Imports `gguf_loader` when needed
   - Auto-detects format by file extension

2. **`nodes/ovi_wan_component_loader.py`**
   - Registered `.gguf` extension for `text_encoders` folder
   - Updated class docstring with format support
   - Added tooltip for umt5_file parameter

3. **`ovi/utils/gguf_loader.py`** (already supporting diffusion models)
   - Now also used for text encoder loading
   - Handles Q4_K, Q6_K, Q8_0 dequantization
   - Works with both fusion models and text encoders

## Performance Characteristics

### Loading Time
- **GGUF Q4_K:** Fast (4GB file, compact format)
- **SafeTensors:** Fast (15GB file, optimized format)
- **PyTorch .pth:** Slower (15GB file, legacy pickle format)

### Inference Speed
- **All formats:** Identical (all dequantized to BF16)
- No performance difference during generation

### Memory Usage
- **All formats:** ~15GB VRAM for text encoder
- GGUF saves disk space, not VRAM

## Recommendations

### For Most Users
**Q6_K GGUF** - Best balance of size and quality (~5.5GB)

### For Limited Disk Space
**Q4_K GGUF** - Maximum compression (~4GB)

### For Maximum Quality
**BF16 SafeTensors or PyTorch** - No quantization loss (~15GB)

### For Low VRAM
**FP8 SafeTensors** - Already well optimized (~7.5GB disk, ~7GB VRAM)

## Notes

- GGUF text encoders use **eager dequantization** (not lazy)
- VRAM usage is the same as non-GGUF formats
- Main benefit is **disk space savings**
- Quality is preserved well with Q6_K or Q8_0
- Compatible with all OVI workflows

## Future Improvements

Potential enhancements:
- Lazy dequantization for text encoders (requires gradient-free parameters)
- On-the-fly quantization during text encoding
- Mixed precision text encoders (quantize less important layers)
- Caching of frequently used text embeddings

---

**Status:** ✓ Production Ready  
**Date:** 2025-11-16  
**Disk Savings:** 60-75% vs original  
**Quality:** Excellent with Q6_K/Q8_0
