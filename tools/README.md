# OVI GGUF Quantization Tools

This directory contains tools for quantizing OVI models to GGUF format for reduced VRAM usage.

## Quick Start

### Windows

```batch
REM 1. Setup quantization tools (one-time)
setup_llama_cpp.bat

REM 2. Quantize your model
cd ..\..\..
python custom_nodes\ComfyUI-Ovi\tools\quantize_ovi_model.py models\diffusion_models\Ovi-960x960-10s.safetensors
```

### Linux/Mac

```bash
# 1. Setup (manual - see QUANTIZATION_SETUP.md)
cd ~/ComfyUI
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make && cd ..

# 2. Quantize your model
python custom_nodes/ComfyUI-Ovi/tools/quantize_ovi_model.py models/diffusion_models/Ovi-960x960-10s.safetensors
```

## Files

### `quantize_ovi_model.py`
Main quantization script. Converts safetensors/pth models to GGUF Q4_K_M format.

**Usage:**
```bash
python quantize_ovi_model.py <input_file> [options]

Options:
  --type {diffusion,text_encoder}  Model type (default: diffusion)
  --output OUTPUT                  Custom output filename
  --keep-f16                       Keep intermediate F16 GGUF
  --no-move                        Don't move to models directory
  --quantize-tool PATH             Path to llama.cpp quantize tool
```

**Examples:**
```bash
# Quantize fusion model
python quantize_ovi_model.py Ovi-960x960-10s.safetensors

# Quantize text encoder
python quantize_ovi_model.py models_t5_umt5-xxl-enc-bf16.pth --type text_encoder

# Custom output
python quantize_ovi_model.py model.safetensors --output custom-q4.gguf --no-move
```

### `setup_llama_cpp.bat` (Windows)
Automated setup script for installing llama.cpp and building the quantize tool.

**Usage:**
```batch
setup_llama_cpp.bat
```

This will:
1. Check prerequisites (git, cmake, VS Build Tools)
2. Clone llama.cpp to ComfyUI root
3. Build with CMake
4. Verify installation

### `QUANTIZATION_SETUP.md`
Comprehensive setup guide covering:
- Prerequisites installation
- Step-by-step setup instructions
- Troubleshooting common issues
- Memory requirements
- Expected file sizes
- Quality comparison tips

## Quantization Workflow

```
BF16 Safetensors (23GB)
        ↓
  [ComfyUI-GGUF convert.py]
        ↓
    F16 GGUF (23GB)
        ↓
  [llama.cpp quantize]
        ↓
  Q4_K_M GGUF (6.5GB) ✅
        ↓
  models/diffusion_models/
```

## Memory Savings

| Model | Original | Q4_K_M | Savings |
|-------|----------|--------|---------|
| Ovi-960x960-10s | ~23 GB | ~6.5 GB | **72%** |
| Ovi-11B-5s | ~23 GB | ~6.5 GB | **72%** |
| UMT5-XXL | ~3 GB | ~1.8 GB | **40%** |

**VRAM Usage:**
- BF16: ~100 GB peak → OOM errors ❌
- Q4_K_M: ~70 GB peak → Fits in 128GB ✅

## Prerequisites

1. **ComfyUI-GGUF**
   ```bash
   cd custom_nodes
   git clone https://github.com/city96/ComfyUI-GGUF.git
   cd ComfyUI-GGUF
   pip install -r requirements.txt
   ```

2. **llama.cpp** (for quantization)
   - Windows: Run `setup_llama_cpp.bat`
   - Linux/Mac: See QUANTIZATION_SETUP.md

## Quantization Options

The script defaults to Q4_K_M (best size/quality balance). For other quantization levels:

```bash
# After creating F16 GGUF, manually run:
quantize input-F16.gguf output-Q5_K_M.gguf Q5_K_M  # Better quality
quantize input-F16.gguf output-Q6_K.gguf Q6_K      # High quality
quantize input-F16.gguf output-Q8_0.gguf Q8_0      # Highest quality
```

| Quant | Size | Quality | Recommendation |
|-------|------|---------|----------------|
| Q4_K_M | ~6.5 GB | 95% | Best for VRAM-constrained systems |
| Q5_K_M | ~8 GB | 97% | Balanced (recommended) |
| Q6_K | ~10 GB | 98% | High quality |
| Q8_0 | ~12 GB | 99%+ | Near-original quality |

## Troubleshooting

### "ComfyUI-GGUF not found"
Install ComfyUI-GGUF in custom_nodes/

### "quantize tool not found"
- Windows: Run `setup_llama_cpp.bat`
- Linux/Mac: Build llama.cpp manually
- Or: Provide path with `--quantize-tool`

### "Out of memory during quantization"
Close other applications or quantize on a machine with more RAM. Need ~35GB for fusion models, ~5GB for text encoders.

### Build errors (Windows)
- Ensure Visual Studio Build Tools installed
- Try running from "Developer Command Prompt for VS"
- Check QUANTIZATION_SETUP.md for detailed solutions

## After Quantization

1. GGUF files are automatically moved to:
   - `models/diffusion_models/` (fusion models)
   - `models/text_encoders/` (text encoders)

2. Restart ComfyUI

3. Select GGUF model from dropdown in OviEngineLoader

4. Generate with 65-70% less VRAM usage!

## Support

For issues or questions:
1. Check QUANTIZATION_SETUP.md
2. Review GGUF_SUPPORT.md in parent directory
3. Verify prerequisites are properly installed
4. Check ComfyUI console for error messages

## References

- [Parent Documentation](../GGUF_SUPPORT.md)
- [Implementation Summary](../GGUF_IMPLEMENTATION_SUMMARY.md)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
