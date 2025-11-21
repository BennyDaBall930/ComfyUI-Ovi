# GGUF Quantization Setup Guide

This guide will help you set up the tools needed to quantize OVI models to GGUF Q4_K_M format.

## Prerequisites

- Python 3.10+
- Git
- CMake (for building llama.cpp)
- Visual Studio Build Tools (Windows) or GCC (Linux)

## Step 1: Install ComfyUI-GGUF

```bash
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import gguf; print('✅ GGUF installed successfully')"
```

## Step 2: Install llama.cpp (For Quantization)

### Windows (PowerShell or CMD)

```powershell
# Navigate to ComfyUI root
cd C:\Users\Benjamin\AI\ComfyUI

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CMake
cmake -B build
cmake --build build --config Release

# Verify
.\build\bin\Release\quantize.exe
```

**Using the automated script (recommended):**
```bash
cd custom_nodes\ComfyUI-Ovi\tools
setup_llama_cpp.bat
```

### Linux / Mac

```bash
# Navigate to ComfyUI root
cd ~/ComfyUI

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with make
make

# Verify
./quantize
```

### Troubleshooting Build Issues

**Windows: If CMake not found**
```powershell
# Install CMake via winget
winget install Kitware.CMake

# Or download from: https://cmake.org/download/
```

**Windows: If Visual Studio Build Tools not found**
```powershell
# Download Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/

# Or install via winget
winget install Microsoft.VisualStudio.2022.BuildTools
```

**Linux: If GCC not found**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake

# Fedora/RHEL
sudo dnf install gcc gcc-c++ cmake

# Arch
sudo pacman -S base-devel cmake
```

## Step 3: Verify Installation

Run the verification script:

```bash
cd custom_nodes/ComfyUI-Ovi/tools
python quantize_ovi_model.py --help
```

You should see:
- ✅ ComfyUI-GGUF found
- ✅ Found quantize tool at: [path]

## Step 4: Quantize Your Models

### Quantize Fusion Model (Ovi-960x960-10s)

```bash
# Navigate to models directory
cd models/diffusion_models

# Run quantization
python ../../custom_nodes/ComfyUI-Ovi/tools/quantize_ovi_model.py Ovi-960x960-10s.safetensors
```

This will create `Ovi-960x960-10s-Q4_K_M.gguf` in `models/diffusion_models/`

### Quantize Text Encoder

```bash
# Navigate to OVI ckpts directory
cd custom_nodes/ComfyUI-Ovi/ckpts/Wan2.2-TI2V-5B

# Run quantization
python ../../tools/quantize_ovi_model.py models_t5_umt5-xxl-enc-bf16.pth --type text_encoder
```

This will create `models_t5_umt5-xxl-enc-Q4_K_M.gguf` in `models/text_encoders/`

## Quantization Options

### Different Quantization Levels

The script defaults to Q4_K_M. To use other quantization levels, modify the script or manually run llama.cpp quantize:

```bash
# Q5_K_M (better quality, slightly larger)
quantize.exe input-F16.gguf output-Q5_K_M.gguf Q5_K_M

# Q6_K (high quality, larger)
quantize.exe input-F16.gguf output-Q6_K.gguf Q6_K

# Q8_0 (highest quality, close to original size)
quantize.exe input-F16.gguf output-Q8_0.gguf Q8_0
```

### Advanced Usage

```bash
# Keep intermediate F16 file
python quantize_ovi_model.py model.safetensors --keep-f16

# Custom output location
python quantize_ovi_model.py model.safetensors --output custom-name.gguf --no-move

# Custom quantize tool path
python quantize_ovi_model.py model.safetensors --quantize-tool C:\path\to\quantize.exe
```

## Memory Requirements

**During quantization:**
- F16 GGUF creation: ~1.5x model size in RAM
- Q4_K_M quantization: ~2x F16 size in RAM

**For Ovi-960x960-10s (~23GB model):**
- Need ~35GB free RAM for F16 conversion
- Need ~50GB free RAM for Q4_K_M quantization
- Total process time: 15-30 minutes (depending on CPU)

**For text encoder (~3GB model):**
- Need ~5GB free RAM for F16 conversion
- Need ~7GB free RAM for Q4_K_M quantization
- Total process time: 2-5 minutes

## Expected File Sizes

### Fusion Models

| Model | Original (BF16) | F16 GGUF | Q4_K_M GGUF | Savings |
|-------|----------------|----------|-------------|---------|
| Ovi-11B-5s | ~23 GB | ~23 GB | ~6.5 GB | 72% |
| Ovi-960x960-5s | ~23 GB | ~23 GB | ~6.5 GB | 72% |
| Ovi-960x960-10s | ~23 GB | ~23 GB | ~6.5 GB | 72% |

### Text Encoder

| Model | Original (BF16) | F16 GGUF | Q4_K_M GGUF | Savings |
|-------|----------------|----------|-------------|---------|
| UMT5-XXL | ~3 GB | ~3 GB | ~1.8 GB | 40% |

## Automated Batch Quantization

To quantize all your OVI models at once:

```bash
cd custom_nodes/ComfyUI-Ovi/tools
python batch_quantize.py
```

This will find all safetensors models in your diffusion_models folder and quantize them.

## Troubleshooting

### "ComfyUI-GGUF not found"
- Ensure ComfyUI-GGUF is installed in `custom_nodes/ComfyUI-GGUF`
- Run: `cd custom_nodes && git clone https://github.com/city96/ComfyUI-GGUF.git`

### "quantize tool not found"
- Build llama.cpp as shown above
- Or provide path: `--quantize-tool /path/to/quantize`

### "Out of memory during quantization"
- Close other applications
- Use swap/pagefile (slower but works)
- Quantize on a machine with more RAM

### "F16 conversion fails"
- Check if input file is valid safetensors/pth
- Ensure you have write permissions
- Check disk space (~30GB free needed)

### "ValueError: Can only handle tensor names up to 127 characters"
- This is a GGUF limitation
- Contact model author to shorten tensor names
- Or manually edit state dict keys before conversion

## Quality Comparison

After quantization, compare outputs:

1. Generate with original BF16 model
2. Generate with Q4_K_M GGUF (same seed, same prompt)
3. Compare visually

**Expected quality:** Q4_K_M should be ~95% similar to original BF16. Differences are typically imperceptible.

If quality is noticeably worse, try Q5_K_M or Q6_K quantization.

## Next Steps

After quantization:
1. GGUF files will be in `models/diffusion_models/` or `models/text_encoders/`
2. Restart ComfyUI
3. Select GGUF model from OviEngineLoader dropdown
4. Select GGUF text encoder from OviWanComponentLoader dropdown
5. Generate and enjoy 65-70% VRAM savings!

## References

- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
