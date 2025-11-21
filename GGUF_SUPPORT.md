# GGUF Model Support for ComfyUI-Ovi

## Overview

ComfyUI-Ovi now supports loading GGUF quantized models for both the fusion diffusion model and UMT5 text encoder. This allows you to use quantized models to reduce VRAM usage significantly.

## Requirements

- **ComfyUI-GGUF** must be installed in `custom_nodes/ComfyUI-GGUF`
- GGUF quantized model files (`.gguf` format)

## Benefits

### Memory Savings

Using GGUF quantized models can dramatically reduce VRAM usage:

| Format | Size | VRAM Usage (approx) |
|--------|------|---------------------|
| BF16 Safetensors | ~23 GB | ~25 GB peak |
| GGUF Q4_K | ~6-8 GB | ~8-10 GB peak |
| GGUF Q5_K | ~8-10 GB | ~10-12 GB peak |
| GGUF Q6_K | ~10-12 GB | ~12-15 GB peak |
| GGUF Q8 | ~12-15 GB | ~15-18 GB peak |

**Example for Ovi-960x960-10s model:**
- Safetensors BF16: ~100 GB peak VRAM (often causes OOM)
- GGUF Q4_K: ~70 GB peak VRAM ✅ Much more manageable!

## Supported Files

### Fusion Models
Place GGUF fusion models in `models/diffusion_models/`:
- `Ovi-11B-bf16.gguf` (720x720, 5s)
- `Ovi-11B-fp8.gguf` (720x720, 5s)
- `Ovi-960x960-5s.gguf` (960x960, 5s)
- `Ovi-960x960-10s.gguf` (960x960, 10s)

### Text Encoders
Place GGUF text encoders in `models/text_encoders/`:
- `models_t5_umt5-xxl-enc-q4_k.gguf`
- `models_t5_umt5-xxl-enc-q5_k.gguf`
- `models_t5_umt5-xxl-enc-q6_k.gguf`
- `models_t5_umt5-xxl-enc-q8.gguf`

### VAE (Not Quantized)
VAE models remain as `.pth` files (not quantized):
- `Wan2.2_VAE.pth` in `models/vae/`

## How to Use

### 1. With OviEngineLoader

The engine loader will automatically detect and load GGUF fusion models if they exist in `models/diffusion_models/`:

```
OviEngineLoader
├─ model_precision: "Ovi-960x960-10s.gguf" 
├─ cpu_offload: False
└─ device: Auto
```

The loader will:
1. Check for GGUF version first (e.g., `Ovi-960x960-10s.gguf`)
2. Fall back to safetensors if GGUF not found
3. Automatically use GGUF loaders from ComfyUI-GGUF

### 2. With OviWanComponentLoader

For text encoders, select the GGUF file from the dropdown:

```
OviWanComponentLoader
├─ engine: [from OviEngineLoader]
├─ vae_file: "Wan2.2_VAE.pth"
└─ umt5_file: "models_t5_umt5-xxl-enc-q5_k.gguf"  ← Select GGUF text encoder
```

The loader will:
1. Detect `.gguf` extension
2. Use `gguf_clip_loader()` for GGUF files
3. Use standard loading for `.pth`/`.pt`/`.safetensors` files

## Implementation Details

### Modified Files

1. **`ovi/utils/model_loading_utils.py`**
   - Added GGUF loader import helper
   - Updated `load_fusion_checkpoint()` to detect and load `.gguf` files
   - Added `load_text_encoder_checkpoint()` helper for GGUF text encoders

2. **`ovi/utils/checkpoint_manager.py`**
   - Registered `.gguf` extension for `diffusion_models` folder recognition

3. **`nodes/ovi_wan_component_loader.py`**
   - Registered `.gguf` extension for `text_encoders` folder recognition
   - Components will now show GGUF files in dropdown

4. **`ovi/ovi_fusion_engine.py`**
   - Already uses updated `load_fusion_checkpoint()` - no changes needed!

### Loading Logic

```python
# Fusion Model Loading
if checkpoint_path.endswith(".gguf"):
    # Use GGUF loader from ComfyUI-GGUF
    df = gguf_sd_loader(checkpoint_path, handle_prefix="model.diffusion_model.")
elif checkpoint_path.endswith(".safetensors"):
    # Standard safetensors loading
    df = load_file(checkpoint_path, device="cpu")
# ... load into model
```

```python
# Text Encoder Loading  
if checkpoint_path.endswith(".gguf"):
    # Use GGUF CLIP loader
    state_dict = gguf_clip_loader(checkpoint_path)
elif checkpoint_path.endswith(".pth"):
    # Standard PyTorch loading
    state_dict = torch.load(checkpoint_path, map_location="cpu")
# ... load into text encoder
```

## Quantization Recommendations

### For Maximum Quality
- **Fusion Model**: GGUF Q6_K or Q8 (~10-15 GB)
- **Text Encoder**: GGUF Q5_K or Q6_K (~2-3 GB)
- **Total VRAM**: ~60-80 GB for 10s model

### For Maximum Memory Savings
- **Fusion Model**: GGUF Q4_K (~6-8 GB)
- **Text Encoder**: GGUF Q4_K (~1.5-2 GB)
- **Total VRAM**: ~50-60 GB for 10s model

### For Balanced Performance
- **Fusion Model**: GGUF Q5_K (~8-10 GB)
- **Text Encoder**: GGUF Q5_K (~2-2.5 GB)
- **Total VRAM**: ~55-70 GB for 10s model

## Creating GGUF Models

To convert existing safetensors models to GGUF format, use llama.cpp quantization tools:

```bash
# For diffusion models
python convert-safetensors-to-gguf.py Ovi-960x960-10s.safetensors \
    --outfile Ovi-960x960-10s-q5_k.gguf \
    --outtype q5_k

# For text encoders
python convert-safetensors-to-gguf.py models_t5_umt5-xxl-enc-bf16.pth \
    --outfile models_t5_umt5-xxl-enc-q5_k.gguf \
    --outtype q5_k
```

Check [ComfyUI-GGUF documentation](https://github.com/city96/ComfyUI-GGUF) for conversion tools.

## Troubleshooting

### Error: "GGUF model loading requires ComfyUI-GGUF to be installed"

**Solution**: Install ComfyUI-GGUF:
```bash
cd custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF
pip install -r requirements.txt
```

### GGUF files not showing in dropdown

**Solution**: Ensure files are in the correct directory:
- Fusion models: `models/diffusion_models/*.gguf`
- Text encoders: `models/text_encoders/*.gguf`
- Restart ComfyUI to refresh file lists

### Quality degradation with Q4_K

**Solution**: Q4_K is the most aggressive quantization. Try:
- Q5_K for better quality with minimal size increase
- Q6_K for near-original quality
- Q8 for best quality (still smaller than BF16)

### OOM errors even with GGUF

**Solution**: 
- Use Q4_K or Q5_K quantization
- Enable `cpu_offload` in OviEngineLoader
- Close other GPU-intensive applications
- Check that you're not loading multiple models simultaneously

## Performance Notes

- **Loading time**: GGUF models may take slightly longer to load initially
- **Inference speed**: Similar to BF16 with modern GPUs (built-in dequantization)
- **Quality**: Q5_K and above are nearly indistinguishable from BF16
- **VRAM fragmentation**: GGUF models help reduce fragmentation issues

## Compatibility

- ✅ Works with all OVI model variants (720x720, 960x960, 5s, 10s)
- ✅ Compatible with CPU offload mode
- ✅ Compatible with FP8 workflows (though GGUF is preferred)
- ✅ Works with both T2V and I2V workflows
- ✅ Compatible with custom attention backends (FlashAttention, xFormers, etc.)

## Advanced Usage

### Mixed Precision
You can mix GGUF and safetensors models:
```
Fusion: Ovi-960x960-10s.gguf (Q5_K)
Text Encoder: models_t5_umt5-xxl-enc-bf16.pth (Full precision)
VAE: Wan2.2_VAE.pth (Full precision)
```

### Automatic Fallback
If GGUF file is missing, the system automatically falls back to safetensors:
```
1st attempt: Ovi-960x960-10s.gguf (not found)
2nd attempt: Ovi-960x960-10s.safetensors (loaded!) ✓
```

## Questions?

For issues or questions about GGUF support:
1. Ensure ComfyUI-GGUF is properly installed
2. Check file paths and naming conventions
3. Verify GGUF files are properly quantized (not corrupted)
4. Check ComfyUI console for detailed error messages
