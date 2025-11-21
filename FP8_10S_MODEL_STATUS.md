# FP8 10s Model Availability

**Date:** 2025-11-15  
**User Request:** Get FP8 version of 960x960_10s model

## Status: NOT AVAILABLE ❌

### Official Repository Check

Checked **chetwinlow1/Ovi** HuggingFace repository:

**Available Models:**
- ✅ `model.safetensors` - 720x720_5s bf16 (23.3 GB)
- ✅ `model_960x960.safetensors` - 960x960_5s bf16 (23.3 GB)
- ✅ `model_960x960_10s.safetensors` - 960x960_10s bf16 (23.3 GB)

**NOT Available:**
- ❌ `model_960x960_10s_fp8.safetensors` - Does not exist
- ❌ `model_960x960_fp8.safetensors` - Does not exist

### Official Ovi 1.1 Code Confirmation

From Ovi 1.1's `ovi_fusion_engine.py`:

```python
if fp8:
    assert model_name == "720x720_5s", \
        "FP8 quantization is only supported for 720x720_5s model currently."
    basename = "model_fp8_e4m3fn.safetensors"
```

**Character.AI explicitly states:** FP8 quantization only supports 720x720_5s model.

---

## Why FP8 Not Available for 960x960_10s?

### Technical Reasons:

1. **Model-Specific Quantization**
   - FP8 quantization requires careful calibration per model
   - Different resolutions/durations have different activation patterns
   - Character.AI only quantized the base model

2. **Quality Concerns**
   - Higher resolution models may degrade more with FP8
   - 10s model has 2x the temporal complexity
   - Maintaining quality at FP8 with 960x960_10s is harder

3. **Development Priority**
   - Base 720x720_5s model is most popular
   - FP8 targets users with limited VRAM
   - Users with 128GB VRAM (like you) can use BF16

---

## Your Options

### Option 1: Use BF16 960x960_10s ✅ RECOMMENDED

**Pros:**
- ✅ You have 128GB VRAM - no problem!
- ✅ Better quality than FP8
- ✅ Full 10 second duration
- ✅ 960x960 resolution
- ✅ Works with fusion unloading fix

**Cons:**
- ~23GB vs ~12GB (but you have plenty of VRAM)

**With your fixes, memory usage:**
- Model load: ~25GB
- Peak diffusion: ~90-100GB
- After cleanup: ~5GB  
- VAE decode: ~30GB
- **Total peak: ~100GB** ✅ Fits in 128GB!

---

### Option 2: Use 720x720_5s FP8

**Pros:**
- ✅ Smaller file (~12GB)
- ✅ FP8 quantization

**Cons:**
- ❌ Only 5 seconds (not 10s)
- ❌ Lower resolution (720x720)
- ❌ Slightly lower quality vs BF16

---

### Option 3: Create Your Own FP8 Quantization

If you really need FP8 for the 10s model, you could:

1. Use `optimum.quanto` to quantize the BF16 model yourself
2. Would need testing/calibration
3. Risk of quality degradation
4. Not officially supported

**Code would be:**
```python
from optimum.quanto import freeze, qint8, quantize

# After loading model
if int8:
    quantize(self.model, qint8)
    freeze(self.model)
```

But this isn't tested for 960x960_10s and could produce poor results.

---

## Recommendation

**Use BF16 960x960_10s model!**

Your system has:
- ✅ 128GB unified VRAM
- ✅ Fusion model unloading (frees 30GB)
- ✅ Aggressive memory management
- ✅ No need for FP8 compression

The FP8 version is for users with 16-24GB VRAM who can't fit the full BF16 model. **You don't have this limitation!**

---

## If You Still Want FP8...

I can:
1. ✅ Add experimental qint8 quantization option to the loader
2. ⚠️ Quality/stability not guaranteed for 960x960_10s
3. ⚠️ Not officially supported by Character.AI

Let me know if you want me to add this experimental option, or if you'll use the BF16 model (recommended!).
