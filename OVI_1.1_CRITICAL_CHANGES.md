# Ovi 1.1 Critical Changes & 10s Clip Support

**Date:** 2025-11-15  
**Source:** Your fork https://github.com/BennyDaBall930/Ovi + Official Ovi 1.1

## Critical Changes Applied ✅

### 1. Updated clean_text() Function
**File:** `ovi/utils/processing_utils.py`

Added removal of "Audio: ..." lines for 960x960 model compatibility:
```python
# Remove 'Audio: ...' lines (including multiline ones)
text = re.sub(r"Audio:\s*.*", "", text, flags=re.DOTALL)
```

### 2. Added ipc_collect() to offload_to_cpu()
**File:** `ovi/ovi_fusion_engine.py`

```python
def offload_to_cpu(self, model):
    model = model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # ← FROM OVI 1.1
    return model
```

---

## 10 Second Clip Support (NOT YET IMPLEMENTED)

### What's Different in Ovi 1.1 for 10s Clips

**Model Specifications Map:**
```python
NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "video_latent_length": 31,   # 5 seconds
        "audio_latent_length": 157,
        "video_area": 720 * 720,
    },
    "960x960_10s": {
        "video_latent_length": 61,   # 10 seconds! Double the frames
        "audio_latent_length": 314,  # Double the audio tokens
        "video_area": 960 * 960,
    }
}
```

**Your Current Implementation (Hardcoded):**
```python
# In __init__:
self.video_latent_length = 31   # ← Hardcoded for 5s
self.audio_latent_length = 157  # ← Hardcoded for 5s
```

### To Support 10s Clips, You'd Need:

#### 1. Download New Models
```bash
# From character-ai/Ovi or your fork
python download_weights.py --models 960x960_10s
```

Downloads: `model_960x960_10s.safetensors` (~20GB)

#### 2. Update Checkpoint Manager
Add model specifications to your implementation:
```python
# In ovi_fusion_engine.py or utils/checkpoint_manager.py:
NAME_TO_MODEL_SPECS_MAP = {
    "Ovi-11B-bf16.safetensors": {  # Current 5s model
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
    },
    "model_960x960_10s.safetensors": {  # NEW 10s model
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
    },
}
```

#### 3. Make Latent Lengths Configurable
```python
# In __init__:
model_specs = NAME_TO_MODEL_SPECS_MAP.get(model_name, default_specs)
self.video_latent_length = model_specs["video_latent_length"]
self.audio_latent_length = model_specs["audio_latent_length"]
self.target_area = model_specs["video_area"]
```

#### 4. Add Model Selection to Nodes
Update `ovi_engine_loader.py`:
```python
"model_variant": (["720x720_5s", "960x960_5s", "960x960_10s"], 
                  {"default": "720x720_5s"})
```

---

## Why 10s Support Needs More Work

**Memory Impact:**
- 10s model has **2x the latents** (61 vs 31 frames, 314 vs 157 audio)
- **2x memory during diffusion** (~40-50GB instead of 20-25GB)
- **2x memory during VAE decode** (~20-30GB instead of 10-15GB)
- Total peak: **~70-80GB** (still fits in 128GB!)

**Current OOM Fix Compatibility:**
- ✅ Should work! Fusion model unloading still critical
- ✅ 10s uses same architecture, just longer sequences
- ⚠️ May need even more aggressive cleanup

---

## Recommendation:  Test OOM Fix First, Then Add 10s Support

### Phase 1: Test C Current Fix (NOW)
1. Restart ComfyUI
2. Run 5s generation workflow
3. Verify logs show fusion model unloading
4. Confirm GPU VAE decode works without OOM

### Phase 2: Add 10s Support (LATER)
1. Download `model_960x960_10s.safetensors`
2. Add model specs map
3. Make latent lengths configurable  
4. Add model variant selection to UI
5. Test 10s generation

---

## Quick Reference

### Current Configuration (5s @ 720x720)
- Video latents: 31 frames
- Audio latents: 157 tokens
- Model: `Ovi-11B-bf16.safetensors` or `Ovi-11B-fp8.safetensors`

### Future 10s Configuration (10s @ 960x960)
- Video latents: 61 frames (2x)
- Audio latents: 314 tokens (2x)
- Model: `model_960x960_10s.safetensors` (NEW)
- Est. memory: ~70-80GB peak (vs ~40-50GB for 5s)

---

## Next Steps

**Immediate:**
1. ✅ Applied clean_text() fix
2. ✅ Applied ipc_collect() fix
3. ⏳ **Test OOM fix with current 5s model**

**Future (After OOM Fix Verified):**
1. Download 960x960_10s model
2. Implement model variant system
3. Test 10s generation
4. Benchmark memory usage

---

**Bottom Line:** Your OOM fixes should work even better with 10s models since we're now unloading fusion+text (~30GB freed). Test the 5s model first, then we can add 10s support if needed!
