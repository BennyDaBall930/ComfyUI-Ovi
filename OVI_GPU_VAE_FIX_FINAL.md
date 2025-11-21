# Ovi GPU VAE Decode Fix - Final Implementation

**Date:** 2025-11-15  
**Issue:** OOM during VAE decode even with 128GB VRAM  
**Root Cause:** Fusion model (25GB) not unloading before VAE decode  

## The Problem

From your error log:
```
[OVI][ENGINE][DEBUG] GPU mem before cleanup -> alloc=25.21 GB, reserved=42.78 GB
[OVI][ENGINE][DEBUG] GPU mem after cleanup -> alloc=25.20 GB, reserved=25.91 GB
```

**Issue:** Fusion model was still loaded = 25GB allocated!  
**Result:** Not enough contiguous memory for VAE decode (needs 6.54  GB BUT only 1.17 GB unallocated due to fragmentation)

## The Solution - 3 Critical Fixes

### 1. Unload Fusion Model After Diffusion
Added to `generate()` after diffusion completes:
```python
# Unload fusion model (25GB freed!)
if hasattr(self, 'model') and self.model is not None:
    self.model = self.model.to('cpu')
    del self.model
    self.model = None
```

### 2. Unload Text Encoder
Also unload text encoder (~5GB freed):
```python
# Unload text encoder (no longer needed)
if hasattr(self, 'text_model') and self.text_model is not None:
    if hasattr(self.text_model, 'model'):
        self.text_model.model = self.text_model.model.to('cpu')
    del self.text_model
    self.text_model = None
```

### 3. Aggressive Defragmentation
```python
# Defragment VRAM (5x empty_cache + reset_peak_stats)
for i in range(5):
    torch.cuda.empty_cache()
    if i < 4:
        torch.cuda.synchronize()

torch.cuda.ipc_collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(self.device)
```

### 4. Prevent Fusion Model Reload in decode_latents()
**CRITICAL FIX:** Removed `ensure_loaded()` call that was reloading fusion model!

```python
# OLD (BAD):
def decode_latents(...):
    self.ensure_loaded()  # ← This reloads fusion model!

# NEW (GOOD):
def decode_latents(...):
    # DON'T call ensure_loaded() - would reload 25GB fusion model!
    # Only check VAEs are available
    if getattr(self, "vae_model_audio", None) is None:
        raise RuntimeError("Audio VAE not loaded")
```

## Expected Memory After Cleanup

**Before (Old):**
- Fusion model: 25GB ❌ Still loaded
- Text encoder: 5GB ❌ Still loaded  
- Latents: 2GB
- **Total: ~32GB allocated**
- Free (after fragmentation): Only 1.17GB! → OOM

**After (New):**
- Fusion model: 0GB ✅ Unloaded to CPU
- Text encoder: 0GB ✅ Unloaded to CPU
- Video VAE: 12GB (will load for decode)
- Audio VAE: 3GB
- Latents: 2GB
- **Total: ~17GB** (before VAE decode)
- **Total during decode: ~20GB** (with VAE loaded)
- **Free: ~95GB!** → No OOM! 🎉

## IMPORTANT: You Must Restart ComfyUI!

**The changes require a restart to take effect:**

1. Close ComfyUI (Ctrl+C in terminal)
2. Restart: `Launch ComfyUI.bat`
3. Run your workflow again
4. Watch the logs for:
   - `[OVI][ENGINE][DEBUG] Unloading fusion model...`
   - `[OVI][ENGINE][DEBUG] Fusion model unloaded`
   - `[OVI][ENGINE][DEBUG] Text encoder unloaded`
   - After cleanup: Should see **~2-5 GB allocated** (not 25GB!)

## What to Look For in Logs

### ✅ SUCCESS Indicators:
```
[OVI][ENGINE][DEBUG] Starting aggressive memory cleanup after diffusion
[OVI][ENGINE][DEBUG] Unloading fusion model to free VRAM for VAE
[OVI][ENGINE][DEBUG] Fusion model unloaded
[OVI][ENGINE][DEBUG] Text encoder unloaded
[OVI][ENGINE][DEBUG] Starting aggressive CUDA defragmentation
[OVI][ENGINE][DEBUG] GPU mem after cleanup -> alloc=2.50 GB, reserved=15.00 GB, free=92.00 GB
```

### ❌ FAILURE Indicators (Old Behavior):
```
[OVI][ENGINE][DEBUG] GPU mem after cleanup -> alloc=25.20 GB  ← Fusion still loaded!
```

## If Still OOM After Restart

If you still get OOM **after restarting ComfyUI**, check:

1. **Log shows fusion unloaded?** Look for "Fusion model unloaded" message
2. **Allocated memory after cleanup?** Should be <10GB, not 25GB
3. **Reserved memory?** Should drop significantly after defragmentation
4. **Other models loaded?** Check if other ComfyUI nodes have models in VRAM

## Performance Trade-off

**Trade-off:** Fusion model unloads after generation, must reload for next generation

- ✅ **Pro:** VAE decode now works on GPU (much faster than CPU!)
- âšī¸ **Con:** Next generation requires reloading fusion model (~10-15 seconds)
- đĄ **Net:** Still faster overall since GPU VAE >> CPU VAE decode
