# Ovi 1.1 vs Current Implementation - Comparison Analysis

**Date:** 2025-11-15  
**Source:** https://github.com/character-ai/Ovi (Ovi 1.1 release)  
**Current:** ComfyUI-Ovi (adapted from older version with ROCm fixes)

## Key Findings

### ✅ IMPROVEMENT: Better offload_to_cpu() Method

**Ovi 1.1 has one critical improvement in memory cleanup:**

```python
# Ovi 1.1 version:
def offload_to_cpu(self, model):
    model = model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # ← ADDED! Releases IPC memory
    return model
```

**Our current version:**
```python
def offload_to_cpu(self, model):
    model = model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Missing: torch.cuda.ipc_collect()
    return model
```

**Impact:** This could help with memory cleanup, though we're already calling it in our new cleanup code.

---

### â OUR IMPLEMENTATION IS BETTER: Attention System

**Ovi 1.1 attention.py:**
- âī¸ **Requires flash-attn package** (asserts FLASH_ATTN_2_AVAILABLE)
- â **NOT compatible with Windows ROCm** (no flash-attn package available)
- â **No fallback** to PyTorch SDPA

**Our current attention.py:**
- â **Flexible backend system** (flash_attn_2, flash_attn_3, sage, sdpa)
- â **Fallback to SDPA** when flash-attn unavailable
- â **Works with AOTriton** on Windows ROCm
- â **Better for gfx1151** (proven 1920x speedup with AOTriton)

**Recommendation:** KEEP our current attention implementation - it's superior for ROCm!

---

### â OUR IMPLEMENTATION IS BETTER: Separate decode_latents()

**Ovi 1.1:**
- Decodes inline in `generate()` method
- No separate `decode_latents()` method
- Less flexible for ComfyUI node architecture

**Our implementation:**
- Separate `decode_latents()` method
- Better fits ComfyUI's node-based architecture  
- Allows decode in separate node (better UX)
- More modular and reusable

**Recommendation:** KEEP our separate decode_latents() approach

---

### đ NEUTRAL: Model Specifications

**Ovi 1.1 introduces model variants:**
```python
NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {...},
    "960x960_5s": {...},  # NEW
    "960x960_10s": {...}, # NEW - 10 second videos!
}
```

**Impact:** 
- Supports higher resolution (960x960) 
- Supports longer videos (10s instead of 5s)
- Could add these as options if you get the newer models

**Recommendation:** Optional - only if you want 960x960 or 10s capabilities

---

### đ NEUTRAL: quantize() Support  

**Ovi 1.1 adds qint8 quantization:**
```python
from optimum.quanto import freeze, qint8, quantize

if int8:
    quantize(self.model, qint8)
    freeze(self.model)
```

**Impact:** Reduces VRAM for lower-end GPUs

**Recommendation:** Not needed for your 128GB system, but could add as option

---

### đ NEUTRAL: Text Format Validation

**Ovi 1.1 auto-corrects audio caption format:**
```python
formatted_text_prompt = self.text_formatter(text_prompt)
if formatted_text_prompt != text_prompt:
    logging.info(f"Wrong audio description format detected!")
    text_prompt = formatted_text_prompt
```

**Impact:** Helps users who use wrong format  

**Recommendation:** Nice-to-have QOL feature, not critical

---

## Summary of Actionable Improvements

### đ HIGH PRIORITY: Add ipc_collect() to offload_to_cpu()

**What:** Add `torch.cuda.ipc_collect()` to your offload_to_cpu() method

**Why:** Better IPC memory cleanup (though we already call it in cleanup code)

**Effort:** 1 line change

**Code Change:**
```python
def offload_to_cpu(self, model):
    model = model.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # ← ADD THIS
    return model
```

---

### đ˘ MEDIUM PRIORITY: Optional Model Variants

**What:** Add support for 960x960_5s and 960x960_10s models

**Why:** Higher resolution and longer videos if needed

**Effort:** Moderate - need to download new models and update config

**Benefit:** Better quality videos at higher res (if you have the models)

---

### đ  LOW PRIORITY: Add qint8 quantization option

**What:** Add optional qint8 quantization support

**Why:** Could be useful for others, not needed for your 128GB system

**Effort:** Small - add quantization logic in __init__

**Benefit:** Minimal for your use case

---

## What We Should NOT Adopt

### â **DON'T** Replace Attention System
- Ovi 1.1 requires flash-attn package
- Won't work on Windows ROCm
- Our current system with AOTriton is better

### â **DON'T** Merge decode_latents() into generate()  
- Separate method works better for ComfyUI nodes
- More flexible architecture
- Already working well

### â **DON'T** Remove Our Recent Fixes
- Our fusion model unloading is critical
- Our text encoder unloading is beneficial
- Our aggressive defragmentation is necessary
- None of these are in Ovi 1.1

---

## Recommended Action Plan

### Immediate (Apply Now):

1. **Add `ipc_collect()` to `offload_to_cpu()` method** ✅
   - Simple 1-line addition
   - Consistent with Ovi 1.1
   - Can't hurt, might help

### Future (Optional):

2. **Consider 960x960 models** if you want higher res
   - Requires downloading new checkpoints
   - Update config path logic
   - Add model variant selection to nodes

3. **Add text format validation** for better UX
   - Auto-correct <AUDCAP> format
   - Helpful error messages

---

##  Bottom Line

**Your current implementation is actually BETTER for ROCm gfx1151!**

The only useful addition from Ovi 1.1 is the `ipc_collect()` call in `offload_to_cpu()`, which we can add immediately.

Everything else:
- â Attention: Yours is better (AOTriton support)
- â Decode: Yours is better (modularity)
- â Memory: Yours is better (fusion model unloading)
- đ Model variants: Optional upgrade
- đ Quantization: Not needed for 128GB

---

**Next Step:** Add the `ipc_collect()` improvement and test your OOM fix!
