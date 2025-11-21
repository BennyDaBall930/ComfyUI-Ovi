# ComfyUI-Ovi Development Logs & Research Notes

**"The Chronicles of Making It Work"**

This document aggregates all the research, status updates, and victories from the development of the ComfyUI-Ovi ROCm edition. If you're wondering "why did they do that?" or "how did they fix the OOM?", the answers are buried here.

---

## 📋 Table of Contents

1.  [All-GPU Mode & Memory Fixes](#1-all-gpu-mode--memory-fixes)
2.  [GGUF Quantization Saga](#2-gguf-quantization-saga)
3.  [Ovi 1.1 Upgrades](#3-ovi-11-upgrades)
4.  [FP8 & Model Availability](#4-fp8--model-availability)
5.  [Node Loading Crisis](#5-node-loading-crisis)

---

## 1. All-GPU Mode & Memory Fixes

**Date:** 2025-11-15  
**System:** AMD Strix Halo gfx1151 (128GB unified VRAM)

### The Challenge
Despite having 128GB of unified memory, we hit OOM during VAE decode.
- **Root Cause:** The 25GB fusion model + 5GB text encoder stayed loaded during the memory-intensive VAE decode step.
- **Fragmentation:** Memory was allocated but fragmented, leaving only ~1.17GB contiguous for the VAE.

### The Solution (The "Get Off My Lawn" Approach)
We implemented an **All-GPU Mode** with aggressive cleanup:

1.  **Hardcoded GPU:** `cpu_offload = False`. We have 128GB; we use it.
2.  **Aggressive Unloading:** Explicitly unload the Fusion Model and Text Encoder after the diffusion loop finishes.
3.  **Defragmentation:** Run `torch.cuda.empty_cache()` and `ipc_collect()` multiple times before VAE decode.
4.  **Prevent Reloads:** Modified `decode_latents()` to stop it from accidentally reloading the fusion model.

**Result:** ~95GB free memory available for VAE decode. Zero OOMs.

### Status: ✅ COMPLETED

---

## 2. GGUF Quantization Saga

**Date:** 2025-01-16

### Goal
Run the massive 10s Ovi model on consumer GPUs (16-24GB) by shrinking the ~23GB model to ~7GB.

### Achievements
- **GGUF Loading:** Successfully implemented loading for `.gguf` files using `comfyui-gguf` logic.
- **Text Encoder:** Full support for quantized T5 encoders (Q4_K, Q8_0).
    - *Savings:* Reduces text encoder from 15GB -> 4GB.
- **Q4_K Dequantization:** Ported block-based dequantization to fix shape mismatch errors.
    - *Original Error:* `size mismatch... shape [3072, 1728] vs [3072, 3072]`
    - *Fix:* Implemented proper block unpacking.

### The Current Hurdle (Help Wanted!)
While we can *load* GGUF files, the **quantization tool** itself (`quantize_ovi_model.py`) struggles with Ovi's unique twin-backbone architecture. We need help ensuring the tensors are packed correctly so the loader doesn't choke on them.

### VRAM Savings (Projected)
| Model | BF16 | GGUF Q4_K | Savings |
|-------|------|-----------|---------|
| Fusion | 23 GB | ~7 GB | **70%** |
| Text Enc | 15 GB | ~4 GB | **73%** |

### Status: 🚧 IN PROGRESS (Loader Ready, Quantizer Needs Fix)

---

## 3. Ovi 1.1 Upgrades

**Date:** 2025-11-15

We synced with the official Ovi 1.1 release but kept our ROCm-specific optimizations.

### What We Took
- **`ipc_collect()`:** Added to `offload_to_cpu()` for better memory hygiene.
- **Text Cleaning:** Updated `clean_text()` to handle "Audio:" prefixes better.
- **Model Specs:** Added support for `960x960_10s` (10-second clips!) and `960x960_5s`.

### What We Rejected (And Why)
- **Attention:** Ovi 1.1 forces `flash-attn`. We stuck with our flexible backend (AOTriton/SDPA) because `flash-attn` doesn't exist for Windows ROCm.
- **Inline Decode:** Ovi 1.1 merged decoding into generation. We kept it separate for better ComfyUI modularity.

### Status: ✅ MERGED

---

## 4. FP8 & Model Availability

**Date:** 2025-11-15

### The Request
"Get me the FP8 version of the 10s model!"

### The Reality Check
It doesn't exist. Character.AI only released FP8 for the base 5s model.
- **Why?** Quantizing 10s temporal data is hard; quality degrades.
- **Alternative:** Use GGUF Q4_K/Q8_0 (once the quantizer is fixed) or stick to BF16 if you have the VRAM (which Strix Halo does).

### Status: ❌ NOT AVAILABLE (Use BF16 or GGUF)

---

## 5. Node Loading Crisis

**Date:** 2025-11-16

### The Panic
"ModuleNotFoundError: No module named 'diffusers'" -> All nodes vanished.

### The Fix
**Lazy Imports.**
The `ovi_wan_component_loader.py` was importing heavy modules at the top level, triggering a dependency chain that required `diffusers` (which ComfyUI doesn't have by default). We moved imports inside the `load()` method.

**Lesson:** Never import heavy dependencies at the top level of a ComfyUI node.

### Status: ✅ FIXED

---

*End of Logs. Now go generate something cool.*
