# ComfyUI-Ovi: The "Fine, I'll do it myself" ROCm Edition

Welcome to the fork that exists because AMD users deserve nice things too.

This is a modified version of **ComfyUI-Ovi** designed to run on AMD hardware (specifically verified on Strix Halo / gfx1151) via ROCm on Windows. If you're running an NVIDIA card, you *could* use this, but why are you here? Go back to the easy life.

## 🎯 Key Features (ROCm Survival Kit)

- **AMD Strix Halo (gfx1151) Support**: Verified on AMD Ryzen AI Max+ 395. Because we like living on the bleeding edge.
- **Windows ROCm AOTriton Integration**: Since `flash-attn` decided to leave us on read, we use AOTriton for attention when available.
- **The "Please Don't Crash" VAE Fix**: We force the VAE to run in `float32`. Why? Because mixed-precision decode on ROCm crashes harder than a unicycle on ice.
- **Aggressive Memory Hygiene**: We call `torch.cuda.ipc_collect()` like a nervous cleaning lady to keep unified memory from exploding into an OOM fireball.

## 🛠️ Installation

1. **Clone this repo** into your `custom_nodes` folder:
   ```bash
   git clone https://github.com/BennyDaBall930/ComfyUI-Ovi.git
   cd ComfyUI-Ovi
   ```

2. **Install Requirements** (ROCm Flavor - The Rock Edition):
   You need the **nightly** builds for the latest hardware support (especially Strix Halo/gfx1151). We recommend using "The Rock" approach or pointing directly to the nightly index.
   
   Our `requirements.txt` is set up to pull from the nightly ROCm wheels.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have a specific ROCm version requirement, check [The Rock](https://github.com/ROCm/TheRock) for compatible nightly builds.*

3. **Restart ComfyUI**. Pray to the GPU gods.

## 🚀 Environment & Launcher Flags (The "Turbo Mode" Button)

ROCm on Windows is like a cat; it does what it wants unless you give it very specific instructions. Here is how to wrangle it, especially for Strix Halo.

### 1. The "I Am Strix Halo" Variables (Launch ComfyUI.bat)
If PyTorch refuses to acknowledge your shiny new APU, you might need to force the GFX version. We also use aggressive memory settings to prevent fragmentation on unified memory.

```bat
@echo off
echo ========================================
echo Starting ComfyUI [WINDOWS - STRIX HALO OPTIMIZED]
echo GMKtec Evo-X2 - AMD Ryzen AI Max+ 395
echo Unified Memory Architecture
echo ========================================

REM ===== Memory Management - Unified APU Mode =====
REM Enable Zero-Copy: Allows CPU to write directly to GPU memory (Best for APUs)
set HIP_HOST_COHERENT=1
set HSA_ENABLE_INTERRUPT=0

REM Aggressive Allocator: 8GB splits + Lazy Garbage Collection (80%)
REM Prevents fragmentation on huge 100GB+ pool; delays cleanup to save bandwidth
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:8192,garbage_collection_threshold:0.8
set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:8192,garbage_collection_threshold:0.8

REM ===== ROCm Device Settings =====
REM Force gfx1151 to load ROCm kernels and fix MIOpen VAE failures
set HSA_OVERRIDE_GFX_VERSION=11.5.1
set MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1
REM Enable experimental AOTriton kernels required for SDPA on ROCm
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

REM ===== Threading Optimization =====
REM Set to Physical Cores (e.g., 16) to avoid logical thread scheduler thrashing
set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set OPENBLAS_NUM_THREADS=16
set TORCH_NUM_THREADS=16

REM ===== OVI Specific Debug Flags =====
REM set OVI_VAE_FORCE_DTYPE=fp16  (Use if you feel lucky)
set OVI_VAE_DEBUG=1
set OVI_ENGINE_DEBUG=1
set OVI_VAE_TILE_SIZE=256
REM set OVI_VAE_DECODE_ON_CPU=1 (Uncomment if VRAM is tight, but Strix Halo usually handles it)

echo Activating Python environment...
call .venv\Scripts\activate.bat

echo Launching ComfyUI...
python main.py --preview-method auto --highvram --use-pytorch-cross-attention --listen 0.0.0.0
pause
```

### 2. Key Flags Explained

- **`HSA_OVERRIDE_GFX_VERSION=11.5.1`**: Forces the runtime to treat your GPU as gfx1151 (Strix Halo). Critical for detection.
- **`HIP_HOST_COHERENT=1`**: Enables Zero-Copy for unified memory, reducing overhead between CPU and GPU portions of the APU.
- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`**: Enables Flash Attention (via AOTriton) support, which is otherwise disabled by default on Windows ROCm.
- **`MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1`**: Forces a naive convolution path in MIOpen, which fixes specific VAE decoding crashes.
- **`OVI_VAE_TILE_SIZE=256`**: Reduces tile size for VAE decoding to manage peak memory usage.

## 🐛 Troubleshooting

**"It crashed during VAE decode!"**
- It shouldn't. We forced it to `float32` by default. If you set `OVI_VAE_FORCE_DTYPE=fp16`, try unsetting it.
- Ensure `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=1` is set.

**"I'm running out of memory!"**
- Unified memory is shared. Close Chrome.
- Check `PYTORCH_HIP_ALLOC_CONF`. The `garbage_collection_threshold:0.8` helps delays fragmentation cleanup, but `max_split_size_mb:8192` ensures large contiguous blocks are preserved.

**"Flash Attention isn't working!"**
- It's ROCm on Windows. We try to use AOTriton (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). If that fails, we fall back to SDPA or vanilla attention.

## ⚖️ Credits
- Original Ovi implementation by Character.AI.
- Suffering and ROCm porting by BennyDaBall930.
- Wan 2.2 VAE and UMT5 folks for the heavy lifting.

## 📉 Quantization & GGUF Support (Work in Progress)

We are actively working on GGUF quantization to help users with 16GB-24GB cards run the 11B model without aggressive offloading.

**Current State:**
- **Tools:** A quantization setup guide and scripts are available in the `tools/` directory.
- **Docs:** See `tools/QUANTIZATION_SETUP.md` for instructions on how to build `llama.cpp` and quantize Ovi models to Q4_K_M.
- **Goal:** Reduce the ~23GB BF16 model to ~6.5GB (Q4_K_M), saving ~70% VRAM.

**Current Capabilities:**
- **Ovi Engine Loading:** ✅ **Supported!** The fusion loader is ready to load GGUF files.
- **Text Encoder:** ✅ **Supported!** You can load GGUF quantized T5 encoders right now.
- **Quantization Tool:** 🚧 **Help Wanted!** This is where we need you. Our quantization script (`quantize_ovi_model.py`) struggles with Ovi's unique twin-backbone (Audio/Video) architecture. It needs to be updated to properly handle and pack the specific tensor structure of the Ovi engine into a valid GGUF file.

**How to Help:**
1. Check `tools/QUANTIZATION_SETUP.md` for the current setup.
2. Dive into `tools/quantize_ovi_model.py` and help us figure out the correct tensor mapping for the twin backbone.
3. Successfully create a working GGUF file that the loader accepts!

---
*Disclaimer: This software is provided "as is". If it melts your GPU or becomes sentient, I'm not responsible.*
