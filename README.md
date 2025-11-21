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

### 1. The "I Am Strix Halo" Variables
If PyTorch refuses to acknowledge your shiny new APU, you might need to force the GFX version. Set these in your environment or bat file before launching ComfyUI.

```bat
:: For Strix Halo (Ryzen AI Max+ 395)
:: Try 11.5.1 first. If that fails, lie and say it's 11.5.0.
set HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### 2. Memory Management (Critical for APUs)
Since your "VRAM" is just system RAM that took a gym class, you need to manage fragmentation.
```bat
:: Helps prevent fragmentation when your VRAM is actually system RAM
set PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
```

### 3. Recommended ComfyUI Launch Flags
Add these to your `python main.py` command:

- `--use-pytorch-cross-attention`: **Highly Recommended.** ROCm generally prefers this over the custom optimization paths that NVIDIA cards use.
- `--force-fp32`: If VAEs are still crashing despite our internal fixes, this global flag might save you.
- **DO NOT USE** `--directml`. We are using native ROCm. Do not insult the hardware.

**Example `run_ovi.bat`:**
```bat
@echo off
set HSA_OVERRIDE_GFX_VERSION=11.5.1
set PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
python main.py --use-pytorch-cross-attention
pause
```

## 🐛 Troubleshooting

**"It crashed during VAE decode!"**
- It shouldn't. We forced it to `float32`. If it still crashes, your VRAM is probably crying. Try the FP8 weights or buy more RAM.

**"I'm running out of memory!"**
- We added aggressive garbage collection, but Strix Halo (gfx1151) unified memory is a shared resource. Close Chrome. Yes, all the tabs.

**"Flash Attention isn't working!"**
- It's ROCm on Windows. We try to use AOTriton. If that fails, we fall back to SDPA or vanilla attention. It's not a bug, it's a "feature" of the ecosystem.

## ⚖️ Credits
- Original Ovi implementation by Character.AI.
- Suffering and ROCm porting by BennyDaBall930.
- Wan 2.2 VAE and UMT5 folks for the heavy lifting.

---
*Disclaimer: This software is provided "as is". If it melts your GPU or becomes sentient, I'm not responsible.*
