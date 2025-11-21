import torch
import os
import json
import logging
from pathlib import Path

from safetensors.torch import load_file

from ovi.modules.fusion import FusionModel
from ovi.modules.t5 import T5EncoderModel
from ovi.modules.vae2_2 import Wan2_2_VAE
from ovi.modules.mmaudio.features_utils import FeaturesUtils

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"

# Import local GGUF loader (no external dependencies)
try:
    from .gguf_loader import gguf_sd_loader as _gguf_sd_loader
    from .ggml_ops import patch_linear_layers, check_quantized_weights
    from .ggml_tensor import GGMLTensor
    _gguf_available = True
except ImportError as e:
    logging.warning(f"GGUF support not available: {e}")
    _gguf_sd_loader = None
    _gguf_available = False
    GGMLTensor = None
    patch_linear_layers = None
    check_quantized_weights = None


def init_wan_vae_2_2(ckpt_dir, rank=0):
    device = rank
    device_index = 0
    if isinstance(device, int):
        device_index = device
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            device_index = int(device.split(":", 1)[1])
        except (ValueError, IndexError):
            device_index = 0

    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "device_count") and device_index < torch.cuda.device_count():
                torch.cuda.set_device(device_index)
            supports_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        except Exception:
            supports_bf16 = False
    else:
        supports_bf16 = False

    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    vae_model = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        dtype=dtype,
        device=device,
    )

    return vae_model


def init_mmaudio_vae(ckpt_dir, rank=0):
    vae_config = {}
    vae_config['mode'] = '16k'
    vae_config['need_vae_encoder'] = True

    tod_vae_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(ckpt_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config['tod_vae_ckpt'] = tod_vae_ckpt
    vae_config['bigvgan_vocoder_ckpt'] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae


def init_fusion_score_model_ovi(rank: int = 0, meta_init=False):
    video_config_path = CONFIG_ROOT / "model" / "dit" / "video.json"
    audio_config_path = CONFIG_ROOT / "model" / "dit" / "audio.json"

    if not video_config_path.exists():
        raise FileNotFoundError(f"Missing video config at {video_config_path}")
    if not audio_config_path.exists():
        raise FileNotFoundError(f"Missing audio config at {audio_config_path}")

    with video_config_path.open() as f:
        video_config = json.load(f)

    with audio_config_path.open() as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)

    params_all = sum(p.numel() for p in fusion_model.parameters())

    if rank == 0:
        print(
            f"Score model (Fusion) all parameters:{params_all}"
        )

    return fusion_model, video_config, audio_config


def init_text_model(ckpt_dir, rank):
    wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        shard_fn=None)

    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, lazy_dequant=False):
    """
    Load fusion model checkpoint with optional lazy dequantization for GGUF.
    
    Args:
        model: FusionModel to load into
        checkpoint_path: Path to checkpoint file
        from_meta: Whether model was initialized with meta tensors
        lazy_dequant: If True, keep Q4_K as GGMLTensor (experimental, has gradient issues)
    
    Note: lazy_dequant is disabled by default due to PyTorch gradient compatibility.
          GGMLTensor (uint8 dtype) cannot be used as parameters that require gradients.
          Use eager dequantization (default) for stability.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".gguf"):
            # Load GGUF quantized model using local OVI GGUF loader
            if not _gguf_available or _gguf_sd_loader is None:
                raise RuntimeError(
                    "GGUF model loading failed. Please ensure 'gguf' Python package is installed: "
                    "pip install gguf"
                )
            
            lazy_mode_str = "lazy (VRAM saving)" if lazy_dequant else "eager (full dequant)"
            logging.info(f"[OVI GGUF] Loading GGUF fusion model from {checkpoint_path} [{lazy_mode_str}]")
            df = _gguf_sd_loader(checkpoint_path, handle_prefix=None, lazy_dequant=lazy_dequant)
            logging.info(f"[OVI GGUF] Loaded {len(df)} tensors successfully")
        elif checkpoint_path.endswith(".safetensors"):
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df['module'] if 'module' in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df['app']['model']
        else:
            raise RuntimeError("We only support .safetensors, .pt, and .gguf checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)

        # Check if we have quantized tensors (GGMLTensor)
        has_ggml_tensors = False
        if _gguf_available and GGMLTensor is not None and checkpoint_path.endswith(".gguf") and lazy_dequant:
            for param in model.parameters():
                if isinstance(param, GGMLTensor):
                    has_ggml_tensors = True
                    break
        
        # Patch Linear layers if we have quantized weights
        if has_ggml_tensors and patch_linear_layers is not None:
            logging.info("[OVI GGUF] Detected GGMLTensor weights, patching Linear layers...")
            patched_count = patch_linear_layers(model)
            logging.info(f"[OVI GGUF] Patched {patched_count} Linear layers for lazy dequantization")
            
            # Log VRAM savings
            if check_quantized_weights is not None:
                q_count, total_count, q_size_mb, dequant_size_mb = check_quantized_weights(model)
                vram_saved_mb = dequant_size_mb - q_size_mb
                vram_saved_pct = (vram_saved_mb / dequant_size_mb * 100) if dequant_size_mb > 0 else 0
                logging.info(f"[OVI GGUF] Quantized: {q_count}/{total_count} parameters")
                logging.info(f"[OVI GGUF] VRAM Usage: {q_size_mb:.1f}MB (compressed) vs {dequant_size_mb:.1f}MB (full)")
                logging.info(f"[OVI GGUF] VRAM Saved: {vram_saved_mb:.1f}MB ({vram_saved_pct:.1f}%)")

        del df
        import gc
        gc.collect()
        print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    else:
        raise RuntimeError(f"{checkpoint_path} does not exist")


def load_text_encoder_checkpoint(checkpoint_path):
    """Load text encoder checkpoint, supporting .pth, .pt, .safetensors, and .gguf formats."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Text encoder checkpoint not found: {checkpoint_path}")
    
    if checkpoint_path.endswith(".gguf"):
        # Load GGUF quantized text encoder
        if _gguf_clip_loader is None:
            raise RuntimeError(
                "GGUF text encoder loading requires ComfyUI-GGUF to be installed. "
                "Please install it in custom_nodes/ComfyUI-GGUF"
            )
        logging.info(f"Loading GGUF text encoder from {checkpoint_path}")
        state_dict = _gguf_clip_loader(checkpoint_path)
        logging.info("GGUF text encoder loaded successfully with quantized weights")
        return state_dict
    elif checkpoint_path.endswith((".pth", ".pt")):
        # Load PyTorch checkpoint
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if 'module' in state_dict:
                state_dict = state_dict['module']
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        return state_dict
    elif checkpoint_path.endswith(".safetensors"):
        # Load safetensors
        return load_file(checkpoint_path, device="cpu")
    else:
        raise RuntimeError(
            f"Unsupported text encoder format: {checkpoint_path}. "
            "Expected .pth, .pt, .safetensors, or .gguf"
        )
