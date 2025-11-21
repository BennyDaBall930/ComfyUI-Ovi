"""Load Wan VAE and UMT5 encoder using ComfyUI-style selectors."""
from __future__ import annotations

import logging
from pathlib import Path

import folder_paths
import torch

# Register GGUF extension support for text encoders
# This ensures .gguf files appear in the dropdown
try:
    if 'text_encoders' in folder_paths.folder_names_and_paths:
        base_paths, extensions = folder_paths.folder_names_and_paths['text_encoders']
        if isinstance(extensions, set):
            extensions.add('.gguf')
            logging.info("[OVI] Registered .gguf extension for text_encoders")
except Exception as e:
    logging.warning(f"[OVI] Could not register .gguf for text_encoders: {e}")

# Also try to add .gguf as a supported extension globally
try:
    folder_paths.supported_pt_extensions.add('.gguf')
except Exception:
    pass


class OviWanComponentLoader:
    """
    Load Wan VAE and UMT5 text encoder components.
    
    Supported formats:
    - VAE: .pth, .pt, .safetensors
    - Text Encoder: .pth, .pt, .safetensors, .gguf
    
    GGUF text encoders provide disk space savings while maintaining quality.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        import os
        import glob
        
        vae_files = folder_paths.get_filename_list('vae') or ['']
        
        # Get text encoder files including .gguf manually
        text_files = []
        if 'text_encoders' in folder_paths.folder_names_and_paths:
            text_encoder_paths = folder_paths.folder_names_and_paths['text_encoders'][0]
            if not isinstance(text_encoder_paths, list):
                text_encoder_paths = [text_encoder_paths]
            
            # Scan for all supported extensions including .gguf
            for path in text_encoder_paths:
                if os.path.exists(path):
                    for ext in ['.pt', '.pth', '.safetensors', '.gguf']:
                        pattern = os.path.join(path, f'*{ext}')
                        files = glob.glob(pattern)
                        text_files.extend([os.path.basename(f) for f in files])
        
        # Remove duplicates and sort
        text_files = sorted(list(set(text_files))) or ['']
        
        # Set defaults with preference for common files
        if 'Wan2.2_VAE.pth' in vae_files:
            default_vae = 'Wan2.2_VAE.pth'
        elif 'wan2.2_vae.safetensors' in vae_files:
            default_vae = 'wan2.2_vae.safetensors'
        else:
            default_vae = vae_files[0]
        
        if 'models_t5_umt5-xxl-enc-bf16.pth' in text_files:
            default_umt5 = 'models_t5_umt5-xxl-enc-bf16.pth'
        elif 'umt5-xxl-enc-bf16.safetensors' in text_files:
            default_umt5 = 'umt5-xxl-enc-bf16.safetensors'
        elif 'umt5-xxl-enc-fp8_e4m3fn.safetensors' in text_files:
            default_umt5 = 'umt5-xxl-enc-fp8_e4m3fn.safetensors'
        else:
            default_umt5 = text_files[0]
        
        return {
            "required": {
                "engine": ("OVI_ENGINE",),
                "vae_file": (vae_files, {"default": default_vae}),
                "umt5_file": (text_files, {"default": default_umt5, "tooltip": "Supports .pth, .pt, .safetensors, and .gguf formats"}),
            },
        }

    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("components",)
    FUNCTION = "load"
    CATEGORY = "Ovi"

    def load(self, engine, vae_file: str, umt5_file: str, tokenizer: str = ''):
        # Lazy imports to avoid dependency issues at node registration time
        from ovi.modules.t5 import T5EncoderModel
        from ovi.modules.vae2_2 import Wan2_2_VAE
        from ovi.ovi_fusion_engine import OviFusionEngine

        if not isinstance(engine, OviFusionEngine):
            raise TypeError("engine input must come from OviEngineLoader")

        vae_path = Path(folder_paths.get_full_path_or_raise('vae', vae_file)).resolve()
        umt5_path = Path(folder_paths.get_full_path_or_raise('text_encoders', umt5_file)).resolve()

        wan_device = engine.device if not getattr(engine, "cpu_offload", False) else 'cpu'
        wan_vae = Wan2_2_VAE(device=wan_device, vae_pth=str(vae_path))
        wan_vae.model.requires_grad_(False).eval()

        tokenizer_path = Path(engine.get_config().ckpt_dir) / 'google' / 'umt5-xxl'
        if not tokenizer_path.exists():
            raise FileNotFoundError(f'Wan tokenizer not found at {tokenizer_path}. Run OviEngineLoader with auto_download or place it manually.')

        text_device = engine.device if not getattr(engine, "cpu_offload", False) else 'cpu'
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=text_device,
            checkpoint_path=str(umt5_path),
            tokenizer_path=str(tokenizer_path),
            shard_fn=None,
        )

        engine.override_models(video_vae=wan_vae, text_model=text_encoder)
        return (engine,)
