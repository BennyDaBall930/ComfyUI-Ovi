"""Register ComfyUI-Ovi custom nodes."""

import os
import sys
import logging

_PACKAGE_ROOT = os.path.dirname(__file__)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

# Register GGUF extension for text encoders EARLY (before directory scanning)
try:
    import folder_paths
    
    # Ensure text_encoders folder is registered with proper extensions
    if 'text_encoders' not in folder_paths.folder_names_and_paths:
        # text_encoders folder doesn't exist, create it with GGUF support
        text_encoder_path = os.path.join(folder_paths.models_dir, "text_encoders")
        folder_paths.folder_names_and_paths['text_encoders'] = (
            [text_encoder_path],
            {'.pt', '.pth', '.safetensors', '.gguf'}
        )
        logging.info(f"[OVI] Created text_encoders folder config with .gguf support: {text_encoder_path}")
    else:
        # text_encoders exists, add .gguf to its extensions
        base_paths, extensions = folder_paths.folder_names_and_paths['text_encoders']
        if isinstance(extensions, set):
            extensions.add('.gguf')
            logging.info("[OVI] Early registration: Added .gguf extension for text_encoders")
    
    # Register .gguf as a globally supported extension
    if hasattr(folder_paths, 'supported_pt_extensions'):
        folder_paths.supported_pt_extensions.add('.gguf')
        logging.info("[OVI] Early registration: Added .gguf to supported_pt_extensions")
        
except Exception as e:
    logging.warning(f"[OVI] Could not register .gguf extension early: {e}")
    import traceback
    traceback.print_exc()

from .nodes.ovi_engine_loader import OviEngineLoader
from .nodes.ovi_video_generator import OviVideoGenerator
from .nodes.ovi_attention_selector import OviAttentionSelector
from .nodes.ovi_wan_component_loader import OviWanComponentLoader
from .nodes.ovi_latent_decoder import OviLatentDecoder
from .nodes.ovi_latent_io import SaveOviLatents, LoadOviLatents


NODE_CLASS_MAPPINGS = {
    "OviEngineLoader": OviEngineLoader,
    "OviVideoGenerator": OviVideoGenerator,
    "OviAttentionSelector": OviAttentionSelector,
    "OviWanComponentLoader": OviWanComponentLoader,
    "OviLatentDecoder": OviLatentDecoder,
    "SaveOviLatents": SaveOviLatents,
    "LoadOviLatents": LoadOviLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OviEngineLoader": "OVI Engine Loader",
    "OviVideoGenerator": "OVI Video Generator",
    "OviAttentionSelector": "OVI Attention Selector",
    "OviWanComponentLoader": "OVI Wan Component Loader",
    "OviLatentDecoder": "OVI Latent Decoder",
    "SaveOviLatents": "Save OVI Latents",
    "LoadOviLatents": "Load OVI Latents",
}
