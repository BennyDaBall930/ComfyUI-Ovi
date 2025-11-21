"""ComfyUI node for initializing and caching the OviFusionEngine."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import logging

import torch

import comfy.model_management as model_management
from omegaconf import OmegaConf

from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from ovi.utils.checkpoint_manager import (
    ensure_checkpoints,
    MissingDependencyError,
    DownloadError,
    OVI_MODEL_SPECS_MAP,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT_DIR = str((REPO_ROOT / "ckpts").resolve())

MODEL_VARIANT_CHOICES = [
    ("Ovi-11B-bf16.safetensors (720x720, 5s)", False, "bf16"),
    ("Ovi-11B-fp8.safetensors (720x720, 5s)", True, "fp8"),
    ("Ovi-960x960-5s.safetensors (960x960, 5s)", False, "960x960_5s"),
    ("Ovi-960x960-10s.safetensors (960x960, 10s)", False, "960x960_10s"),
    ("Ovi-960x960-10s-F16.gguf (960x960, 10s) [GGUF F16]", False, "960x960_10s_gguf_f16"),
    ("Ovi-960x960-10s-Q4_K.gguf (960x960, 10s) [GGUF Q4_K] ⭐", False, "960x960_10s_gguf_q4k"),
]
MODEL_VARIANT_LABELS = [choice[0] for choice in MODEL_VARIANT_CHOICES]
_MODEL_VARIANT_LOOKUP = {label: (fp8, key) for label, fp8, key in MODEL_VARIANT_CHOICES}
_LEGACY_TO_SPEC_VARIANT = {
    "bf16": "720x720_5s",
    "fp8": "720x720_5s_fp8",
}

# Global cache so multiple graphs can reuse the same heavy engine instance.
_ENGINE_CACHE: Dict[Tuple[int, bool, bool], OviFusionEngine] = {}


def _clear_engine_cache():
    for engine in list(_ENGINE_CACHE.values()):
        try:
            engine.unload()
        except Exception as exc:
            logging.warning('Failed to unload OVI engine: %s', exc)
    _ENGINE_CACHE.clear()


def _register_unload_hook():
    if getattr(model_management, '_ovi_unload_hook', False):
        return
    
    # Hook into unload_all_models
    original_unload = model_management.unload_all_models
    def wrapped_unload_all_models(*args, **kwargs):
        logging.info("[OVI] ComfyUI unload_all_models called - clearing OVI engine cache")
        _clear_engine_cache()
        return original_unload(*args, **kwargs)
    model_management.unload_all_models = wrapped_unload_all_models
    
    # Hook into free_memory (used by "Free All Memory" button)
    if hasattr(model_management, 'free_memory'):
        original_free = model_management.free_memory
        def wrapped_free_memory(*args, **kwargs):
            logging.info("[OVI] ComfyUI free_memory called - clearing OVI engine cache")
            _clear_engine_cache()
            return original_free(*args, **kwargs)
        model_management.free_memory = wrapped_free_memory
    
    # Hook into soft_empty_cache (alternative memory cleanup)
    if hasattr(model_management, 'soft_empty_cache'):
        original_soft = model_management.soft_empty_cache
        def wrapped_soft_empty_cache(*args, **kwargs):
            logging.info("[OVI] ComfyUI soft_empty_cache called - clearing OVI engine cache")
            _clear_engine_cache()
            return original_soft(*args, **kwargs)
        model_management.soft_empty_cache = wrapped_soft_empty_cache
    
    # Hook into cleanup_models (used during workflow cleanup)
    if hasattr(model_management, 'cleanup_models'):
        original_cleanup = model_management.cleanup_models
        def wrapped_cleanup_models(*args, **kwargs):
            logging.info("[OVI] ComfyUI cleanup_models called - clearing OVI engine cache")
            _clear_engine_cache()
            return original_cleanup(*args, **kwargs)
        model_management.cleanup_models = wrapped_cleanup_models
    
    model_management._ovi_unload_hook = True
    logging.info("[OVI] Unload hooks registered for: unload_all_models, free_memory, soft_empty_cache, cleanup_models")


_register_unload_hook()



def _safe_get_comfy_device():
    try:
        return model_management.get_torch_device()
    except Exception:
        return None


def _coerce_cuda_index(device) -> int | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        if device.type == "cuda":
            if device.index is not None:
                return device.index
            try:
                return torch.cuda.current_device()
            except Exception:
                return 0
        return None
    if isinstance(device, str):
        if device.startswith("cuda"):
            parts = device.split(":", 1)
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    if isinstance(device, int):
        return device
    return None


def _preferred_device_index() -> int | None:
    device = _safe_get_comfy_device()
    index = _coerce_cuda_index(device)
    if index is not None:
        return index
    if torch.cuda.is_available():
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0
    return None


def _default_device_choice() -> Tuple[str, int] | None:
    device = _safe_get_comfy_device()
    index = _coerce_cuda_index(device)
    if index is None:
        return None
    try:
        label = model_management.get_torch_device_name(device)
    except Exception:
        label = f"cuda:{index}"
    return (f"Auto (Comfy: {label})", index)


def _format_device_label(idx: int) -> str:
    try:
        label = model_management.get_torch_device_name(torch.device("cuda", idx))
    except Exception:
        try:
            label = torch.cuda.get_device_name(idx)
        except Exception:
            label = f"cuda:{idx}"
    return f"{idx}: {label}"


def _build_config(cpu_offload: bool, fp8: bool) -> OmegaConf:
    """Clone DEFAULT_CONFIG without mutating the module-level object."""
    base = OmegaConf.to_container(DEFAULT_CONFIG, resolve=True)
    config = OmegaConf.create(base)
    config.ckpt_dir = DEFAULT_CKPT_DIR
    config.cpu_offload = bool(cpu_offload)
    config.fp8 = bool(fp8)
    return config


class OviEngineLoader:
    CACHEABLE = False
    _DEVICE_LABEL_TO_INDEX: Dict[str, int] = {}

    @classmethod
    def INPUT_TYPES(cls):
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_inputs = {}
        device_choices = cls._build_device_choices(gpu_count)
        if device_choices:
            device_inputs["device"] = (device_choices, {"default": device_choices[0]})
        return {
            "required": {
                "model_precision": (MODEL_VARIANT_LABELS, {"default": MODEL_VARIANT_LABELS[0]}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                **device_inputs,
            }
        }

    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = "Ovi"

    def load(self, model_precision: str, cpu_offload: bool, device=None):
        if model_precision not in _MODEL_VARIANT_LOOKUP:
            raise ValueError(f"Unknown model precision selection '{model_precision}'.")
        fp8, variant_key = _MODEL_VARIANT_LOOKUP[model_precision]

        target_device = self._resolve_device(device)

        config = _build_config(cpu_offload, fp8)

        # IMPORTANT: Tell config which specific model was selected.
        spec_variant_key = _LEGACY_TO_SPEC_VARIANT.get(variant_key, variant_key)
        if spec_variant_key not in OVI_MODEL_SPECS_MAP:
            logging.warning(
                "Selected OVI variant '%s' is not registered in OVI_MODEL_SPECS_MAP; "
                "falling back to automatic model discovery.",
                spec_variant_key,
            )
            config.selected_model_variant = None
        else:
            config.selected_model_variant = spec_variant_key
        
        try:
            ensure_checkpoints(config.ckpt_dir, download=True, variants=(variant_key,))
        except MissingDependencyError as exc:
            raise RuntimeError(
                "huggingface_hub package is required for initial downloads."
            ) from exc
        except DownloadError as exc:
            raise RuntimeError(str(exc)) from exc

        # Cache key now includes variant to avoid loading wrong model
        cache_key = (target_device, config.cpu_offload, fp8, variant_key)

        engine = _ENGINE_CACHE.get(cache_key)
        if engine is None or getattr(engine, 'model', None) is None:
            engine = OviFusionEngine(config=config, device=target_device)
            _ENGINE_CACHE[cache_key] = engine
            available = engine.available_attention_backends()
            logging.info(
                'OVI engine attention backends: %s (current: %s)',
                ', '.join(available),
                engine.get_attention_backend(resolved=True),
            )
        else:
            engine.config = config

        return (engine,)

    @classmethod
    def _build_device_choices(cls, gpu_count: int) -> list[str]:
        cls._DEVICE_LABEL_TO_INDEX = {}
        choices: list[str] = []
        auto_choice = _default_device_choice()
        if auto_choice is not None:
            label, index = auto_choice
            choices.append(label)
            cls._DEVICE_LABEL_TO_INDEX[label] = index
        if gpu_count <= 0:
            return choices
        for idx in range(gpu_count):
            label = _format_device_label(idx)
            if label in cls._DEVICE_LABEL_TO_INDEX:
                continue
            cls._DEVICE_LABEL_TO_INDEX[label] = idx
            choices.append(label)
        return choices

    @classmethod
    def _resolve_device(cls, device) -> int:
        if isinstance(device, str):
            mapped = cls._DEVICE_LABEL_TO_INDEX.get(device)
            if mapped is not None:
                return mapped
            try:
                return int(device.split(":")[0].strip())
            except (ValueError, IndexError):
                pass
        elif isinstance(device, (int, float)):
            return int(device)

        preferred = _preferred_device_index()
        if preferred is not None:
            return preferred

        raise RuntimeError(
            "No CUDA/HIP device detected. Please ensure your GPU is exposed to PyTorch/ComfyUI."
        )
