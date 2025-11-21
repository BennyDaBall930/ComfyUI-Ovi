import logging
import os
import time
import traceback
from pathlib import Path

import comfy.model_management as model_management
import folder_paths
import torch
from comfy.utils import ProgressBar
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from tqdm import tqdm

from ovi.utils.checkpoint_manager import (
    OVI_MODEL_SOURCE_NAME,
    OVI_MODEL_TARGET_NAME,
    OVI_MODEL_FP8_SOURCE_NAME,
    OVI_MODEL_FP8_TARGET_NAME,
    OVI_MODEL_SPECS_MAP,
)
from ovi.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ovi.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ovi.utils.model_loading_utils import (
    init_fusion_score_model_ovi,
    init_mmaudio_vae,
    init_text_model,
    init_wan_vae_2_2,
    load_fusion_checkpoint,
)
from ovi.utils.processing_utils import clean_text, preprocess_image_tensor, snap_hw_to_multiple_of_32, scale_hw_to_area_divisible
from ovi.modules.attention import (
    available_attention_backends as attention_available_backends,
    set_attention_backend as attention_set_backend,
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


_ENGINE_DEBUG = _env_flag("OVI_ENGINE_DEBUG") or _env_flag("OVI_VAE_DEBUG")


def _engine_debug(message: str):
    if _ENGINE_DEBUG:
        logging.warning(f"[OVI][ENGINE][DEBUG] {message}")


def _device_index_from_any(device) -> int | None:
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return None
    if isinstance(device, str):
        if device.startswith("cuda:"):
            try:
                return int(device.split(":", 1)[1])
            except (ValueError, IndexError):
                return 0
    if isinstance(device, int):
        return device
    return None


def _engine_gpu_mem(tag: str = "", device=None):
    if not (_ENGINE_DEBUG and torch.cuda.is_available()):
        return
    try:
        idx = _device_index_from_any(device)
        if idx is None:
            idx = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        free, total = torch.cuda.mem_get_info(idx)
        logging.warning(
            "[OVI][ENGINE][DEBUG] GPU mem %s -> alloc=%.2f GB, reserved=%.2f GB, free=%.2f GB, total=%.2f GB",
            tag,
            allocated,
            reserved,
            free / 1e9,
            total / 1e9,
        )
    except Exception:
        pass

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = OmegaConf.load(PACKAGE_ROOT / 'configs' / 'inference' / 'inference_fusion.yaml')
try:
    _DIFFUSION_MODEL_DIRS = [Path(p) for p in folder_paths.get_folder_paths('diffusion_models')]
except Exception:
    _DIFFUSION_MODEL_DIRS = []

# Build candidate lists for all model variants
OVI_MODEL_CANDIDATES_BF16 = [Path(p) / OVI_MODEL_TARGET_NAME for p in _DIFFUSION_MODEL_DIRS]
OVI_MODEL_CANDIDATES_FP8 = [Path(p) / OVI_MODEL_FP8_TARGET_NAME for p in _DIFFUSION_MODEL_DIRS]

# Add new 960x960 variants
OVI_MODEL_CANDIDATES_ALL = []
for diffusion_dir in _DIFFUSION_MODEL_DIRS:
    for variant_key, specs in OVI_MODEL_SPECS_MAP.items():
        candidate = Path(diffusion_dir) / specs["target_name"]
        if candidate not in OVI_MODEL_CANDIDATES_ALL:
            OVI_MODEL_CANDIDATES_ALL.append(candidate)

_AUTO_KEEP_VIDEO_MARGIN_BYTES = 4 * 1024 ** 3  # ~4 GB headroom
_AUTO_KEEP_AUDIO_MARGIN_BYTES = 1 * 1024 ** 3  # ~1 GB headroom


def _module_size_bytes(module) -> int:
    total = 0
    if module is None:
        return total
    seen = set()

    def _accumulate(mod):
        nonlocal total
        if mod is None or id(mod) in seen:
            return
        seen.add(id(mod))
        if isinstance(mod, torch.nn.Module):
            for param in mod.parameters(recurse=False):
                total += param.numel() * param.element_size()
            for buffer in mod.buffers(recurse=False):
                total += buffer.numel() * buffer.element_size()
            for child in mod.children():
                _accumulate(child)

    if isinstance(module, torch.nn.Module):
        _accumulate(module)
    elif hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        _accumulate(module.model)

    return total


def _gpu_memory_info(device_index: int) -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    try:
        torch.cuda.device(device_index)
    except Exception:
        pass
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device_index)
        return int(free_mem), int(total_mem)
    except Exception:
        try:
            props = torch.cuda.get_device_properties(device_index)
            return None, int(getattr(props, "total_memory", 0))
        except Exception:
            return None, None


class OviFusionEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        # Load fusion model
        self.device = device
        self.target_dtype = target_dtype
        self.config = config
        meta_init = True
        self.cpu_offload = False  # HARDCODED: All-GPU mode for 128GB unified VRAM
        self.fp8 = bool(config.get("fp8", False))
        logging.info("[OVI] ALL-GPU MODE: All models will remain on GPU permanently (128GB unified VRAM)")
        if self.fp8:
            logging.info("FP8 quantized fusion model requested.")

        model, video_config, audio_config = init_fusion_score_model_ovi(rank=device, meta_init=meta_init)

        if not meta_init:
            if not self.fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
    
        # Load VAEs
        vae_path = Path(config.ckpt_dir) / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"
        if vae_path.exists():
            vae_model_video = init_wan_vae_2_2(config.ckpt_dir, rank=device)
            vae_model_video.model.requires_grad_(False).eval()
            # Only cast to bfloat16 if the VAE itself thinks it should be bfloat16 (or defaults to it)
            # If OVI_VAE_FORCE_DTYPE set it to float16/float32, respect that.
            vae_dtype = getattr(vae_model_video, "dtype", torch.bfloat16)
            if vae_dtype is not None:
                vae_model_video.model = vae_model_video.model.to(dtype=vae_dtype)
            else:
                vae_model_video.model = vae_model_video.model.bfloat16()
            self.vae_model_video = vae_model_video
            self._video_vae_size_bytes = _module_size_bytes(getattr(vae_model_video, "model", vae_model_video))
        else:
            self.vae_model_video = None
            self._video_vae_size_bytes = 0

        vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device)
        vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = vae_model_audio.bfloat16()
        self._audio_vae_size_bytes = _module_size_bytes(getattr(self.vae_model_audio, "model", self.vae_model_audio))

        # Load T5 text model
        text_model_path = Path(config.ckpt_dir) / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth"
        if text_model_path.exists():
            self.text_model = init_text_model(config.ckpt_dir, rank=device)
            if config.get("shard_text_model", False):
                raise NotImplementedError("Sharding text model is not implemented yet.")
            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)
        else:
            self.text_model = None

        # Find fusion ckpt - use specific model selected by user if available
        checkpoint_path = None
        selected_variant = config.get("selected_model_variant", None)
        
        if selected_variant and selected_variant in OVI_MODEL_SPECS_MAP:
            # User selected a specific model variant - try to load ONLY that one
            target_name = OVI_MODEL_SPECS_MAP[selected_variant]["target_name"]
            for diffusion_dir in _DIFFUSION_MODEL_DIRS:
                candidate = Path(diffusion_dir) / target_name
                if candidate.exists():
                    checkpoint_path = candidate
                    logging.info(f"[OVI] Loading selected model variant '{selected_variant}': {checkpoint_path.name}")
                    break
            
            if checkpoint_path is None:
                # Try ckpt_dir fallback
                source_name = OVI_MODEL_SPECS_MAP[selected_variant]["source_name"]
                fallback_path = Path(config.ckpt_dir) / 'Ovi' / source_name
                if fallback_path.exists():
                    checkpoint_path = fallback_path
                    logging.info(f"[OVI] Found selected model '{selected_variant}' in fallback: {checkpoint_path.name}")
        else:
            # No specific selection - search all variants (old behavior)
            for candidate in OVI_MODEL_CANDIDATES_ALL:
                if candidate.exists():
                    checkpoint_path = candidate
                    logging.info(f"[OVI] Found fusion model: {checkpoint_path.name}")
                    break
            
            # Fallback to ckpt_dir/Ovi if not found
            if checkpoint_path is None:
                fallback_source = OVI_MODEL_FP8_SOURCE_NAME if self.fp8 else OVI_MODEL_SOURCE_NAME
                fallback_path = Path(config.ckpt_dir) / 'Ovi' / fallback_source
                if fallback_path.exists():
                    checkpoint_path = fallback_path
                    logging.info(f"[OVI] Found fusion model in fallback location: {checkpoint_path.name}")

        if checkpoint_path is None:
            raise RuntimeError(
                'No fusion checkpoint found. Available models: '
                '720x720_5s (Ovi-11B-bf16.safetensors), '
                '720x720_5s_fp8 (Ovi-11B-fp8.safetensors), '
                '960x960_5s (Ovi-960x960-5s.safetensors), '
                '960x960_10s (Ovi-960x960-10s.safetensors). '
                f'Selected variant was: {selected_variant}'
            )

        load_fusion_checkpoint(model, checkpoint_path=str(checkpoint_path), from_meta=meta_init)

        if meta_init:
            if not self.fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()
        self.model = model
        self._requested_attention_backend = 'auto'
        try:
            self._check_cancel()
            self._resolved_attention_backend = attention_set_backend('auto')
        except RuntimeError as exc:
            available = ', '.join(attention_available_backends(include_auto=False))
            raise RuntimeError(
                f"Failed to initialise attention backend (requested 'auto'). Available backends: {available or 'none'}"
            ) from exc

        ## Load t2i as part of pipeline
        if hasattr(self, 'image_model'):
            self.image_model = None
        if hasattr(self, 'image_model'):
            self.image_model = None
        # Determine model variant from checkpoint filename to set latent lengths
        checkpoint_name = checkpoint_path.name if checkpoint_path else "unknown"
        model_variant = None
        
        # Match checkpoint to model variant
        for variant_key, specs in OVI_MODEL_SPECS_MAP.items():
            if checkpoint_name == specs["target_name"] or checkpoint_name == specs["source_name"]:
                model_variant = variant_key
                break
        
        # Fallback: default to 720x720_5s
        if model_variant is None:
            logging.warning(f"[OVI] Unknown model checkpoint '{checkpoint_name}', defaulting to 720x720_5s specs")
            model_variant = "720x720_5s"
        
        model_specs = OVI_MODEL_SPECS_MAP[model_variant]
        
        # Set attributes from model specs (supports 5s and 10s models!)
        self.audio_latent_channel = audio_config.get("in_dim")
        self.video_latent_channel = video_config.get("in_dim")
        self.audio_latent_length = model_specs["audio_latent_length"]
        self.video_latent_length = model_specs["video_latent_length"]
        self.video_area = model_specs["video_area"]
        self.model_variant = model_variant
        
        logging.info(
            f"[OVI] Model variant: {model_variant} ({model_specs['duration_seconds']}s, "
            f"{model_specs['video_area']}px² area, "
            f"{self.video_latent_length} video frames, "
            f"{self.audio_latent_length} audio tokens)"
        )

        # Track external overrides so they can be restored after reloads.
        self._override_video_vae = getattr(self, "_override_video_vae", None)
        self._override_text_model = getattr(self, "_override_text_model", None)

        logging.info(f"OVI Fusion Engine initialized, cpu_offload={self.cpu_offload}. GPU VRAM allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, reserved: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

    def ensure_loaded(self):
        """Reload weights in-place if they were released via unload()."""
        model_missing = getattr(self, "model", None) is None
        audio_vae_missing = getattr(self, "vae_model_audio", None) is None
        video_vae_missing = getattr(self, "vae_model_video", None) is None
        text_model_missing = getattr(self, "text_model", None) is None

        # If only missing overridden modules, restore from cached overrides.
        if video_vae_missing and self._override_video_vae is not None:
            self.override_models(video_vae=self._override_video_vae)
            video_vae_missing = False
        if text_model_missing and self._override_text_model is not None:
            self.override_models(text_model=self._override_text_model)
            text_model_missing = False

        if not (model_missing or audio_vae_missing or video_vae_missing or text_model_missing):
            return

        overrides = (self._override_video_vae, self._override_text_model)
        logging.info(
            "Reinitialising OVI Fusion Engine after unload for device %s (fp8=%s, cpu_offload=%s).",
            self.device,
            getattr(self, "fp8", False),
            getattr(self, "cpu_offload", False),
        )
        self.__class__.__init__(self, config=self.config, device=self.device, target_dtype=self.target_dtype)
        if overrides[0] is not None or overrides[1] is not None:
            self.override_models(video_vae=overrides[0], text_model=overrides[1])

    def set_attention_backend(self, backend: str) -> str:
        try:
            resolved = attention_set_backend(backend)
        except (RuntimeError, ValueError) as exc:
            available = ', '.join(attention_available_backends())
            raise RuntimeError(
                f"Failed to select attention backend '{backend}': {exc}. Available backends: {available or 'none'}"
            ) from exc
        self._requested_attention_backend = backend
        self._resolved_attention_backend = resolved
        logging.info('OVI attention backend set to %s (requested %s)', resolved, backend)
        return resolved

    def _device_index(self) -> int:
        if isinstance(self.device, int):
            return self.device
        if isinstance(self.device, str):
            if self.device.startswith("cuda:"):
                try:
                    return int(self.device.split(":", 1)[1])
                except (ValueError, IndexError):
                    return 0
            try:
                return int(self.device)
            except ValueError:
                return 0
        return 0

    def _should_keep_module_on_device(self, module_bytes: int, margin_bytes: int) -> bool:
        if not (self.cpu_offload and module_bytes > 0 and torch.cuda.is_available()):
            return False
        free_mem, total_mem = _gpu_memory_info(self._device_index())
        if free_mem is None:
            return False
        required = module_bytes + margin_bytes
        if free_mem >= required:
            logging.debug(
                "OVI auto-keep enabled (module %.2f GB, free %.2f GB, margin %.2f GB)",
                module_bytes / 1e9,
                free_mem / 1e9,
                margin_bytes / 1e9,
            )
            return True
        return False

    def get_config(self):
        return self.config

    def resolve_ckpt_path(self, path: str):
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        base = Path(self.config.ckpt_dir)
        return (base / candidate).resolve()

    def override_models(self, video_vae=None, text_model=None):
        if video_vae is not None:
            self.vae_model_video = video_vae
            self._override_video_vae = video_vae
            self._set_video_vae_device(self.device)
            if self.cpu_offload:
                self._set_video_vae_device("cpu")
        if text_model is not None:
            model_obj = getattr(text_model, "model", text_model)
            try:
                model_obj = model_obj.to(device=self.device)
            except Exception:
                pass
            if hasattr(text_model, "model"):
                text_model.model = model_obj
            self.text_model = text_model
            self._override_text_model = text_model
            if self.cpu_offload:
                try:
                    self.offload_to_cpu(text_model.model)
                except Exception:
                    pass

    def _check_cancel(self):
        model_management.throw_exception_if_processing_interrupted()

    def _set_video_vae_device(self, device: str):
        """
        ALL-GPU MODE: Always keep Video VAE on GPU.
        If CPU device is requested, force GPU instead.
        """
        # Force GPU in all-GPU mode
        # We IGNORE OVI_VAE_DECODE_ON_CPU because the user has 107GB VRAM and only 32GB RAM.
        if device == "cpu":
            logging.warning("[OVI] ALL-GPU MODE: Attempted to move Video VAE to CPU - forcing GPU instead")
            device = self.device
        
        video_vae = self._require_video_vae()
        target_dtype = getattr(video_vae, "dtype", self.target_dtype)
        
        if hasattr(video_vae, "model"):
            video_vae.model = video_vae.model.to(device=device, dtype=target_dtype).eval()
        
        try:
            video_vae.device = device
        except Exception:
            pass
        
        # Sync scale tensors to GPU
        scale = getattr(video_vae, "scale", None)
        if isinstance(scale, (list, tuple)):
            scale_dtype = getattr(video_vae, "dtype", target_dtype)
            def _move_scale_tensor(value):
                if isinstance(value, torch.Tensor):
                    return value.to(device=device, dtype=scale_dtype) if scale_dtype is not None else value.to(device=device)
                try:
                    kwargs = {"device": device}
                    if scale_dtype is not None:
                        kwargs["dtype"] = scale_dtype
                    return torch.as_tensor(value, **kwargs)
                except Exception:
                    return value
            moved = [_move_scale_tensor(value) for value in scale]
            video_vae.scale = type(scale)(moved)
        
        return video_vae

    def _require_video_vae(self):
        if self.vae_model_video is None:
            raise RuntimeError('Wan video VAE is not loaded. Please add OviWanComponentLoader to your workflow.')
        return self.vae_model_video

    def _require_text_model(self):
        if self.text_model is None:
            raise RuntimeError('Wan text encoder is not loaded. Please add OviWanComponentLoader to your workflow.')
        return self.text_model

    def unload(self):
        import gc
        
        logging.info("[OVI] Starting comprehensive model unload...")
        
        modules = [
            getattr(self, 'model', None),
            getattr(self, 'vae_model_video', None),
            getattr(self, 'vae_model_audio', None),
            getattr(self, 'text_model', None),
            getattr(self, 'image_model', None),
        ]

        # Move to CPU and explicitly delete
        for module in modules:
            if module is None:
                continue
            candidates = [module]
            if hasattr(module, 'model'):
                model_attr = getattr(module, 'model', None)
                if model_attr is not None:
                    candidates.append(model_attr)
            
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    # Only move to CPU if not already there
                    if hasattr(candidate, 'device'):
                        current_device = candidate.device
                        if isinstance(current_device, torch.device) and current_device.type == 'cuda':
                            candidate.to('cpu')
                    elif hasattr(candidate, 'parameters'):
                        # Check if module has parameters on CUDA
                        try:
                            first_param = next(candidate.parameters(), None)
                            if first_param is not None and first_param.device.type == 'cuda':
                                candidate.to('cpu')
                        except (StopIteration, AttributeError):
                            pass
                    
                    # Explicitly delete the candidate
                    del candidate
                except Exception as exc:
                    logging.warning("[OVI] Failed to move/delete module component: %s", exc)

        # Clear all model attributes
        if hasattr(self, 'model'):
            try:
                del self.model
            except:
                pass
            self.model = None
            
        if hasattr(self, 'vae_model_video'):
            try:
                del self.vae_model_video
            except:
                pass
            self.vae_model_video = None
            
        if hasattr(self, 'vae_model_audio'):
            try:
                del self.vae_model_audio
            except:
                pass
            self.vae_model_audio = None
            
        if hasattr(self, 'text_model'):
            try:
                del self.text_model
            except:
                pass
            self.text_model = None
            
        if hasattr(self, 'image_model'):
            try:
                del self.image_model
            except:
                pass
            self.image_model = None

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Try to release memory back to OS (CPython specific)
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):
                # Windows: Try to trim working set
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                logging.info("[OVI] Requested Windows to trim process working set")
        except Exception as exc:
            logging.debug("[OVI] Could not trim working set: %s", exc)
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                # Empty cache multiple times
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Print memory stats
                device_idx = self._device_index()
                allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                reserved = torch.cuda.memory_reserved(device_idx) / 1e9
                logging.info(
                    "[OVI] CUDA after unload: allocated=%.2f GB, reserved=%.2f GB", 
                    allocated, reserved
                )
            except Exception as exc:
                logging.warning("[OVI] CUDA cleanup failed: %s", exc)
        
        logging.info("[OVI] Model unload completed")


    def get_attention_backend(self, resolved: bool = False) -> str:
        if resolved:
            return getattr(self, '_resolved_attention_backend', self._requested_attention_backend)
        return self._requested_attention_backend

    @staticmethod
    def available_attention_backends(include_auto: bool = True):
        return attention_available_backends(include_auto=include_auto)

    @torch.inference_mode()
    def generate(self,
                    text_prompt, 
                    image_path=None,
                    video_frame_height_width=None,
                    seed=100,
                    solver_name="unipc",
                    sample_steps=50,
                    shift=5.0,
                    video_guidance_scale=5.0,
                    audio_guidance_scale=4.0,
                    slg_layer=9,
                    video_negative_prompt="",
                    audio_negative_prompt=""
                ):

        try:
            self.ensure_loaded()
            resolved_backend = attention_set_backend(self._requested_attention_backend)
            self._resolved_attention_backend = resolved_backend
        except RuntimeError as exc:
            available = ', '.join(attention_available_backends())
            raise RuntimeError(
                f"Failed to select attention backend '{self._requested_attention_backend}': {exc}. Available backends: {available or 'none'}"
            ) from exc

        params = {
            "Text Prompt": text_prompt,
            "Image Path": image_path if image_path else "None (T2V mode)",
            "Frame Height Width": video_frame_height_width,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Video Guidance Scale": video_guidance_scale,
            "Audio Guidance Scale": audio_guidance_scale,
            "Attention Backend": resolved_backend,
            "SLG Layer": slg_layer,
            "Video Negative Prompt": video_negative_prompt,
            "Audio Negative Prompt": audio_negative_prompt,
        }

        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n"
                    f"{pretty}\n"
                    "==========================================")
        try:
            scheduler_video, timesteps_video = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift
            )

            def _reset_scheduler(scheduler):
                if hasattr(scheduler, "set_begin_index"):
                    scheduler.set_begin_index(0)
                if hasattr(scheduler, "_begin_index"):
                    scheduler._begin_index = 0
                if hasattr(scheduler, "_step_index"):
                    scheduler._step_index = 0

            _reset_scheduler(scheduler_video)
            _reset_scheduler(scheduler_audio)

            # ROCm equality checks against device tensors can be flaky; keep CPU scalars.
            def _to_plain_timesteps(sequence):
                if isinstance(sequence, torch.Tensor):
                    seq = sequence.detach().cpu()
                    if seq.dtype.is_floating_point:
                        return [float(x) for x in seq.tolist()]
                    return [int(x) for x in seq.tolist()]
                result = []
                for value in sequence:
                    if isinstance(value, torch.Tensor):
                        scalar = value.item()
                    else:
                        scalar = value
                    if isinstance(scalar, float):
                        result.append(float(scalar))
                    else:
                        result.append(int(scalar))
                return result

            timesteps_video_list = _to_plain_timesteps(timesteps_video)
            timesteps_audio_list = _to_plain_timesteps(timesteps_audio)

            first_frame = None
            is_i2v = False

            if image_path is not None:
                first_frame = preprocess_image_tensor(image_path, self.device, self.target_dtype)
                is_i2v = first_frame is not None
            else:
                assert video_frame_height_width is not None, f"If mode=t2v or t2i2v, video_frame_height_width must be provided."
                video_h, video_w = video_frame_height_width
                video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area = 720 * 720)
                video_latent_h, video_latent_w = video_h // 16, video_w // 16
                image_model = getattr(self, 'image_model', None)
                if image_model is not None:
                    image_h, image_w = scale_hw_to_area_divisible(video_h, video_w, area = 1024 * 1024)
                    generated_frame = image_model(
                        clean_text(text_prompt),
                        height=image_h,
                        width=image_w,
                        guidance_scale=4.5,
                        generator=torch.Generator().manual_seed(seed),
                    ).images[0]
                    first_frame = preprocess_image_tensor(generated_frame, self.device, self.target_dtype)
                    is_i2v = first_frame is not None
                else:
                    print(f"Pure T2V mode: calculated video latent size: {video_latent_h} x {video_latent_w}")

            text_model = self._require_text_model()
            previous_device = getattr(text_model, "device", self.device)
            if self.cpu_offload:
                text_model.model = text_model.model.to(self.device)
                text_model.device = self.device
            text_embeddings = text_model([text_prompt, video_negative_prompt, audio_negative_prompt], text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]

            if self.cpu_offload:
                self.offload_to_cpu(text_model.model)
                text_model.device = previous_device

            # Split embeddings
            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_video_pos = text_embeddings[0] 

            text_embeddings_video_neg = text_embeddings[1]
            text_embeddings_audio_neg = text_embeddings[2]

            if is_i2v:              
                with torch.no_grad():
                    self._check_cancel()
                    # Force GPU for encoding (ignore OVI_VAE_DECODE_ON_CPU)
                    encode_device = self.device
                    
                    video_vae = self._set_video_vae_device(encode_device)
                    # Use the VAE's actual dtype, not hardcoded bfloat16
                    vae_dtype = getattr(video_vae, "dtype", torch.bfloat16)
                    first_frame_tensor = first_frame.to(device=encode_device, dtype=vae_dtype)
                    latents_images = video_vae.wrapped_encode(first_frame_tensor[:, :, None]).to(self.target_dtype).squeeze(0) # c 1 h w 
                    
                    if self.cpu_offload:
                        self._set_video_vae_device("cpu")
                latents_images = latents_images.to(self.target_dtype)
                video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]

            video_noise = torch.randn((self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # c, f, h, w
            audio_noise = torch.randn((self.audio_latent_length, self.audio_latent_channel), device=self.device, dtype=self.target_dtype, generator=torch.Generator(device=self.device).manual_seed(seed))  # 1, l c -> l, c
            
            # Calculate sequence lengths from actual latents
            max_seq_len_audio = audio_noise.shape[0]  # L dimension from latents_audios shape [1, L, D]
            _patch_size_h, _patch_size_w = self.model.video_model.patch_size[1], self.model.video_model.patch_size[2]
            max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w) # f * h * w from [1, c, f, h, w]
            
            # Sampling loop
            if self.cpu_offload:
                self.model = self.model.to(self.device)
            ui_progress = ProgressBar(len(timesteps_video_list)) if 'ProgressBar' in globals() and ProgressBar is not None else None
            pair_iterator = zip(timesteps_video_list, timesteps_audio_list)
            if 'tqdm' in globals() and tqdm is not None:
                pair_iterator = tqdm(pair_iterator, total=len(timesteps_video_list))
            with torch.amp.autocast('cuda', enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for step_index, (t_v, t_a) in enumerate(pair_iterator):
                    self._check_cancel()
                    timestep_dtype = torch.float32 if isinstance(t_v, float) else torch.int64
                    timestep_input = torch.full((1,), t_v, device=self.device, dtype=timestep_dtype)

                    if is_i2v:
                        video_noise[:, :1] = latents_images

                    # Positive (conditional) forward pass
                    pos_forward_args = {
                        'audio_context': [text_embeddings_audio_pos],
                        'vid_context': [text_embeddings_video_pos],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v
                    }

                    pred_vid_pos, pred_audio_pos = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **pos_forward_args
                    )
                    
                    # Negative (unconditional) forward pass  
                    neg_forward_args = {
                        'audio_context': [text_embeddings_audio_neg],
                        'vid_context': [text_embeddings_video_neg],
                        'vid_seq_len': max_seq_len_video,
                        'audio_seq_len': max_seq_len_audio,
                        'first_frame_is_clean': is_i2v,
                        'slg_layer': slg_layer
                    }
                    
                    pred_vid_neg, pred_audio_neg = self.model(
                        vid=[video_noise],
                        audio=[audio_noise],
                        t=timestep_input,
                        **neg_forward_args
                    )

                    # Apply classifier-free guidance
                    pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                    # Update noise using scheduler
                    video_noise = scheduler_video.step(
                        pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                    )[0].squeeze(0)

                    if ui_progress is not None:
                        ui_progress.update(1)

            if self.cpu_offload:
                self.offload_to_cpu(self.model)
            if ui_progress is not None and hasattr(ui_progress, "update_absolute"):
                ui_progress.update_absolute(0)

            if is_i2v:
                video_noise[:, :1] = latents_images

            video_latents = video_noise.detach()
            audio_latents = audio_noise.detach()

            # === AGGRESSIVE MEMORY CLEANUP FOR ALL-GPU MODE ===
            # Free all intermediate tensors AND fusion model to make room for VAE decode
            _engine_debug("Starting aggressive memory cleanup after diffusion")
            _engine_gpu_mem("before cleanup", self.device)
            
            # Delete all intermediate diffusion tensors
            try:
                del pred_vid_pos, pred_audio_pos, pred_vid_neg, pred_audio_neg
                del pred_video_guided, pred_audio_guided
                _engine_debug("Deleted prediction tensors")
            except:
                pass
            
            try:
                del text_embeddings, text_embeddings_audio_pos, text_embeddings_video_pos
                del text_embeddings_video_neg, text_embeddings_audio_neg
                _engine_debug("Deleted text embeddings")
            except:
                pass
            
            try:
                if 'first_frame' in locals() and first_frame is not None:
                    del first_frame
                if 'first_frame_tensor' in locals():
                    del first_frame_tensor
                if 'latents_images' in locals() and is_i2v:
                    # Keep latents_images if i2v, it's still needed
                    pass
                _engine_debug("Deleted image tensors")
            except:
                pass
            
            # CRITICAL: Unload fusion model AND text encoder to free ~30GB for VAE decode
            try:
                if hasattr(self, 'model') and self.model is not None:
                    _engine_debug("Unloading fusion model to free VRAM for VAE")
                    self.model = self.model.to('cpu')
                    del self.model
                    self.model = None
                    _engine_debug("Fusion model unloaded")
            except Exception as exc:
                logging.warning(f"[OVI] Failed to unload fusion model: {exc}")
            
            # Also unload text encoder (no longer needed after embeddings generated)
            try:
                if hasattr(self, 'text_model') and self.text_model is not None:
                    _engine_debug("Unloading text encoder to free additional VRAM")
                    if hasattr(self.text_model, 'model'):
                        self.text_model.model = self.text_model.model.to('cpu')
                    del self.text_model
                    self.text_model = None
                    _engine_debug("Text encoder unloaded")
            except Exception as exc:
                logging.warning(f"[OVI] Failed to unload text encoder: {exc}")
            
            # Aggressive CUDA cache cleanup with defragmentation
            # Key: empty_cache() multiple times to defragment
            _engine_debug("Starting aggressive CUDA defragmentation")
            for i in range(5):
                torch.cuda.empty_cache()
                if i < 4:
                    torch.cuda.synchronize()
            
            # IPC collect to release shared memory
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
            
            # Final defragmentation pass
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Reset memory stats to help with fragmentation tracking
            try:
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats(self.device)
            except Exception as exc:
                _engine_debug(f"reset_peak_memory_stats failed (not critical): {exc}")
            
            _engine_gpu_mem("after cleanup (ready for VAE)", self.device)
            logging.info("[OVI] Memory cleanup complete - fusion+text unloaded, memory defragmented, ready for VAE decode on GPU")
            # === END CLEANUP ===

            return video_latents, audio_latents


        except Exception as e:
            logging.error(traceback.format_exc())
            return None

    @torch.inference_mode()
    def decode_latents(
        self,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        to_cpu: bool = True,
    ):
        """
        ALL-GPU MODE: Decode video/audio latents entirely on GPU.
        No CPU offloading, no dtype fallbacks, maximum performance.
        """
        if video_latents is None and audio_latents is None:
            raise ValueError("At least one of video_latents or audio_latents must be provided.")

        # DON'T call ensure_loaded() - it would reload fusion model!
        # Only ensure VAEs are available (they should still be loaded)
        if audio_latents is not None and getattr(self, "vae_model_audio", None) is None:
            raise RuntimeError("Audio VAE not loaded")
        if video_latents is not None and getattr(self, "vae_model_video", None) is None:
            raise RuntimeError("Video VAE not loaded")
        
        _engine_debug("Starting VAE decode - ALL-GPU MODE (fusion model should be unloaded)")
        _engine_gpu_mem("pre-decode", self.device)

        decoded_video = None
        decoded_audio = None

        # Audio VAE decode (always on GPU, smaller model)
        if audio_latents is not None:
            if not isinstance(audio_latents, torch.Tensor):
                raise TypeError("audio_latents must be a torch.Tensor.")
            audio_tensor = audio_latents.to(self.target_dtype)
            if audio_tensor.dim() != 2:
                raise ValueError("audio_latents must have shape [length, channels].")
            if audio_tensor.device != self.device:
                audio_tensor = audio_tensor.to(self.device)
            
            self._check_cancel()
            # ALL-GPU: Audio VAE always on GPU
            self.vae_model_audio = self.vae_model_audio.to(self.device)
            audio_latents_for_vae = audio_tensor.unsqueeze(0).transpose(1, 2)  # 1, c, l
            decoded_audio = (
                self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                .squeeze()
                .to(torch.float32)
            )
            if to_cpu:
                decoded_audio = decoded_audio.cpu()
            _engine_debug("Audio VAE decode complete")

        # Video VAE decode (ALL-GPU mode)
        if video_latents is not None:
            if not isinstance(video_latents, torch.Tensor):
                raise TypeError("video_latents must be a torch.Tensor.")
            if video_latents.dim() != 4:
                raise ValueError("video_latents must have shape [channels, frames, height, width].")

            # ALL-GPU: Use target dtype, decode on GPU
            video_vae = self._require_video_vae()
            video_tensor = video_latents.to(self.target_dtype)
            
            # Force GPU for decoding (ignore OVI_VAE_DECODE_ON_CPU)
            decode_device = self.device

            if video_tensor.device != decode_device:
                video_tensor = video_tensor.to(decode_device)
            
            _engine_debug(f"Video VAE decode - dtype={self.target_dtype}, device={decode_device}")
            
            # Ensure Video VAE is on correct device
            video_vae = self._set_video_vae_device(decode_device)
            
            # Sync scale tensors to GPU if needed
            scale = getattr(video_vae, "scale", None)
            if isinstance(scale, (list, tuple)):
                scale_dtype = getattr(video_vae, "dtype", self.target_dtype)
                def _to_tensor(value):
                    if isinstance(value, torch.Tensor):
                        return value.to(device=self.device, dtype=scale_dtype)
                    try:
                        return torch.as_tensor(value, device=self.device, dtype=scale_dtype)
                    except Exception:
                        return value
                moved = [_to_tensor(value) for value in scale]
                video_vae.scale = type(scale)(moved)
            
            # Decode
            self._check_cancel()
            decoded_video = video_vae.decode_latents(
                video_tensor,
                device=decode_device,
                normalize=True,
                return_cpu=to_cpu,
                dtype=torch.float32,
                pbar=ProgressBar(video_tensor.shape[1]) if to_cpu else None,
            )
            _engine_debug("Video VAE decode complete")

        _engine_gpu_mem("post-decode", self.device)
        return decoded_video, decoded_audio

    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name='unipc', device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps

        elif solver_name == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
            
        elif solver_name == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=shift
            )
            timesteps, sampling_steps = retrieve_timesteps(
                sample_scheduler,
                sampling_steps,
                device=device,
            )
        
        else:
            raise NotImplementedError("Unsupported solver.")
        
        return sample_scheduler, timesteps
