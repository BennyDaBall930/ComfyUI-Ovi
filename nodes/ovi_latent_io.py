import os
import torch
import folder_paths
import time

class SaveOviLatents:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_latents": ("OVI_VIDEO_LATENTS",),
                "filename_prefix": ("STRING", {"default": "ovi_latents"}),
            },
            "optional": {
                "audio_latents": ("OVI_AUDIO_LATENTS",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "OVI"

    def save(self, video_latents, filename_prefix="ovi_latents", audio_latents=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, video_latents[0].shape[1], video_latents[0].shape[2]
        )
        
        # Construct filename with counter
        file = f"{filename}_{counter:05}_.pt"
        file_path = os.path.join(full_output_folder, file)
        
        data = {
            "video_latents": video_latents,
            "audio_latents": audio_latents,
            "version": 1
        }
        
        torch.save(data, file_path)
        
        return {"ui": {"text": [f"Saved latents to {file}"]}}

class LoadOviLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": (sorted([f for f in os.listdir(folder_paths.get_output_directory()) if f.endswith(".pt")]), ),
            }
        }

    RETURN_TYPES = ("OVI_VIDEO_LATENTS", "OVI_AUDIO_LATENTS")
    RETURN_NAMES = ("video_latents", "audio_latents")
    FUNCTION = "load"
    CATEGORY = "OVI"

    def load(self, filename):
        path = os.path.join(folder_paths.get_output_directory(), filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Latent file not found: {path}")
            
        data = torch.load(path, map_location="cpu", weights_only=True)
        
        if "video_latents" not in data:
            raise ValueError("Invalid latent file: missing video_latents")
            
        return (data["video_latents"], data.get("audio_latents"))
