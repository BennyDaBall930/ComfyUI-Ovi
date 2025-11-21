#!/usr/bin/env python3
"""
Quantize OVI Models to GGUF Q4_K_M Format

This script automates the process of converting OVI fusion models and text encoders
from safetensors/pth format to GGUF Q4_K_M quantized format.

Workflow:
1. Convert BF16/FP16 model to F16 GGUF (using ComfyUI-GGUF)
2. Quantize F16 GGUF to Q4_K_M (using llama.cpp quantize tool)
3. Move final quantized model to appropriate directory

Requirements:
- ComfyUI-GGUF installed in custom_nodes/
- llama.cpp with quantize tool compiled
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

# Add ComfyUI-GGUF to path
SCRIPT_DIR = Path(__file__).parent
COMFYUI_ROOT = SCRIPT_DIR.parents[1]
GGUF_PLUGIN_DIR = COMFYUI_ROOT / "custom_nodes" / "ComfyUI-GGUF"
GGUF_CONVERT_SCRIPT = GGUF_PLUGIN_DIR / "tools" / "convert.py"

if str(GGUF_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(GGUF_PLUGIN_DIR))

# Output directories
DIFFUSION_MODELS_DIR = COMFYUI_ROOT / "models" / "diffusion_models"
TEXT_ENCODERS_DIR = COMFYUI_ROOT / "models" / "text_encoders"


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_comfyui_gguf():
    """Check if ComfyUI-GGUF is installed."""
    if not GGUF_CONVERT_SCRIPT.exists():
        print("❌ ERROR: ComfyUI-GGUF not found!")
        print(f"   Expected at: {GGUF_PLUGIN_DIR}")
        print("\nPlease install ComfyUI-GGUF:")
        print("  cd custom_nodes")
        print("  git clone https://github.com/city96/ComfyUI-GGUF.git")
        print("  cd ComfyUI-GGUF")
        print("  pip install -r requirements.txt")
        return False
    print(f"✅ ComfyUI-GGUF found at: {GGUF_PLUGIN_DIR}")
    return True


def find_quantize_tool():
    """Find llama.cpp quantize tool."""
    # Common locations
    possible_paths = [
        Path("llama.cpp") / "build" / "bin" / "quantize.exe",
        Path("llama.cpp") / "build" / "bin" / "quantize",
        Path("llama.cpp") / "quantize.exe",
        Path("llama.cpp") / "quantize",
        Path.home() / "llama.cpp" / "build" / "bin" / "quantize.exe",
        Path.home() / "llama.cpp" / "build" / "bin" / "quantize",
        Path.home() / "llama.cpp" / "quantize.exe",
        Path.home() / "llama.cpp" / "quantize",
        COMFYUI_ROOT / "llama.cpp" / "build" / "bin" / "quantize.exe",
        COMFYUI_ROOT / "llama.cpp" / "build" / "bin" / "quantize",
        COMFYUI_ROOT / "llama.cpp" / "quantize.exe",
        COMFYUI_ROOT / "llama.cpp" / "quantize",
    ]
    
    # Check system PATH
    path_quantize = shutil.which("quantize")
    if path_quantize:
        print(f"✅ Found quantize tool in PATH: {path_quantize}")
        return Path(path_quantize)
    
    # Check common locations
    for path in possible_paths:
        if path.exists():
            print(f"✅ Found quantize tool at: {path}")
            return path
    
    print("❌ WARNING: llama.cpp quantize tool not found!")
    print("\nTo install llama.cpp and compile quantize tool:")
    print("  git clone https://github.com/ggerganov/llama.cpp.git")
    print("  cd llama.cpp")
    print("  cmake -B build")
    print("  cmake --build build --config Release")
    print("\nAlternatively, provide path with --quantize-tool argument")
    return None


def convert_to_f16_gguf(input_path, output_path=None):
    """Convert model to F16 GGUF format using ComfyUI-GGUF."""
    print_header("Step 1: Converting to F16 GGUF")
    
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ ERROR: Input file not found: {input_path}")
        return None
    
    print(f"Input:  {input_path}")
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}-F16.gguf"
    else:
        output_path = Path(output_path)
    
    print(f"Output: {output_path}")
    
    # Run conversion
    try:
        cmd = [
            sys.executable,
            str(GGUF_CONVERT_SCRIPT),
            "--src", str(input_path),
            "--dst", str(output_path)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024**3)
            print(f"\n✅ F16 GGUF created: {output_path.name} ({file_size:.2f} GB)")
            return output_path
        else:
            print(f"❌ ERROR: F16 GGUF file not created!")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Conversion failed: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def quantize_to_q4_k_m(f16_gguf_path, output_path=None, quantize_tool=None):
    """Quantize F16 GGUF to Q4_K_M using llama.cpp."""
    print_header("Step 2: Quantizing to Q4_K_M")
    
    if quantize_tool is None:
        quantize_tool = find_quantize_tool()
    
    if quantize_tool is None or not Path(quantize_tool).exists():
        print("❌ ERROR: Cannot proceed without quantize tool!")
        print(f"   F16 GGUF file saved at: {f16_gguf_path}")
        print("   You can manually quantize it later after installing llama.cpp")
        return None
    
    f16_gguf_path = Path(f16_gguf_path)
    if not f16_gguf_path.exists():
        print(f"❌ ERROR: F16 GGUF not found: {f16_gguf_path}")
        return None
    
    if output_path is None:
        output_path = f16_gguf_path.parent / f"{f16_gguf_path.stem.replace('-F16', '')}-Q4_K_M.gguf"
    else:
        output_path = Path(output_path)
    
    print(f"Input:  {f16_gguf_path}")
    print(f"Output: {output_path}")
    
    # Run quantization
    try:
        cmd = [
            str(quantize_tool),
            str(f16_gguf_path),
            str(output_path),
            "Q4_K_M"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024**3)
            print(f"\n✅ Q4_K_M GGUF created: {output_path.name} ({file_size:.2f} GB)")
            return output_path
        else:
            print(f"❌ ERROR: Q4_K_M GGUF file not created!")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Quantization failed: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def move_to_models_dir(gguf_path, model_type="diffusion"):
    """Move quantized model to appropriate models directory."""
    print_header("Step 3: Moving to Models Directory")
    
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        print(f"❌ ERROR: File not found: {gguf_path}")
        return False
    
    if model_type == "diffusion":
        target_dir = DIFFUSION_MODELS_DIR
    elif model_type == "text_encoder":
        target_dir = TEXT_ENCODERS_DIR
    else:
        print(f"❌ ERROR: Unknown model type: {model_type}")
        return False
    
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / gguf_path.name
    
    print(f"Source: {gguf_path}")
    print(f"Target: {target_path}")
    
    try:
        if target_path.exists():
            response = input(f"\n⚠️  File already exists at target location. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Skipping move. File remains at:")
                print(f"  {gguf_path}")
                return False
            target_path.unlink()
        
        shutil.move(str(gguf_path), str(target_path))
        print(f"\n✅ Model moved to: {target_path}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to move file: {e}")
        print(f"   File remains at: {gguf_path}")
        return False


def cleanup_intermediate(f16_path, keep_f16=False):
    """Clean up intermediate F16 GGUF file."""
    if keep_f16:
        print(f"\nKeeping intermediate F16 GGUF: {f16_path}")
        return
    
    try:
        f16_path = Path(f16_path)
        if f16_path.exists():
            f16_path.unlink()
            print(f"\n🗑️  Cleaned up intermediate file: {f16_path.name}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to delete intermediate file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize OVI models to GGUF Q4_K_M format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize fusion model
  python quantize_ovi_model.py Ovi-960x960-10s.safetensors
  
  # Quantize text encoder
  python quantize_ovi_model.py models_t5_umt5-xxl-enc-bf16.pth --type text_encoder
  
  # Custom output name
  python quantize_ovi_model.py Ovi-11B-bf16.safetensors --output Ovi-11B-Q4_K_M.gguf
  
  # Keep intermediate F16 file
  python quantize_ovi_model.py model.safetensors --keep-f16
  
  # Custom quantize tool location
  python quantize_ovi_model.py model.safetensors --quantize-tool /path/to/quantize
        """
    )
    
    parser.add_argument("input", type=str, help="Input model file (safetensors, pth, pt)")
    parser.add_argument("--output", "-o", type=str, help="Output GGUF filename")
    parser.add_argument("--type", "-t", choices=["diffusion", "text_encoder"], 
                       default="diffusion", help="Model type (default: diffusion)")
    parser.add_argument("--keep-f16", action="store_true", 
                       help="Keep intermediate F16 GGUF file")
    parser.add_argument("--no-move", action="store_true",
                       help="Don't move final file to models directory")
    parser.add_argument("--quantize-tool", type=str,
                       help="Path to llama.cpp quantize tool")
    
    args = parser.parse_args()
    
    print_header("OVI Model Quantization to Q4_K_M")
    
    # Check prerequisites
    if not check_comfyui_gguf():
        return 1
    
    quantize_tool = None
    if args.quantize_tool:
        quantize_tool = Path(args.quantize_tool)
        if not quantize_tool.exists():
            print(f"❌ ERROR: Quantize tool not found at: {quantize_tool}")
            return 1
    else:
        quantize_tool = find_quantize_tool()
    
    # Step 1: Convert to F16 GGUF
    input_path = Path(args.input)
    f16_gguf = convert_to_f16_gguf(input_path)
    if f16_gguf is None:
        return 1
    
    # Step 2: Quantize to Q4_K_M
    output_name = args.output if args.output else None
    q4_gguf = quantize_to_q4_k_m(f16_gguf, output_name, quantize_tool)
    if q4_gguf is None:
        return 1
    
    # Step 3: Move to models directory
    if not args.no_move:
        move_to_models_dir(q4_gguf, args.type)
    
    # Cleanup intermediate files
    cleanup_intermediate(f16_gguf, args.keep_f16)
    
    print_header("Quantization Complete!")
    print(f"✅ Final Q4_K_M model: {q4_gguf.name}")
    
    if not args.no_move:
        target_dir = DIFFUSION_MODELS_DIR if args.type == "diffusion" else TEXT_ENCODERS_DIR
        print(f"✅ Available in ComfyUI at: {target_dir / q4_gguf.name}")
    
    print("\nMemory savings: ~65-70% compared to BF16!")
    print("You can now use this model in your OVI workflows.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
