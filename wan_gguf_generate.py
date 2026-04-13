#!/usr/bin/env python3
"""
Batch animate images using Wan2.2 GGUF (two-pass HIGH+LOW) via ComfyUI API.
Best quality/speed balance for 4090 + 24GB RAM.

Usage:
    python wan_gguf_generate.py image.jpg --prompt "woman walks through neon arcade"
    python wan_gguf_generate.py ./images/ --prompt "..." --output ./clips
    python wan_gguf_generate.py ./images/ --prompt "..." --width 480 --height 832  (portrait)

Setup:
    1. Install ComfyUI-WanVideoWrapper (already have it from VBVR workflow)
    2. Download both GGUF models to ComfyUI/models/diffusion_models/:
       wan2.2_i2v_high_noise_14B_Q4_K_M.gguf  (~9.65GB)
       wan2.2_i2v_low_noise_14B_Q4_K_M.gguf   (~9.65GB)
       https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF
    3. Download VAE and T5 to their respective folders:
       ComfyUI/models/vae/           wan_2.1_vae.safetensors
       ComfyUI/models/text_encoders/ umt5_xxl_fp16.safetensors

How it works:
    Two-pass sampling — HIGH model handles initial structure (steps 0→split),
    LOW model refines details (steps split→end). Same architecture as the VBVR
    workflow but in GGUF format. Models run sequentially, each offloading before
    the next loads.

VRAM breakdown (Q4_K_M on 4090):
    Each model:  ~9.65GB (only one active at a time)
    T5:          ~10GB   (offloads after text encoding)
    VAE:         ~0.1GB
    Latents:     ~1-2GB
    Peak:        ~12GB during generation  (comfortable on 24GB)

RAM breakdown:
    T5 offloaded to CPU:  ~10GB
    OS + ComfyUI:         ~4GB
    Peak:                 ~14GB  (comfortable on 24GB RAM)
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8189")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_workflow(image_name, prompt, negative, width, height, frames,
                   steps, split_step, cfg, shift, seed, blocks_to_swap,
                   high_model, low_model, vae_name, t5_name):
    return {
        "1": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": high_model,
                "base_precision": "fp16",
                "quantization": "disabled",
                "load_device": "offload_device",
                "attention_mode": "sageattn",
                "rms_norm_function": "default",
            },
            "_meta": {"title": "HIGH Model"},
        },
        "2": {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": low_model,
                "base_precision": "fp16",
                "quantization": "disabled",
                "load_device": "offload_device",
                "attention_mode": "sageattn",
                "rms_norm_function": "default",
            },
            "_meta": {"title": "LOW Model"},
        },
        "3": {
            "class_type": "WanVideoBlockSwap",
            "inputs": {
                "blocks_to_swap": blocks_to_swap,
                "offload_img_emb": True,
                "offload_txt_emb": True,
                "use_non_blocking": True,
                "vace_blocks_to_swap": 0,
                "prefetch_blocks": 0,
                "block_swap_debug": False,
            },
            "_meta": {"title": "Block Swap"},
        },
        "4": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {
                "model": ["1", 0],
                "block_swap_args": ["3", 0],
            },
            "_meta": {"title": "Apply Block Swap (HIGH)"},
        },
        "5": {
            "class_type": "WanVideoSetBlockSwap",
            "inputs": {
                "model": ["2", 0],
                "block_swap_args": ["3", 0],
            },
            "_meta": {"title": "Apply Block Swap (LOW)"},
        },
        "6": {
            "class_type": "WanVideoVAELoader",
            "inputs": {
                "model_name": vae_name,
                "precision": "bf16",
                "use_cpu_cache": False,
                "verbose": False,
            },
            "_meta": {"title": "VAE"},
        },
        "7": {
            "class_type": "LoadWanVideoT5TextEncoder",
            "inputs": {
                "model_name": t5_name,
                "precision": "bf16",
                "load_device": "offload_device",
                "quantization": "disabled",
            },
            "_meta": {"title": "T5 Encoder"},
        },
        "8": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
            "_meta": {"title": "Input Image"},
        },
        "9": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["8", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
            "_meta": {"title": "Resize"},
        },
        "10": {
            "class_type": "WanVideoTextEncode",
            "inputs": {
                "t5": ["7", 0],
                "positive_prompt": prompt,
                "negative_prompt": negative,
                "force_offload": True,
                "use_disk_cache": False,
                "device": "gpu",
            },
            "_meta": {"title": "Text Encode"},
        },
        "11": {
            "class_type": "WanVideoImageToVideoEncode",
            "inputs": {
                "vae": ["6", 0],
                "start_image": ["9", 0],
                "width": width,
                "height": height,
                "num_frames": frames,
                "noise_aug_strength": 0,
                "start_latent_strength": 1,
                "end_latent_strength": 1,
                "force_offload": True,
                "fun_or_fl2v_model": False,
                "tiled_vae": False,
                "augment_empty_frames": 0,
            },
            "_meta": {"title": "Image to Video Encode"},
        },
        "12": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["4", 0],
                "image_embeds": ["11", 0],
                "text_embeds": ["10", 0],
                "steps": steps,
                "cfg": cfg,
                "shift": shift,
                "seed": seed,
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
                "denoise_strength": 1,
                "batched_cfg": False,
                "rope_function": "comfy",
                "end_step": split_step,
            },
            "_meta": {"title": "Sampler Pass 1 (HIGH)"},
        },
        "13": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "model": ["5", 0],
                "image_embeds": ["11", 0],
                "text_embeds": ["10", 0],
                "samples": ["12", 0],
                "steps": steps,
                "cfg": cfg,
                "shift": shift,
                "seed": seed,
                "force_offload": True,
                "scheduler": "unipc",
                "riflex_freq_index": 0,
                "denoise_strength": 1,
                "batched_cfg": False,
                "rope_function": "comfy",
                "start_step": split_step,
            },
            "_meta": {"title": "Sampler Pass 2 (LOW)"},
        },
        "14": {
            "class_type": "WanVideoDecode",
            "inputs": {
                "vae": ["6", 0],
                "samples": ["13", 0],
                "enable_vae_tiling": False,
                "tile_x": 272,
                "tile_y": 272,
                "tile_stride_x": 144,
                "tile_stride_y": 128,
                "normalization": "default",
            },
            "_meta": {"title": "Decode"},
        },
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["14", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "wan22_gguf",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
            "_meta": {"title": "Save Video"},
        },
    }


def api_get(endpoint):
    with urllib.request.urlopen(f"{COMFYUI_URL}{endpoint}") as resp:
        return json.loads(resp.read())


def api_post(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}{endpoint}", data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()}") from e


def upload_image(image_path):
    image_path = Path(image_path)
    boundary = uuid.uuid4().hex
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        image_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{COMFYUI_URL}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["name"]


def wait_for_completion(prompt_id, poll_interval=5, timeout=3600):
    print("  Generating", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        history = api_get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print(f" done ({int(time.time() - start)}s).")
                return entry
            if status.get("status_str") == "error":
                print()
                raise RuntimeError(f"Generation failed: {entry}")
        print(".", end="", flush=True)
        time.sleep(poll_interval)
    print()
    raise TimeoutError(f"Timed out after {timeout}s")


def download_outputs(history_entry, output_dir, image_stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for node_id, node_output in history_entry.get("outputs", {}).items():
        for video in node_output.get("videos", []) + node_output.get("gifs", []):
            filename = video["filename"]
            params = urllib.parse.urlencode({
                "filename": filename,
                "subfolder": video.get("subfolder", ""),
                "type": video.get("type", "output"),
            })
            out_path = output_dir / f"{image_stem}_{filename}"
            urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", out_path)
            print(f"  Saved: {out_path}")
            downloaded.append(out_path)

    return downloaded


def collect_images(path):
    p = Path(path)
    if p.is_dir():
        return [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        return [p]
    print(f"Error: {path} is not an image or image directory.")
    sys.exit(1)


def check_comfyui():
    try:
        api_get("/system_stats")
    except (urllib.error.URLError, ConnectionRefusedError):
        print(f"Error: Cannot reach ComfyUI at {COMFYUI_URL}")
        print("Start ComfyUI: python main.py --highvram --cuda-malloc --use-pytorch-cross-attention")
        sys.exit(1)


def main():
    global COMFYUI_URL
    parser = argparse.ArgumentParser(
        description="Animate images with Wan2.2 Q4_K_M GGUF via ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Image path or directory of images")
    parser.add_argument("--prompt", "-p", required=True, help="Describe the motion and scene")
    parser.add_argument("--negative", "-n", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", "-o", default="outputs")
    parser.add_argument("--width", type=int, default=832, help="Width (default: 832)")
    parser.add_argument("--height", type=int, default=480, help="Height (default: 480, use 832 for portrait)")
    parser.add_argument("--frames", type=int, default=81, help="Frames, must be 4n+1 (default: 81 ≈ 3.4s)")
    parser.add_argument("--steps", type=int, default=20, help="Total steps (default: 20)")
    parser.add_argument("--split-step", type=int, default=10,
                        help="Step where HIGH hands off to LOW (default: 10, half of steps)")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG scale (default: 6.0)")
    parser.add_argument("--shift", type=float, default=8.0, help="Shift (default: 8.0)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--blocks-to-swap", type=int, default=30,
                        help="Blocks to offload to CPU per model, 14B has 40 total (default: 30)")
    parser.add_argument("--high-model", default="wan2.2_i2v_high_noise_14B_Q4_K_M.gguf")
    parser.add_argument("--low-model", default="wan2.2_i2v_low_noise_14B_Q4_K_M.gguf")
    parser.add_argument("--vae", default="wan_2.1_vae.safetensors")
    parser.add_argument("--t5", default="umt5_xxl_fp16.safetensors")
    parser.add_argument("--comfyui-url", default=COMFYUI_URL)
    args = parser.parse_args()

    COMFYUI_URL = args.comfyui_url

    check_comfyui()

    images = collect_images(args.image)
    print(f"Found {len(images)} image(s) — Wan2.2 GGUF Q4_K_M ({args.width}x{args.height}, {args.frames} frames, {args.steps} steps, split at {args.split_step})")

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img.name}")

        seed = args.seed if args.seed >= 0 else int(uuid.uuid4().int % (2 ** 32))

        uploaded = upload_image(img)
        print(f"  Uploaded: {uploaded}")

        workflow = build_workflow(
            image_name=uploaded,
            prompt=args.prompt,
            negative=args.negative,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            split_step=args.split_step,
            cfg=args.cfg,
            shift=args.shift,
            seed=seed,
            blocks_to_swap=args.blocks_to_swap,
            high_model=args.high_model,
            low_model=args.low_model,
            vae_name=args.vae,
            t5_name=args.t5,
        )

        result = api_post("/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = result["prompt_id"]
        print(f"  Queued: {prompt_id}")

        history = wait_for_completion(prompt_id)
        downloaded = download_outputs(history, args.output, img.stem)

        if not downloaded:
            print("  Warning: no video outputs found. Check ComfyUI logs.")


if __name__ == "__main__":
    main()
