#!/usr/bin/env python3
"""
Batch animate images using LTX-Video 2 via ComfyUI API.
Fast alternative to Wan2.2 — fits entirely in 24GB VRAM, no block swap needed.

Usage:
    python ltx_generate.py image.jpg --prompt "woman walks through neon city"
    python ltx_generate.py ./images/ --prompt "..." --output ./clips
    python ltx_generate.py ./images/ --prompt "..." --width 768 --height 512 --frames 97

Setup:
    1. Install ComfyUI extension: ComfyUI-LTXVideo (by Lightricks)
       https://github.com/Lightricks/ComfyUI-LTXVideo
    2. Download model to ComfyUI/models/checkpoints/:
       ltx-video-2b-distilled.safetensors
    3. ComfyUI launch flags for 4090:
       --highvram --cuda-malloc --use-pytorch-cross-attention

Frames must be 8n+1 (e.g. 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 121)
Resolution: keep width*height under 786432 (e.g. 768x512, 512x768, 960x544)
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

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "watermark, text, logo, ugly, low quality, artifacts"
)


def build_workflow(image_name, prompt, negative, width, height, frames, steps, cfg, seed, model_name):
    return {
        "1": {
            "class_type": "LTXVModelLoader",
            "inputs": {
                "model": model_name,
                "precision": "bfloat16",
            },
            "_meta": {"title": "LTX-Video Model"},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
            "_meta": {"title": "Input Image"},
        },
        "3": {
            "class_type": "ImageResizeKJ",
            "inputs": {
                "image": ["2", 0],
                "width": width,
                "height": height,
                "upscale_method": "lanczos",
                "keep_proportion": "resize",
                "divisible_by": 32,
            },
            "_meta": {"title": "Resize"},
        },
        "4": {
            "class_type": "LTXVTextEncode",
            "inputs": {
                "model": ["1", 0],
                "positive_prompt": prompt,
                "negative_prompt": negative,
            },
            "_meta": {"title": "Text Encode"},
        },
        "5": {
            "class_type": "LTXVImageToVideoLatent",
            "inputs": {
                "model": ["1", 0],
                "image": ["3", 0],
                "width": width,
                "height": height,
                "num_frames": frames,
                "noise_aug_strength": 0.0,
            },
            "_meta": {"title": "Image to Video Latent"},
        },
        "6": {
            "class_type": "LTXVSampler",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["4", 0],
                "latent_image": ["5", 0],
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "sampler": "euler",
                "scheduler": "ltxv_default",
                "denoise": 1.0,
            },
            "_meta": {"title": "LTX Sampler"},
        },
        "7": {
            "class_type": "LTXVDecode",
            "inputs": {
                "model": ["1", 0],
                "samples": ["6", 0],
                "skip_latents_at_start": 1,
            },
            "_meta": {"title": "Decode"},
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": "ltxv",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
            "_meta": {"title": "Save Video"},
        },
    }


def api_get(endpoint):
    url = f"{COMFYUI_URL}{endpoint}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def api_post(endpoint, data):
    url = f"{COMFYUI_URL}{endpoint}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def upload_image(image_path):
    image_path = Path(image_path)
    url = f"{COMFYUI_URL}/upload/image"
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
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["name"]


def queue_prompt(workflow, client_id):
    return api_post("/prompt", {"prompt": workflow, "client_id": client_id})


def wait_for_completion(prompt_id, poll_interval=3, timeout=1800):
    print("  Generating", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        history = api_get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                elapsed = int(time.time() - start)
                print(f" done ({elapsed}s).")
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
        for video in node_output.get("videos", []):
            filename = video["filename"]
            subfolder = video.get("subfolder", "")
            vtype = video.get("type", "output")
            params = urllib.parse.urlencode({
                "filename": filename,
                "subfolder": subfolder,
                "type": vtype,
            })
            out_name = f"{image_stem}_{filename}"
            out_path = output_dir / out_name
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


def validate_frames(n):
    if (n - 1) % 8 != 0:
        valid = [8 * i + 1 for i in range(3, 20)]
        print(f"Error: frames must be 8n+1. Closest valid values: {valid}")
        sys.exit(1)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Animate images with LTX-Video 2 via ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Image path or directory of images")
    parser.add_argument("--prompt", "-p", required=True, help="Describe the motion and scene")
    parser.add_argument("--negative", "-n", default=DEFAULT_NEGATIVE, help="Negative prompt")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--width", type=int, default=768, help="Output width (default: 768)")
    parser.add_argument("--height", type=int, default=512, help="Output height (default: 512)")
    parser.add_argument("--frames", type=int, default=97, help="Number of frames, must be 8n+1 (default: 97 ≈ 4s)")
    parser.add_argument("--steps", type=int, default=6, help="Sampling steps (default: 6)")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale (default: 1.0, distilled)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--model", default="ltx-video-2b-distilled.safetensors", help="Model filename")
    parser.add_argument("--comfyui-url", default=COMFYUI_URL)
    args = parser.parse_args()

    global COMFYUI_URL
    COMFYUI_URL = args.comfyui_url

    validate_frames(args.frames)
    check_comfyui()

    images = collect_images(args.image)
    print(f"Found {len(images)} image(s) — LTX-Video 2 ({args.width}x{args.height}, {args.frames} frames, {args.steps} steps)")

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img.name}")

        seed = args.seed if args.seed >= 0 else int(uuid.uuid4().int % (2**32))

        uploaded_name = upload_image(img)
        print(f"  Uploaded: {uploaded_name}")

        workflow = build_workflow(
            image_name=uploaded_name,
            prompt=args.prompt,
            negative=args.negative,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            cfg=args.cfg,
            seed=seed,
            model_name=args.model,
        )

        result = queue_prompt(workflow, client_id)
        prompt_id = result["prompt_id"]
        print(f"  Queued: {prompt_id}")

        history = wait_for_completion(prompt_id)
        downloaded = download_outputs(history, args.output, img.stem)

        if not downloaded:
            print("  Warning: no video outputs found. Check ComfyUI logs.")


if __name__ == "__main__":
    main()
