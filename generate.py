#!/usr/bin/env python3
"""
Batch animate images using Wan2.2 + VBVR workflow via ComfyUI API.

Usage:
    python generate.py image.jpg --prompt "woman walking in neon city"
    python generate.py ./images/ --prompt "..." --output ./clips
    python generate.py image.jpg --prompt "..." --pipeline 3

Pipelines in the workflow:
    1 = Standard HIGH model + VBVR LoRA
    2 = SNR Calibrated Hybrid (HIGH+LOW) + lightx2v LoRA
    3 = VBVR SNR Calibrated Hybrid (HIGH+LOW) + lightx2v LoRA  [default, best]

Required models in ComfyUI/models/:
    checkpoints/  Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors
                  Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors
                  VBVR-wan2.2-I2V-14B-high-SNR-Calibrated-Hybrid.safetensors
                  VBVR-wan2.2-I2V-14B-low-SNR-Calibrated-Hybrid.safetensors
    vae/          Wan2_1_VAE_bf16.safetensors
    t5/           umt5-xxl-enc-bf16.safetensors
    loras/        lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
                  Wan22_I2V_VBVR_HIGH_rank_64_fp16.safetensors
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
WORKFLOW_PATH = Path(__file__).parent / "workflows" / "wan22_vbvr.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED_CONTROL_VALUES = {"fixed", "randomize", "increment", "increment_seed"}

PIPELINE_OUTPUT_NODES = {
    1: "60",
    2: "154",
    3: "176",
}

PIPELINE_PROMPT_NODES = {
    1: "124",
    2: "153",
    3: "171",
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


def build_link_map(workflow):
    link_map = {}
    for link in workflow["links"]:
        link_id, src_node, src_slot, dst_node, dst_slot, _ = link
        link_map[link_id] = [str(src_node), src_slot]
    return link_map


def convert_to_api(workflow):
    link_map = build_link_map(workflow)
    api = {}

    for node in workflow["nodes"]:
        node_id = str(node["id"])
        inputs = {}

        raw_widgets = node.get("widgets_values", [])
        widget_values = list(raw_widgets.values()) if isinstance(raw_widgets, dict) else list(raw_widgets)

        widget_idx = 0
        for inp in node.get("inputs", []):
            name = inp["name"]
            link_id = inp.get("link")
            has_widget = "widget" in inp

            if has_widget:
                if widget_idx < len(widget_values):
                    val = widget_values[widget_idx]
                    widget_idx += 1
                else:
                    val = None

                if name == "seed" and widget_idx < len(widget_values):
                    next_val = widget_values[widget_idx]
                    if isinstance(next_val, str) and next_val in SEED_CONTROL_VALUES:
                        widget_idx += 1

                inputs[name] = link_map[link_id] if link_id is not None else val

            elif link_id is not None:
                inputs[name] = link_map[link_id]

        api[node_id] = {
            "class_type": node["type"],
            "inputs": inputs,
            "_meta": {"title": node.get("title", node["type"])},
        }

    return api


def enable_save_output(api, pipeline):
    node_id = PIPELINE_OUTPUT_NODES[pipeline]
    if node_id in api:
        api[node_id]["inputs"]["save_output"] = True


def patch_workflow(api, image_name, prompt, pipeline):
    api["67"]["inputs"]["image"] = image_name

    for pid, node_id in PIPELINE_PROMPT_NODES.items():
        if node_id in api:
            api[node_id]["inputs"]["text"] = prompt

    enable_save_output(api, pipeline)


def disable_unused_pipelines(api, pipeline):
    output_nodes = {
        1: ["60", "28"],
        2: ["154", "142", "129", "130"],
        3: ["176", "164", "172", "173"],
    }
    for pid, nodes in output_nodes.items():
        if pid != pipeline:
            for node_id in nodes:
                if node_id in api:
                    del api[node_id]


def queue_prompt(api, client_id):
    return api_post("/prompt", {"prompt": api, "client_id": client_id})


def wait_for_completion(prompt_id, poll_interval=5, timeout=3600):
    print("  Waiting", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        history = api_get(f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print(" done.")
                return entry
            status_str = status.get("status_str", "")
            if status_str == "error":
                print()
                raise RuntimeError(f"ComfyUI error: {entry}")
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
            out_name = f"{image_stem}_node{node_id}_{filename}"
            out_path = output_dir / out_name
            urllib.request.urlretrieve(f"{COMFYUI_URL}/view?{params}", out_path)
            print(f"  Saved: {out_path}")
            downloaded.append(out_path)

    return downloaded


def collect_images(path):
    p = Path(path)
    if p.is_dir():
        images = [
            f for f in sorted(p.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        images = [p]
    else:
        print(f"Error: {path} is not an image or directory of images.")
        sys.exit(1)
    return images


def check_comfyui():
    try:
        api_get("/system_stats")
    except (urllib.error.URLError, ConnectionRefusedError):
        print(f"Error: Cannot reach ComfyUI at {COMFYUI_URL}")
        print("Start ComfyUI with: python main.py --highvram --cuda-malloc --use-pytorch-cross-attention")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Animate images using Wan2.2 + VBVR via ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Image path or directory of images")
    parser.add_argument("--prompt", "-p", required=True, help="Positive prompt describing the motion/scene")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument(
        "--pipeline", type=int, choices=[1, 2, 3], default=3,
        help="Which pipeline: 1=VBVR+LoRA, 2=SNR Hybrid, 3=VBVR SNR Hybrid (default: 3)"
    )
    parser.add_argument("--all-pipelines", action="store_true", help="Run all 3 pipelines (comparison mode)")
    parser.add_argument("--workflow", default=str(WORKFLOW_PATH), help="Path to workflow JSON")
    parser.add_argument("--comfyui-url", default=COMFYUI_URL, help="ComfyUI server URL")
    args = parser.parse_args()

    global COMFYUI_URL
    COMFYUI_URL = args.comfyui_url

    check_comfyui()

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"Error: Workflow not found at {workflow_path}")
        print("Place wan22_vbvr.json in the workflows/ directory.")
        sys.exit(1)

    with open(workflow_path) as f:
        workflow = json.load(f)

    images = collect_images(args.image)
    print(f"Found {len(images)} image(s) to process.")

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img.name}")

        api = convert_to_api(workflow)

        uploaded_name = upload_image(img)
        print(f"  Uploaded: {uploaded_name}")

        patch_workflow(api, uploaded_name, args.prompt, args.pipeline)

        if not args.all_pipelines:
            disable_unused_pipelines(api, args.pipeline)

        result = queue_prompt(api, client_id)
        prompt_id = result["prompt_id"]
        print(f"  Prompt ID: {prompt_id}")

        history = wait_for_completion(prompt_id)
        downloaded = download_outputs(history, args.output, img.stem)

        if not downloaded:
            print("  Warning: no video outputs found in history.")


if __name__ == "__main__":
    main()
