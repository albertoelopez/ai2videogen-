import sys
import time
import uuid
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wan_gguf_generate as w

import gradio as gr

COMFYUI_URL = "http://127.0.0.1:8189"
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
OUTPUT_DIR = "./outputs"

RESOLUTION_MAP = {
    "Landscape 832×480": (832, 480),
    "Portrait 480×832": (480, 832),
    "Square 624×624": (624, 624),
}

FRAMES_LABELS = {
    41: "41 frames (~1.7 s)",
    81: "81 frames (~3.4 s)",
    121: "121 frames (~5.0 s)",
}


def comfyui_is_reachable():
    try:
        w.api_get("/system_stats")
        return True
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return False


def frames_label(frames):
    return FRAMES_LABELS.get(frames, f"{frames} frames")


def generate(image_path, prompt, negative, resolution, frames, steps, cfg, seed):
    if not image_path:
        yield gr.update(value="Error: please upload an image."), None, ""
        return
    if not prompt or not prompt.strip():
        yield gr.update(value="Error: prompt cannot be empty."), None, ""
        return

    width, height = RESOLUTION_MAP[resolution]
    actual_seed = seed if seed >= 0 else int(uuid.uuid4().int % (2 ** 32))
    split_step = max(1, steps // 2)

    yield gr.update(value="Uploading image..."), None, ""

    try:
        uploaded_name = w.upload_image(image_path)
    except Exception as exc:
        yield gr.update(value=f"Upload failed: {exc}"), None, ""
        return

    yield gr.update(value="Building workflow..."), None, ""

    workflow = w.build_workflow(
        image_name=uploaded_name,
        prompt=prompt.strip(),
        negative=negative.strip() if negative else DEFAULT_NEGATIVE,
        width=width,
        height=height,
        frames=frames,
        steps=steps,
        split_step=split_step,
        cfg=cfg,
        shift=8.0,
        seed=actual_seed,
        blocks_to_swap=30,
        high_model="wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
        low_model="wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
        vae_name="wan_2.1_vae.safetensors",
        t5_name="umt5_xxl_fp16.safetensors",
    )

    yield gr.update(value="Queuing job..."), None, ""

    try:
        result = w.api_post("/prompt", {"prompt": workflow, "client_id": uuid.uuid4().hex})
    except Exception as exc:
        yield gr.update(value=f"Queue failed: {exc}"), None, ""
        return

    prompt_id = result["prompt_id"]
    yield gr.update(value=f"Queued (id: {prompt_id[:8]}…). Waiting for ComfyUI..."), None, ""

    start = time.time()
    timeout = 3600
    poll_interval = 5
    elapsed_ticks = 0

    while time.time() - start < timeout:
        try:
            history = w.api_get(f"/history/{prompt_id}")
        except Exception as exc:
            yield gr.update(value=f"Poll error: {exc}"), None, ""
            return

        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                break
            if status.get("status_str") == "error":
                yield gr.update(value="Generation failed — check ComfyUI logs."), None, ""
                return

        elapsed = int(time.time() - start)
        elapsed_ticks += 1
        dots = "." * (elapsed_ticks % 4)
        yield gr.update(value=f"Generating{dots} ({elapsed}s elapsed)"), None, ""
        time.sleep(poll_interval)
    else:
        yield gr.update(value="Timed out waiting for ComfyUI."), None, ""
        return

    yield gr.update(value="Downloading output..."), None, ""

    image_stem = Path(image_path).stem
    try:
        paths = w.download_outputs(entry, OUTPUT_DIR, image_stem)
    except Exception as exc:
        yield gr.update(value=f"Download failed: {exc}"), None, ""
        return

    if not paths:
        yield gr.update(value="Done — but no video output found. Check ComfyUI logs."), None, ""
        return

    video_path = str(paths[0])
    yield gr.update(value="Done!"), video_path, video_path


def build_ui():
    reachable = comfyui_is_reachable()

    with gr.Blocks(theme=gr.themes.Soft(), title="Wan 2.2 GGUF — Image to Video") as demo:

        gr.Markdown("# Wan 2.2 GGUF — Image to Video")

        if not reachable:
            gr.HTML(
                "<div style='background:#7f1d1d;color:#fecaca;padding:12px 16px;"
                "border-radius:8px;margin-bottom:8px;font-weight:500;'>"
                "⚠ ComfyUI is not reachable at "
                + COMFYUI_URL
                + ". Start it before generating."
                "</div>"
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input Image",
                    type="filepath",
                    height=300,
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the motion...",
                    lines=4,
                )

                with gr.Accordion("Negative Prompt", open=False):
                    negative_input = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE,
                        lines=4,
                    )

                resolution_input = gr.Radio(
                    choices=list(RESOLUTION_MAP.keys()),
                    value="Landscape 832×480",
                    label="Resolution",
                )

                frames_input = gr.Radio(
                    choices=[
                        (FRAMES_LABELS[41], 41),
                        (FRAMES_LABELS[81], 81),
                        (FRAMES_LABELS[121], 121),
                    ],
                    value=81,
                    label="Duration",
                )

                steps_input = gr.Slider(
                    minimum=10,
                    maximum=30,
                    step=1,
                    value=20,
                    label="Steps",
                )

                cfg_input = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=6.0,
                    label="CFG Scale",
                )

                seed_input = gr.Number(
                    value=-1,
                    label="Seed (-1 = random)",
                    precision=0,
                )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Status",
                    value="Idle",
                    interactive=False,
                    lines=2,
                )

                video_output = gr.Video(
                    label="Output Video",
                    height=400,
                )

                path_output = gr.Textbox(
                    label="Saved to",
                    interactive=False,
                    lines=1,
                )

        generate_btn.click(
            fn=generate,
            inputs=[
                image_input,
                prompt_input,
                negative_input,
                resolution_input,
                frames_input,
                steps_input,
                cfg_input,
                seed_input,
            ],
            outputs=[status_output, video_output, path_output],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
