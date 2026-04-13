#!/usr/bin/env python3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

import wan_gguf_generate as wan_gguf
import ltx_generate as ltx
import generate as wan_fp8

app = typer.Typer(
    name="ai2videogen",
    help="Animate images to video using ComfyUI-backed generation models.",
    add_completion=False,
)

console = Console()

COMFYUI_LAUNCH_CMD = (
    "nohup python /home/darthvader/ComfyUI/main.py "
    "--port 8189 --cuda-malloc --use-pytorch-cross-attention "
    ">> /tmp/comfyui_wan.log 2>&1 &"
)

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8189"
COMFYUI_START_TIMEOUT = 60


class Model(str, Enum):
    wan_gguf = "wan-gguf"
    ltx = "ltx"
    wan_fp8 = "wan-fp8"


def _patch_module_url(module, url: str):
    module.COMFYUI_URL = url


def _resolve_seed(seed: int) -> int:
    if seed < 0:
        return int(uuid.uuid4().int % (2**32))
    return seed


def _probe_comfyui(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{url}/system_stats", timeout=5) as resp:
            import json
            return json.loads(resp.read())
    except Exception:
        return None


def _format_vram(stats: dict) -> str:
    try:
        devices = stats.get("system", {}).get("gpus", [])
        if devices:
            free_bytes = devices[0].get("vram_free", 0)
            free_gb = free_bytes / (1024**3)
            return f"VRAM: {free_gb:.1f} GB free"
    except Exception:
        pass
    return ""


@app.command()
def animate(
    image: str = typer.Argument(..., help="Path to image file or folder of images"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Describe the motion and scene"),
    negative: Optional[str] = typer.Option(None, "--negative", "-n", help="Negative prompt"),
    model: Model = typer.Option(Model.wan_gguf, "--model", help="Generation model to use"),
    output: str = typer.Option("./outputs", "--output", "-o", help="Output directory"),
    portrait: bool = typer.Option(False, "--portrait", help="Portrait mode: 480x832"),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: 41 frames, 15 steps"),
    frames: int = typer.Option(81, "--frames", help="Number of frames"),
    steps: int = typer.Option(20, "--steps", help="Sampling steps"),
    seed: int = typer.Option(-1, "--seed", help="Seed (-1 = random)"),
    comfyui_url: str = typer.Option(DEFAULT_COMFYUI_URL, "--comfyui-url", help="ComfyUI server URL"),
):
    if portrait:
        width, height = 480, 832
    else:
        width, height = 832, 480

    if fast:
        frames = 41
        steps = 15

    if _probe_comfyui(comfyui_url) is None:
        console.print(f"[red]Cannot reach ComfyUI at {comfyui_url}[/red]")
        console.print("[yellow]Start it with: python cli.py start[/yellow]")
        raise typer.Abort()

    if model == Model.wan_gguf:
        _run_wan_gguf(image, prompt, negative, output, width, height, frames, steps, seed, comfyui_url)
    elif model == Model.ltx:
        _run_ltx(image, prompt, negative, output, width, height, frames, steps, seed, comfyui_url)
    elif model == Model.wan_fp8:
        _run_wan_fp8(image, prompt, output, comfyui_url)


def _run_wan_gguf(image, prompt, negative, output, width, height, frames, steps, seed, comfyui_url):
    _patch_module_url(wan_gguf, comfyui_url)

    neg = negative if negative is not None else wan_gguf.DEFAULT_NEGATIVE
    images = wan_gguf.collect_images(image)

    console.print(
        f"[green]Wan2.2 GGUF[/green] — "
        f"{len(images)} image(s), {width}x{height}, {frames} frames, {steps} steps"
    )

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        console.print(f"\n[bold][{i}/{len(images)}][/bold] {img.name}")

        resolved_seed = _resolve_seed(seed)
        uploaded = wan_gguf.upload_image(img)
        console.print(f"  Uploaded: {uploaded}")

        workflow = wan_gguf.build_workflow(
            image_name=uploaded,
            prompt=prompt,
            negative=neg,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            split_step=steps // 2,
            cfg=6.0,
            shift=8.0,
            seed=resolved_seed,
            blocks_to_swap=30,
            high_model="wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
            low_model="wan2.2_i2v_low_noise_14B_Q4_K_M.gguf",
            vae_name="wan_2.1_vae.safetensors",
            t5_name="umt5_xxl_fp16.safetensors",
        )

        result = wan_gguf.api_post("/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = result["prompt_id"]
        console.print(f"  Queued: {prompt_id}")

        history = _wait_with_progress(wan_gguf, prompt_id)
        downloaded = wan_gguf.download_outputs(history, output, img.stem)

        if not downloaded:
            console.print("[yellow]  Warning: no video outputs found. Check ComfyUI logs.[/yellow]")
        else:
            for path in downloaded:
                console.print(f"[green]  Saved:[/green] {path}")


def _run_ltx(image, prompt, negative, output, width, height, frames, steps, seed, comfyui_url):
    _patch_module_url(ltx, comfyui_url)

    neg = negative if negative is not None else ltx.DEFAULT_NEGATIVE
    images = ltx.collect_images(image)

    console.print(
        f"[green]LTX-Video[/green] — "
        f"{len(images)} image(s), {width}x{height}, {frames} frames, {steps} steps"
    )

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        console.print(f"\n[bold][{i}/{len(images)}][/bold] {img.name}")

        resolved_seed = _resolve_seed(seed)
        uploaded = ltx.upload_image(img)
        console.print(f"  Uploaded: {uploaded}")

        workflow = ltx.build_workflow(
            image_name=uploaded,
            prompt=prompt,
            negative=neg,
            width=width,
            height=height,
            frames=frames,
            steps=steps,
            cfg=1.0,
            seed=resolved_seed,
            model_name="ltx-video-2b-distilled.safetensors",
        )

        result = ltx.api_post("/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = result["prompt_id"]
        console.print(f"  Queued: {prompt_id}")

        history = _wait_with_progress(ltx, prompt_id)
        downloaded = ltx.download_outputs(history, output, img.stem)

        if not downloaded:
            console.print("[yellow]  Warning: no video outputs found. Check ComfyUI logs.[/yellow]")
        else:
            for path in downloaded:
                console.print(f"[green]  Saved:[/green] {path}")


def _run_wan_fp8(image, prompt, output, comfyui_url):
    _patch_module_url(wan_fp8, comfyui_url)

    import json

    workflow_path = Path(__file__).parent / "workflows" / "wan22_vbvr.json"
    if not workflow_path.exists():
        console.print(f"[red]Workflow not found: {workflow_path}[/red]")
        raise typer.Abort()

    with open(workflow_path) as f:
        raw_workflow = json.load(f)

    images = wan_fp8.collect_images(image)
    console.print(f"[green]Wan2.2 fp8 + VBVR[/green] — {len(images)} image(s)")

    client_id = uuid.uuid4().hex

    for i, img in enumerate(images, 1):
        console.print(f"\n[bold][{i}/{len(images)}][/bold] {img.name}")

        api = wan_fp8.convert_to_api(raw_workflow)
        uploaded = wan_fp8.upload_image(img)
        console.print(f"  Uploaded: {uploaded}")

        wan_fp8.patch_workflow(api, uploaded, prompt, pipeline=3)
        wan_fp8.disable_unused_pipelines(api, pipeline=3)

        result = wan_fp8.queue_prompt(api, client_id)
        prompt_id = result["prompt_id"]
        console.print(f"  Queued: {prompt_id}")

        history = _wait_with_progress(wan_fp8, prompt_id)
        downloaded = wan_fp8.download_outputs(history, output, img.stem)

        if not downloaded:
            console.print("[yellow]  Warning: no video outputs found. Check ComfyUI logs.[/yellow]")
        else:
            for path in downloaded:
                console.print(f"[green]  Saved:[/green] {path}")


def _wait_with_progress(module, prompt_id: str, poll_interval: int = 5, timeout: int = 3600):
    import time as _time

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Generating...", total=None)
        start = _time.time()
        while _time.time() - start < timeout:
            history = module.api_get(f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    elapsed = int(_time.time() - start)
                    progress.stop()
                    console.print(f"  [green]Done[/green] in {elapsed}s")
                    return entry
                if status.get("status_str") == "error":
                    progress.stop()
                    console.print(f"[red]Generation failed:[/red] {entry}")
                    raise typer.Abort()
            _time.sleep(poll_interval)

    console.print(f"[red]Timed out after {timeout}s[/red]")
    raise typer.Abort()


@app.command()
def status(
    comfyui_url: str = typer.Option(DEFAULT_COMFYUI_URL, "--comfyui-url", help="ComfyUI server URL"),
):
    stats = _probe_comfyui(comfyui_url)
    if stats is not None:
        vram_info = _format_vram(stats)
        suffix = f" ({vram_info})" if vram_info else ""
        console.print(f"[green]✓[/green] ComfyUI running at {comfyui_url}{suffix}")
    else:
        console.print(f"[red]✗[/red] ComfyUI not running. Start with: python cli.py start")


@app.command()
def start(
    comfyui_url: str = typer.Option(DEFAULT_COMFYUI_URL, "--comfyui-url", help="ComfyUI server URL"),
):
    if _probe_comfyui(comfyui_url) is not None:
        console.print(f"[green]✓[/green] ComfyUI already running at {comfyui_url}")
        return

    console.print(f"[yellow]Starting ComfyUI on {comfyui_url}...[/yellow]")
    subprocess.Popen(COMFYUI_LAUNCH_CMD, shell=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Waiting for ComfyUI to come up...", total=None)
        deadline = time.time() + COMFYUI_START_TIMEOUT
        while time.time() < deadline:
            time.sleep(3)
            if _probe_comfyui(comfyui_url) is not None:
                progress.stop()
                console.print(f"[green]✓[/green] ComfyUI is up at {comfyui_url}")
                return

    console.print(f"[red]ComfyUI did not start within {COMFYUI_START_TIMEOUT}s.[/red]")
    console.print("[yellow]Check logs: tail -f /tmp/comfyui_wan.log[/yellow]")
    raise typer.Abort()


if __name__ == "__main__":
    app()
