"""Deploys app.py on Modal with a T4 GPU.

One-time:
    modal volume create tts-weights
    modal volume put tts-weights unet_fsd_ema.pt /
    modal volume put tts-weights vae_gan2.pt /
    modal volume put tts-weights mel_stats.npy /
Deploy (from this folder):
    modal deploy modal_app.py
"""

import modal

PATCH = (
    "sed -i 's/proxies: Optional\\[Dict\\],/proxies: Optional[Dict] = None,/; "
    "s/resume_download: bool,/resume_download: bool = False,/' /root/BigVGAN/bigvgan.py"
)


def download_pretrained():
    """Bake CLAP + BigVGAN weights into the image, so cold starts skip the download."""
    import sys
    sys.path.insert(0, "/root/BigVGAN")
    import bigvgan
    from transformers import ClapModel, ClapProcessor
    ClapModel.from_pretrained("laion/clap-htsat-unfused")
    ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel = False)


image = (
    modal.Image.debian_slim(python_version = "3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands("git clone -q https://github.com/NVIDIA/BigVGAN /root/BigVGAN", PATCH)
    .run_function(download_pretrained)
    .env({"WEIGHTS_DIR": "/weights", "BIGVGAN_DIR": "/root/BigVGAN"})
    .add_local_python_source("app", "model")
)

weights = modal.Volume.from_name("tts-weights", create_if_missing = True)
app = modal.App("text-to-sound")


@app.function(
    image = image,
    gpu = "T4",
    volumes = {"/weights": weights},
    scaledown_window = 300,   # stay warm 5 min after the last request, then scale to zero
    max_containers = 1,       # caps the bill: never more than one GPU at a time
    timeout = 120,
)
@modal.concurrent(max_inputs = 8)   # extra requests wait on app.py's lock and get "queued" events
@modal.asgi_app()
def web():
    from app import app as fastapi_app
    return fastapi_app
