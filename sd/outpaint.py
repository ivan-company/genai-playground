import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from rich import print

from utils import (
    open_image,
    create_mask,
    OUTPAINT_PROMPT,
    OUTPAINT_NEGATIVE_PROMPT,
)


def run():
    # Use the same model that worked for inpainting
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting", use_auth_token=True
    )
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    try:
        image = open_image("output/sd_image.png")
    except FileNotFoundError:
        print("[bold red]Image not found, generating a new one...")
        from sd.generate import run
        run()
        image = open_image("output/sd_image.png")
    extended_image, mask = create_mask(image, 128)

    # Run outpainting
    result = pipe(
        prompt=OUTPAINT_PROMPT,
        negative_prompt=OUTPAINT_NEGATIVE_PROMPT,
        image=extended_image,
        mask_image=mask.resize(
            (extended_image.width, extended_image.height), Image.LANCZOS
        ),
        num_inference_steps=30,  # Increase for better quality
        guidance_scale=6.0,
        width=extended_image.width,
        height=extended_image.height,
    ).images[0]

    # Save the result
    result.save("output/sd_outpaint.png")
