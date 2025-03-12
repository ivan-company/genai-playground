import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from rich import print

from utils import (
    open_image,
    create_mask,
)
from constants import OUTPAINT_PROMPT, OUTPUT_FOLDER, INPUT_FOLDER, OUTPAINT_NEGATIVE_PROMPT

MODEL = "stabilityai/stable-diffusion-2-inpainting"


def outpaint(pipe, image_name, output_prefix, *args, **kwargs):
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
    output_path = os.path.join(
        OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")

    image = open_image(input_path)
    image = image.resize((256, 256), Image.LANCZOS)
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
    result.save(output_path)


def run(image_names, output_prefix, *args, **kwargs):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL, use_auth_token=True
    )
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()
    for image_name in image_names:
        outpaint(pipe, image_name, output_prefix)
        print(f"Outpainted: {output_prefix}_{image_name}.png")
