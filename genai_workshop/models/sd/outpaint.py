import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from rich import print
from models.sd.classify import classify

from utils import (
    open_image,
    create_mask,
    generate_ad_prompt
)
from constants import OUTPUT_FOLDER, INPUT_FOLDER

MODEL = "stabilityai/stable-diffusion-2-inpainting"


def outpaint(pipe, image_name, output_prefix, *args, **kwargs):
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
    output_path = os.path.join(
        OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")

    classification_results = classify(image_name)
    print(f"Ad classified as: {classification_results}")

    prompts = generate_ad_prompt(classification_results)
    print(f"Prompts: {prompts}")

    image = open_image(input_path)
    # image = image.resize((256, 256), Image.LANCZOS)
    image = image.resize(
        (round(image.width / 8) * 8, round(image.height / 8) * 8), Image.LANCZOS)
    extended_image, mask = create_mask(image, 128)

    # Run outpainting
    result = pipe(
        prompt=prompts["prompt"],
        negative_prompt=prompts["negative_prompt"],
        image=extended_image,
        mask_image=mask.resize(
            (extended_image.width, extended_image.height), Image.LANCZOS
        ),
        num_inference_steps=40,  # Increase for better quality
        guidance_scale=7.5,
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
