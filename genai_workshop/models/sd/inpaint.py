import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from constants import INPUT_FOLDER, OUTPUT_FOLDER

MODEL = "stabilityai/stable-diffusion-2-inpainting"


def inpaint(image_name, output_prefix, *args, **kwargs):
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
    output_path = os.path.join(
        OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")
    # Load inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL)
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    image = Image.open(input_path).convert("RGB").resize((512, 512))

    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[150:350, 150:350] = 255  # Creating a square mask in the center
    mask_image = Image.fromarray(mask).convert("RGB")

    # Define what to put in the masked area
    prompt = "a cat sitting, high resolution, detailed"

    # Run inpainting
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # Save the result
    result.save(output_path)


def run(image_names, output_prefix, *args, **kwargs):
    for image_name in image_names:
        inpaint(image_name, output_prefix)
