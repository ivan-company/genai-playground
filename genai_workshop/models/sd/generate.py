import os
import torch
from diffusers import StableDiffusionPipeline
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, GENERATE_PROMPT, GENERATE_NEGATIVE_PROMPT, OUTPUT_FOLDER


MODEL = "stabilityai/stable-diffusion-2-1"


def run(output_prefix, *args, **kwargs):
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_image.png")
    # Load the pipeline with MPS support for M4
    pipe = StableDiffusionPipeline.from_pretrained(MODEL)

    # Use MPS for M4 hardware acceleration
    pipe = pipe.to("mps")

    # Optional: Enable attention slicing to reduce memory requirements
    pipe.enable_attention_slicing()

    # Generate image
    image = pipe(
        prompt=GENERATE_PROMPT,
        negative_prompt=GENERATE_NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    ).images[0]

    # Save the image
    image.save(output_path)
