import torch
from diffusers import StableDiffusionPipeline
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, GENERATE_PROMPT, GENERATE_NEGATIVE_PROMPT


def run():
    # Load the pipeline with MPS support for M4
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1")

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
    image.save("output/sd_image.png")
