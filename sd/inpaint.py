import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

# Load inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
)
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

image = Image.open("output/image.png").convert("RGB").resize((512, 512))

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
result.save("output/inpainting.png")
