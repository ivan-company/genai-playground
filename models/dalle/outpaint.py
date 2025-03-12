import os
from openai import OpenAI
import requests
from rich import print
from PIL import Image

from utils import (
    create_mask,
    open_image,
)

from constants import OUTPAINT_PROMPT, OUTPUT_FOLDER, INPUT_FOLDER


def outpaint_image(image_name, output_prefix):
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
    output_path = os.path.join(
        OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")

    client = OpenAI()
    image = open_image(input_path)
    image = image.resize((256, 256), Image.LANCZOS)
    extended_image, _ = create_mask(image, 128)

    with open("temp/extended_image.png", "rb") as img:
        response = client.images.edit(
            model="dall-e-2",
            image=img,
            prompt=OUTPAINT_PROMPT,
            n=1,
            size=f"{extended_image.width}x{extended_image.height}",
        )

    with open(output_path, "wb") as file:
        file.write(requests.get(response.data[0].url).content)


def run(image_names, output_prefix, *args, **kwargs):
    for image_name in image_names:
        outpaint_image(image_name, output_prefix)
