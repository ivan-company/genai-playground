from openai import OpenAI
import requests
from rich import print

from utils import (
    create_mask,
    open_image,
    OUTPAINT_PROMPT,
)


def run():
    client = OpenAI()

    try:
        image = open_image("output/dalle_image.png")
    except FileNotFoundError:
        print("[bold red]Image not found, generating a new one...")
        from dalle.generate import run
        run()
        image = open_image("output/dalle_image.png")
    extended_image, mask = create_mask(image, 256)

    with open("temp/extended_image.png", "rb") as img:
        response = client.images.edit(
            model="dall-e-2",
            image=img,
            prompt=OUTPAINT_PROMPT,
            n=1,
            size=f"{extended_image.width}x{extended_image.height}",
        )

    # download the image and store it as image.png

    with open("output/dalle_outpainting.png", "wb") as file:
        file.write(requests.get(response.data[0].url).content)
