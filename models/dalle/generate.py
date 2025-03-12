import os
from openai import OpenAI
import requests
from constants import GENERATE_PROMPT, OUTPUT_FOLDER

MODEL = "dall-e-2"


def run(output_prefix, *args, **kwargs):
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_image.png")

    client = OpenAI()

    response = client.images.generate(
        model=MODEL,
        prompt=GENERATE_PROMPT,
        n=1,
        size="512x512",
    )

    with open(output_path, "wb") as file:
        file.write(requests.get(response.data[0].url).content)
