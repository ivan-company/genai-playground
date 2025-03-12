from openai import OpenAI
import requests
from utils import GENERATE_PROMPT


def run():
    client = OpenAI()

    response = client.images.generate(
        model="dall-e-2",
        prompt=GENERATE_PROMPT,
        n=1,
        size="512x512",
    )

    with open("output/dalle_image.png", "wb") as file:
        file.write(requests.get(response.data[0].url).content)
