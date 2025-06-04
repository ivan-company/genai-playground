import os
from google import genai
from google.genai import types

from constants import GENERATE_PROMPT, OUTPUT_FOLDER


def run(output_prefix, *args, **kwargs):
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_image.png")
    client = genai.Client()

    response1 = client.models.generate_images(
        model="imagen-3.0-generate-002",
        prompt=GENERATE_PROMPT,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            include_rai_reason=True,
            output_mime_type="image/jpeg",
        ),
    )

    with open(output_path, "wb") as file:
        file.write(response1.generated_images[0].image.image_bytes)
