import time
import os
from google import genai
from google.genai import types

from constants import GENERATE_PROMPT, OUTPUT_FOLDER


def run(output_prefix, *args, **kwargs):
    output_path = os.path.join(OUTPUT_FOLDER, f"{output_prefix}_video.mp4")
    client = genai.Client()

    # Create operation
    operation = client.models.generate_videos(
        model="veo-2.0-generate-001",
        prompt="A neon hologram of a cat driving at top speed",
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=5,
        ),
    )

    # Poll operation
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    for n, generated_video in enumerate(operation.response.generated_videos):
        client.files.download(file=generated_video.video)
        generated_video.video.save(output_path)  # save the video
