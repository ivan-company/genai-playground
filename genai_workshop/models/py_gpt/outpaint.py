from utils import open_image
from constants import OUTPAINT_PROMPT, OUTPUT_FOLDER, INPUT_FOLDER
import os
import base64
from py_gpt.internal.image_processing.image_outpainter import FourOImageOutpainter


def run(image_names, output_prefix, *args, **kwargs):

    client = FourOImageOutpainter()

    for image_name in image_names:
        input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
        output_path = os.path.join(
            OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")

        with open(input_path, "rb") as image:
            results = client.call(bytearray(image.read()), 1)
            with open(output_path, "wb") as file:
                file.write(base64.b64decode(results[0]))
