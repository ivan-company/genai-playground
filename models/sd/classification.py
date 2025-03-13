import os
import torch
from transformers import CLIPProcessor, CLIPModel

from utils import (
    open_image,
)
from constants import INPUT_FOLDER

MODEL = "openai/clip-vit-large-patch14"


def classification(image_name, *args, **kwargs):
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")

    image = open_image(input_path)

    model = CLIPModel.from_pretrained(MODEL)
    processor = CLIPProcessor.from_pretrained(MODEL)
    ad_categories = [
        "tech product advertisement",
        "food advertisement",
        "fashion advertisement",
        "beauty product advertisement",
        "automotive advertisement",
        "lifestyle product advertisement",
        "luxury goods advertisement",
        "travel advertisement",
        "fitness product advertisement",
        "home decor advertisement"
    ]

    # Process image and categories
    inputs = processor(
        text=ad_categories,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Get top categories with scores
    values, indices = probs[0].topk(3)
    top_predictions = [
        (ad_categories[idx], score.item())
        for idx, score in zip(indices, values)
    ]

    print(top_predictions)


def run(image_names, output_prefix, *args, **kwargs):
    for image_name in image_names:
        classification(image_name)
