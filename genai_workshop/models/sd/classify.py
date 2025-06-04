import os
import torch
from transformers import CLIPProcessor, CLIPModel

from utils import (
    open_image,
)
from constants import INPUT_FOLDER

MODEL = "openai/clip-vit-large-patch14"


def classify(image_name):
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load image
    image = open_image(os.path.join(INPUT_FOLDER, f"{image_name}.png"))

    # Define ad-specific categories
    ad_categories = [
        "product advertisement", "lifestyle advertisement", "food advertisement",
        "tech product ad", "fashion ad", "beauty product ad", "automotive ad",
        "luxury brand ad", "travel advertisement", "fitness product ad",
        "minimalist ad style", "vibrant ad style", "corporate ad style",
        "outdoor advertisement", "indoor advertisement"
    ]

    # Process image and text
    inputs = processor(text=ad_categories, images=image,
                       return_tensors="pt", padding=True)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Get top categories
    top_probs, top_indices = probs.topk(3)
    results = [(ad_categories[idx], prob.item())
               for idx, prob in zip(top_indices[0], top_probs[0])]

    print(results)
    return results


def run(image_names, output_prefix, *args, **kwargs):
    for image_name in image_names:
        classify(image_name)
