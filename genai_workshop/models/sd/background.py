import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from constants import INPUT_FOLDER, OUTPUT_FOLDER

MODEL = 'briaai/RMBG-2.0'


def background_removal(image_name, output_prefix):
    # Define paths
    input_path = os.path.join(INPUT_FOLDER, f"{image_name}.png")
    output_path = os.path.join(
        OUTPUT_FOLDER, f"{output_prefix}_{image_name}.png")

    # Load model
    pipe = AutoModelForImageSegmentation.from_pretrained(
        MODEL, trust_remote_code=True)

    device = 'cpu'  # use "cuda" for GPU inference if available
    torch.set_float32_matmul_precision('high')
    pipe = pipe.to(device)
    pipe.eval()

    original_image = Image.open(input_path).convert('RGB')

    model_size = (1024, 1024)
    resized_image = original_image.resize(model_size, Image.LANCZOS)

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform_image(resized_image).unsqueeze(0).to(device)
    # Generate mask
    with torch.no_grad():
        preds = pipe(input_tensor)[-1].sigmoid().cpu()

    # Get mask
    pred = preds[0].squeeze()
    mask_image = transforms.ToPILImage()(pred)

    # Resize mask back to original image size
    mask_image = mask_image.resize(original_image.size, Image.LANCZOS)

    # Apply mask to create transparency
    result_image = original_image.convert('RGBA')
    result_image.putalpha(mask_image)

    # Save result
    result_image.save(output_path)
    print(f"Background removed: {output_path}")


def run(image_names, output_prefix, *args, **kwargs):
    for image_name in image_names:
        background_removal(image_name, output_prefix)
