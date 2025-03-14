from PIL import Image, ImageFilter


def open_image(path):
    max_size = 512
    image = Image.open(path).convert("RGB")

    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size))
        image.save(path)
    return image


def create_mask(image: Image, expand_pixels=256):
    expanded_size = (
        image.width + expand_pixels * 2,
        image.height + expand_pixels * 2,
    )

    # Use RGBA (transparent background) instead of black
    expanded_image = Image.new("RGBA", expanded_size, (0, 0, 0, 0))
    expanded_image.paste(image, (expand_pixels, expand_pixels))

    # Create the mask (L-mode for grayscale)
    mask = Image.new("L", expanded_size, 255)

    # Create a white mask for the original image area
    mask_draw = Image.new("L", image.size, 0)

    # Paste it at the correct position
    mask.paste(mask_draw, (expand_pixels, expand_pixels))

    # Apply Gaussian Blur to smooth edges for better transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

    # Debugging: Save files
    mask.save("temp/mask.png")
    expanded_image.save("temp/extended_image.png")

    return expanded_image.convert("RGB"), mask


def generate_ad_prompt(classification_results):
    # Base prompt elements optimized for ad continuations
    base_prompt = "professional advertisement, high quality, commercial photography, seamless continuation, professional lighting"

    # Ad-specific elements based on classification
    additional_elements = []

    for category, confidence in classification_results:
        if confidence < 0.1:
            continue

        if "product advertisement" in category:
            additional_elements.append(
                "product-focused composition, commercial product photography, clean background")
        elif "lifestyle advertisement" in category:
            additional_elements.append(
                "aspirational lifestyle setting, authentic environment, brand storytelling")
        elif "food advertisement" in category:
            additional_elements.append(
                "appetizing food presentation, culinary perfection, food marketing")
        elif "tech product" in category:
            additional_elements.append(
                "modern tech environment, innovative setting, clean design")
        elif "fashion ad" in category:
            additional_elements.append(
                "fashion editorial style, trendy setting, style-focused")
        elif "luxury brand" in category:
            additional_elements.append(
                "premium environment, luxury aesthetic, high-end atmosphere")
        elif "minimalist ad" in category:
            additional_elements.append(
                "clean minimal composition, simple elegant background, uncluttered")
        elif "vibrant ad" in category:
            additional_elements.append(
                "vibrant colors, eye-catching elements, energetic atmosphere")
        # Add more ad-specific mappings

    # Advertising-specific negative elements to avoid
    negative_prompt = "blurry, amateur looking, low quality, unprofessional, text overlay, watermarks, logos, inconsistent lighting, cluttered background, distracting elements"

    # Combine everything into a final prompt
    final_prompt = base_prompt
    if additional_elements:
        final_prompt += ", " + ", ".join(additional_elements)

    return {
        "prompt": final_prompt,
        "negative_prompt": negative_prompt
    }
