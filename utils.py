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
