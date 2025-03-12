from PIL import Image, ImageFilter


IMAGE_NAME = "image.png"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

GENERATE_PROMPT = "A serene lake surrounded by autumn trees, digital art style"
GENERATE_NEGATIVE_PROMPT = "blurry, bad quality, deformed"

OUTPAINT_PROMPT = (
    "A seamless and natural extension of the original image, maintaining the same artistic style, "
    "lighting, color tones, and perspective. The outpainted area should be a logical continuation "
    "of the scene, matching details and blending smoothly with the existing image. "
    "High-quality, highly detailed, and photorealistic."
)

OUTPAINT_NEGATIVE_PROMPT = (
    "blurry, distorted, low-quality, unrealistic lighting, mismatched colors, artifacts, "
    "hard edges, visible AI generation, deformed objects, incorrect perspective."
)


def open_image(path):
    return Image.open(path).convert("RGB")


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
