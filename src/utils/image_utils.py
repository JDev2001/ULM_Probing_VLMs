from PIL import Image

def resize_image_aspect_ratio(img, target_size=150, only_shrink=True):
    """
    Resizes a PIL Image while maintaining its aspect ratio.

    The longest side of the image will be scaled to match the `target_size`.

    Args:
        img (PIL.Image.Image): The image to resize.
        target_size (int): The desired size for the longest side.
        only_shrink (bool): If True, images that are already smaller
                            than the target size will not be enlarged.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Get original dimensions
    width, height = img.size

    # Optionally, check if resizing is needed at all (more efficient)
    if only_shrink and max(width, height) <= target_size:
        return img

    # Calculate scaling ratio based on the longest side
    if width > height:
        # Image is wider than it is tall (landscape)
        ratio = target_size / width
        new_width = target_size
        new_height = int(height * ratio)
    else:
        # Image is taller than it is wide or is square (portrait)
        ratio = target_size / height
        new_height = target_size
        new_width = int(width * ratio)

    # Resize the image using a high-quality filter and return it
    # The size must be passed as a tuple of integers
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

