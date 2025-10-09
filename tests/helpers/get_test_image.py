"""Helper to get test image."""

from pathlib import Path

import PIL.Image
from PIL.Image import Image


def get_test_image(image_name: str) -> Image:
    """Helper function to get an image from the `assets` folder."""
    assets_folder = Path(__file__).parents[1] / "assets"
    image_path = assets_folder / image_name
    assert image_path.exists()
    return PIL.Image.open(image_path)
