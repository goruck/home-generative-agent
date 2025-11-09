"""Image processing utilities."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as ImageType


class ImageUtils:
    """Utilities for image loading and processing."""

    @staticmethod
    def load_from_bytes(data: bytes, mode: str = "RGB") -> ImageType:
        """
        Load PIL Image from bytes with optional mode conversion.

        Args:
            data: Image bytes (JPEG, PNG, etc.)
            mode: Target color mode ("RGB", "L" for grayscale, etc.)

        Returns:
            PIL Image

        Raises:
            Image.UnidentifiedImageError: If data is not a valid image

        """
        img = Image.open(io.BytesIO(data))
        if mode and img.mode != mode:
            return img.convert(mode)
        return img

    @staticmethod
    def compute_dhash(image_or_bytes: ImageType | bytes, size: int = 8) -> int:
        """
        Compute dHash (difference hash) for perceptual similarity.

        dHash compares adjacent pixels horizontally to create a binary hash
        that is robust to minor image variations.

        Args:
            image_or_bytes: PIL Image or raw bytes
            size: Hash grid size (8 -> 64-bit hash)

        Returns:
            Integer hash value (up to size*size bits)

        Raises:
            Image.UnidentifiedImageError: If bytes are not a valid image

        """
        # Load from bytes if needed
        if isinstance(image_or_bytes, bytes):
            img = ImageUtils.load_from_bytes(image_or_bytes, mode="L")
        else:
            img = (
                image_or_bytes.convert("L")
                if image_or_bytes.mode != "L"
                else image_or_bytes
            )

        # Resize to (size+1, size) for horizontal gradient comparison
        img = img.resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = img.getdata()

        # Build bit string by comparing horizontally adjacent pixels
        bits = 0
        bitpos = 0
        width = size + 1
        for y in range(size):
            row_off = y * width
            for x in range(size):
                left = pixels[row_off + x]
                right = pixels[row_off + x + 1]
                if left > right:  # Set bit if left > right
                    bits |= 1 << bitpos
                bitpos += 1

        return bits  # Up to size*size bits; with size=8 it's 64-bit

    @staticmethod
    def hamming_distance(hash_a: int, hash_b: int, max_bits: int = 64) -> int:
        """
        Compute Hamming distance between two hashes.

        Args:
            hash_a: First hash
            hash_b: Second hash
            max_bits: Maximum number of bits to consider

        Returns:
            Number of differing bits

        """
        xor_result = (hash_a ^ hash_b) & ((1 << max_bits) - 1)
        return xor_result.bit_count()
