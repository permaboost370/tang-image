import io, os
from typing import Tuple
from PIL import Image
import httpx

class RefImageError(Exception):
    pass

async def load_reference_png_bytes() -> bytes:
    """
    Load the project's fixed reference image as PNG bytes.
    Priority: REFERENCE_IMAGE_PATH (local) -> REFERENCE_IMAGE_URL (download).
    Converts to PNG if needed. Enforces RGBA or RGB.
    """
    path = os.getenv("REFERENCE_IMAGE_PATH", "").strip()
    url  = os.getenv("REFERENCE_IMAGE_URL", "").strip()

    if not path and not url:
        raise RefImageError("Set REFERENCE_IMAGE_PATH or REFERENCE_IMAGE_URL")

    img = None
    if path:
        try:
            img = Image.open(path)
        except Exception as e:
            raise RefImageError(f"Failed to open local ref image: {e}") from e
    else:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.get(url)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content))
        except Exception as e:
            raise RefImageError(f"Failed to download ref image: {e}") from e

    # Convert to PNG bytes
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
