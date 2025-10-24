# image_provider.py
import os
import base64
import httpx

class ImageGenError(Exception):
    pass

# -----------------------
# Provider selection
# -----------------------
PROVIDER = os.getenv("IMAGE_PROVIDER", "openai").lower()

# ---- OpenAI Images API (edits) ----
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL      = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMG_SIZE   = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
OPENAI_MASK_PATH  = os.getenv("OPENAI_MASK_PATH", "").strip()

# Default prompt prefix you can adjust from env (prepended to every user prompt)
DEFAULT_PROMPT_PREFIX = os.getenv("DEFAULT_PROMPT_PREFIX", "").strip()
# Example env value:
# DEFAULT_PROMPT_PREFIX=keep the same character, identical eyes and mouth, do not alter facial expression or proportions, preserve the face details,

# ---- Stability AI (SDXL v1 img2img) ----
STABILITY_API_KEY   = os.getenv("STABILITY_API_KEY", "")
STABILITY_ENGINE    = os.getenv("STABILITY_ENGINE", "stable-diffusion-xl-1024-v1-0")
STABILITY_STRENGTH  = float(os.getenv("STABILITY_STRENGTH", "0.65"))
STABILITY_CFG       = int(os.getenv("STABILITY_CFG_SCALE", "7"))
STABILITY_STEPS     = int(os.getenv("STABILITY_STEPS", "30"))
STABILITY_SEED      = os.getenv("STABILITY_SEED")  # optional

def _prepend_prefix(prompt: str) -> str:
    if DEFAULT_PROMPT_PREFIX:
        # Ensure nice spacing without double spaces
        return (DEFAULT_PROMPT_PREFIX + " " + prompt).strip()
    return prompt

async def generate_image_from_reference(prompt: str, ref_png_bytes: bytes) -> bytes:
    """
    Dispatch to the chosen provider and return PNG bytes.
    """
    full_prompt = _prepend_prefix(prompt)

    if PROVIDER == "stability":
        return await _stability_img2img(full_prompt, ref_png_bytes)
    return await _openai_img_edit(full_prompt, ref_png_bytes)

# ------------------------------------------------------------------
# OPENAI: Images "edits" endpoint (img2img). No response_format param.
# Supports optional mask at OPENAI_MASK_PATH.
# ------------------------------------------------------------------
async def _openai_img_edit(prompt: str, ref_png_bytes: bytes) -> bytes:
    if not OPENAI_API_KEY:
        raise ImageGenError("Missing OPENAI_API_KEY")

    url = f"{OPENAI_BASE_URL}/images/edits"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    files = {
        "image": ("ref.png", ref_png_bytes, "image/png"),
    }
    # If you provide a mask PNG (opaque = KEEP, transparent = EDIT), it will be sent automatically.
    if OPENAI_MASK_PATH and os.path.isfile(OPENAI_MASK_PATH):
        try:
            with open(OPENAI_MASK_PATH, "rb") as f:
                files["mask"] = ("mask.png", f.read(), "image/png")
        except Exception:
            # Non-fatal: continue without a mask
            pass

    data = {
        "model": OPENAI_MODEL,
        "prompt": prompt,
        "size": OPENAI_IMG_SIZE,
        "n": "1",  # single sample for consistency
        # IMPORTANT: do NOT send response_format here; some servers reject it.
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, data=data, files=files)
        if r.status_code != 200:
            raise ImageGenError(f"OpenAI error {r.status_code}: {r.text}")

        js = r.json()
        if not js.get("data"):
            raise ImageGenError("OpenAI returned no data")

        first = js["data"][0]

        # Prefer base64 if present
        b64 = first.get("b64_json")
        if b64:
            try:
                return base64.b64decode(b64)
            except Exception as e:
                raise ImageGenError(f"Failed to decode OpenAI base64 image: {e}") from e

        # Otherwise, fall back to temporary URL
        url_field = first.get("url")
        if url_field:
            try:
                img_resp = await client.get(url_field, timeout=120)
                img_resp.raise_for_status()
                return img_resp.content
            except Exception as e:
                raise ImageGenError(f"Failed to download image from OpenAI URL: {e}") from e

        raise ImageGenError("No image payload in OpenAI response (no b64_json or url).")

# -----------------------------------------------------------
# STABILITY: SDXL image-to-image (legacy v1 endpoint)
# NOTE: Do NOT send 'output_format' here (causes 400 on some gateways).
# -----------------------------------------------------------
async def _stability_img2img(prompt: str, ref_png_bytes: bytes) -> bytes:
    if not STABILITY_API_KEY:
        raise ImageGenError("Missing STABILITY_API_KEY")

    url = f"https://api.stability.ai/v1/generation/{STABILITY_ENGINE}/image-to-image"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}
    files = {
        "init_image": ("ref.png", ref_png_bytes, "image/png"),
    }
    data = {
        "text_prompts[0][text]": prompt,
        "text_prompts[0][weight]": "1",
        "image_strength": str(STABILITY_STRENGTH),
        "cfg_scale": str(STABILITY_CFG),
        "steps": str(STABILITY_STEPS),
        "samples": "1",
        # 'output_format' intentionally omitted
    }
    if STABILITY_SEED:
        data["seed"] = STABILITY_SEED

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, headers=headers, data=data, files=files)
        if r.status_code != 200:
            raise ImageGenError(f"Stability error {r.status_code}: {r.text}")

        # Prefer JSON with base64 artifacts
        ctype = r.headers.get("Content-Type", "")
        if "application/json" in ctype:
            try:
                js = r.json()
            except Exception as e:
                raise ImageGenError(f"Unexpected Stability response format: {e}") from e
            arts = js.get("artifacts", [])
            if not arts or not arts[0].get("base64"):
                raise ImageGenError("No image in Stability response")
            return base64.b64decode(arts[0]["base64"])

        # Fallback: some variants return image/* directly
        if ctype.startswith("image/"):
            return r.content

        raise ImageGenError(f"Unrecognized Stability response type: {ctype}")
