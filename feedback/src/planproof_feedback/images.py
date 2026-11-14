from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

# Preserve large uploaded drawing rasters for native-coordinate detail crops.
# Model overview sizing remains independent of this finite source decode limit.
MAX_SOURCE_PIXELS = 150_000_000


class UnsupportedInput(ValueError):
    pass


@dataclass(frozen=True)
class ImageInput:
    image_id: str
    data_url: str
    metadata: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_image(image: Image.Image, image_id: str, source_sha: str, page: int, config: dict):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        image = rgb
    else:
        image = image.convert("RGB")
    width, height = image.size
    image.thumbnail((config["max_edge"], config["max_edge"]), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    payload = stream.getvalue()
    return ImageInput(
        image_id,
        "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
        {
            "image_id": image_id,
            "source_sha256": source_sha,
            "page_index": page,
            "source_render_size": [width, height],
            "sent_size": list(image.size),
            "sent_to_source_render": [width / image.width, height / image.height],
            "sent_sha256": hashlib.sha256(payload).hexdigest(),
            "coordinate_system": "sent-image pixel edges; top-left; x right; y down",
        },
    )


def load_images(paths: list[Path], config: dict) -> list[ImageInput]:
    images = []
    if not paths:
        raise UnsupportedInput("At least one clean source image or PDF is required")
    for index, path in enumerate(paths):
        source_sha = sha256(path)
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(path)
            try:
                if len(pdf) + len(images) > config["max_pages"]:
                    raise UnsupportedInput(
                        "PDF exceeds max_pages; pages are never silently dropped"
                    )
                for page_index in range(len(pdf)):
                    page = pdf[page_index]
                    width, height = page.get_size()
                    scale = min(config["pdf_dpi"] / 72, config["max_edge"] / max(width, height))
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil().copy()
                    bitmap.close()
                    page.close()
                    images.append(
                        encode_image(
                            image, f"input-{index}-p{page_index}", source_sha, page_index, config
                        )
                    )
            finally:
                pdf.close()
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            with Image.open(path) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise UnsupportedInput("Input bytes do not match a supported PNG/JPEG image")
                images.append(encode_image(image, f"input-{index}-p0", source_sha, 0, config))
        else:
            raise UnsupportedInput("Supported inputs: PNG, JPEG and rendered PDF; no IFC/DWG/DXF")
    if not images or len(images) > config["max_pages"]:
        raise UnsupportedInput("Input image count is outside max_pages")
    return images


def validate_citations(citations, images: list[ImageInput]):
    if not isinstance(citations, list):
        raise ValueError("citations must be a list")
    sizes = {image.image_id: image.metadata["sent_size"] for image in images}
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("image_id") not in sizes:
            raise ValueError("Citation names an image that was not supplied")
        bbox = citation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
                for x in bbox
            )
        ):
            raise ValueError("Citation bbox must contain four finite numbers")
        width, height = sizes[citation["image_id"]]
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Citation is outside its supplied image")
