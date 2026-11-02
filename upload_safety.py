"""Validation for untrusted image uploads before display or model execution."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 35 * 1024 * 1024
MAX_PIXELS = 36_000_000
SUPPORTED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'TIFF'}
SUPPORTED_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}


def validate_upload(data: bytes, filename: str) -> tuple[str, tuple[int, int]]:
    """Validate byte size, actual image format, dimensions, and decoding.

    A filename extension is only a UI hint: genuine PNG bytes with a JPG
    extension remain supported but are reported by the original app.
    """
    if not isinstance(data, bytes) or not data:
        raise ValueError('Dosya boş veya okunamadı.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError('Görsel en fazla 35 MB olabilir.')
    if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError('Desteklenmeyen görüntü uzantısı.')
    try:
        with Image.open(BytesIO(data)) as im:
            fmt = str(im.format).upper()
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError('Bu görüntü biçimi desteklenmiyor.')
            w, h = im.size
            if w < 1 or h < 1 or w * h > MAX_PIXELS:
                raise ValueError('Görsel 36 megapiksel sınırını aşıyor.')
            im.verify()
        return fmt, (w, h)
    except (UnidentifiedImageError, OSError, SyntaxError, OverflowError, Image.DecompressionBombError) as exc:
        raise ValueError('Görsel dosyası bozuk veya okunabilir görüntü değil.') from exc
