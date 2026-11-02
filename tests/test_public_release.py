from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from upload_safety import validate_upload


def image_bytes(size=(32, 32), fmt='PNG'):
    buff = BytesIO()
    Image.fromarray(np.zeros((size[1],size[0],3), dtype=np.uint8)).save(buff, fmt)
    return buff.getvalue()


def test_valid_photo():
    assert validate_upload(image_bytes(), 'photo.jpg')[0] == 'PNG'


def test_non_image_and_suffix_rejected():
    with pytest.raises(ValueError):
        validate_upload(b'Not an image', 'photo.png')
    with pytest.raises(ValueError):
        validate_upload(image_bytes(), 'photo.txt')


def test_oversized_and_invalid_pixels(monkeypatch):
    import upload_safety
    monkeypatch.setattr(upload_safety, 'MAX_UPLOAD_BYTES', 5)
    with pytest.raises(ValueError, match='35 MB'):
        validate_upload(image_bytes(), 'photo.png')
    monkeypatch.setattr(upload_safety, 'MAX_UPLOAD_BYTES', 35*1024*1024)
    monkeypatch.setattr(upload_safety, 'MAX_PIXELS', 30)
    with pytest.raises(ValueError, match='36 megapiksel'):
        validate_upload(image_bytes(), 'photo.png')


def test_github_ignore_protects_personal_data():
    ignore = (Path(__file__).resolve().parents[1] / '.gitignore').read_text()
    for entry in ('datasets/', 'outputs/', 'models/', '.env', '*.pt', 'uploads/'):
        assert entry in ignore
