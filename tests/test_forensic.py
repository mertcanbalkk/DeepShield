import sys
from pathlib import Path
from io import BytesIO
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import cv2
from PIL import Image
from forensic_engine import analyze_image, read_photo, marked_regions, copy_move_signal, robust_scale
from model_adapters import classify_ai, trufor_available


def as_bytes(arr, fmt='PNG'):
    b=BytesIO(); Image.fromarray(arr).save(b,fmt); return b.getvalue()


def test_single_image_no_original():
    rgb=np.full((240,320,3),130,np.uint8)
    result=analyze_image(as_bytes(rgb),'case.png')
    assert result.rgb.shape==(240,320,3)
    assert result.metadata['format']=='PNG'
    assert result.available['noise'] and not result.available['ela']
    assert not result.clone_pairs
    assert len(result.sha256)==64


def test_jpeg_ela_only_jpeg():
    rng=np.random.default_rng(9)
    rgb=rng.integers(0,255,(240,320,3),dtype=np.uint8)
    result=analyze_image(as_bytes(rgb,fmt='JPEG'),'x.jpg')
    assert result.available['ela']
    for x in result.maps.values():
        assert x.shape==(240,320)
        assert np.isfinite(x).all()


def test_region_overlay():
    img=np.full((240,320,3),190,np.uint8)
    score=np.zeros((240,320),np.float32)
    score[50:110,100:180]=.98
    marked, mask, regions=marked_regions(img,score,threshold=.55)
    assert len(regions)==1
    assert 1000 < np.count_nonzero(mask) < 10000
    assert not np.array_equal(marked,img)


def test_blank_zero_normalization():
    assert robust_scale(np.zeros((100,100),np.float32)).sum()==0


def test_classifier_label_mapping():
    class FakePipe:
        def __call__(self,*a,**k):
            return [{'label':'AI-generated','score':.82},{'label':'human','score':.18}]
    model=classify_ai(np.zeros((240,240,3),np.uint8),FakePipe())
    assert abs(model.ai_score-.82)<1e-5


def test_classifier_unknown_label_is_none():
    class FakePipe:
        def __call__(self,*a,**k):
            return [{'label':'LABEL_0','score':.7},{'label':'LABEL_1','score':.3}]
    model=classify_ai(np.zeros((240,240,3),np.uint8),FakePipe())
    assert model.ai_score is None


def test_missing_trufor_is_not_claimed_as_installed():
    assert not trufor_available('/nonexistent','/nonexistent/file.pth')


def test_copy_move_heuristic_with_synthetic_patch():
    rng=np.random.default_rng(31)
    rgb=np.full((480,520,3),145,np.uint8)
    # textured identical patch pasted into different region
    patch=rng.integers(0,256,(145,145,3),dtype=np.uint8)
    rgb[50:195,35:180]=patch
    rgb[265:410,300:445]=patch
    candidate_map, groups=copy_move_signal(rgb)
    assert candidate_map.shape==(480,520)
    assert len(groups) >= 1, 'Exact pixel-duplicated textured area should be considered a candidate'
