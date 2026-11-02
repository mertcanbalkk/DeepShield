"""Offline toy-model tests of wiring; do not measure EfficientNet accuracy."""
import numpy as np
import pytest


def test_real_photo_false_ai_guard_pipeline():
    torch = pytest.importorskip('torch')
    from model_adapters import classify_custom

    class ToyModel(torch.nn.Module):
        def forward(self, x):
            m = x.mean(dim=(1,2,3))
            return torch.stack((10*(1-m),10*m),dim=1)

    def prep(image):
        arr=np.asarray(image, dtype=np.float32).copy()/255
        return torch.from_numpy(arr).permute(2,0,1)

    bundle=(ToyModel(), prep, {'ready':True,'ai_at_least':.80,'real_at_most':.20})
    # This deliberately makes gray camera-like pixels a false AI according to
    # the toy base model. A guard must abstain instead of inflating confidence.
    gray=np.full((64,64,3), 97, dtype=np.uint8)
    base=classify_custom(gray, bundle, conservative=False)
    safe=classify_custom(gray, bundle, conservative=True)
    assert base.decision.startswith('AI üretimi')
    assert safe.decision.startswith('Belirsiz')
    assert safe.ai_score == pytest.approx(base.ai_score)
    assert len([label for label in safe.labels if label['label'].startswith(('Orijinal','JPEG90','JPEG70'))])==3

    dark=np.zeros((64,64,3),dtype=np.uint8)
    assert classify_custom(dark,bundle).decision.startswith('AI üretimi')
    white=np.full((64,64,3),255,dtype=np.uint8)
    assert classify_custom(white,bundle).decision.startswith('Gerçek fotoğraf')
