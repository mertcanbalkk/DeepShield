import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))

import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='Optional large AI dependency not installed')
from model_adapters import classify_ai_modern


class ConstantPredictor(torch.nn.Module):
    def __init__(self, p_real):
        super().__init__()
        self.logit=float(np.log(p_real/(1-p_real)))
    def forward(self, x):
        assert tuple(x.shape)==(1,3,256,256)
        return torch.tensor([[self.logit]])


def prepare(p_real):
    return ConstantPredictor(p_real), lambda im: torch.zeros((3,256,256))


def test_modern_author_thresholds():
    im=np.zeros((240,320,3), np.uint8)
    fake=classify_ai_modern(im,prepare(.85))
    uncertain=classify_ai_modern(im,prepare(.92))
    real=classify_ai_modern(im,prepare(.96))
    assert 'AI' in fake.decision
    assert 'belirsiz' in uncertain.decision
    assert 'gerçek' in real.decision
    assert abs(fake.ai_score-.15)<.0001
