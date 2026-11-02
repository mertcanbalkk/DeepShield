import numpy as np
import pytest
from presentation_metrics import summarize_percentages, text_for_ai_percent, region_summaries


def test_missing_localizer_percent_is_unavailable_not_zero():
    res = summarize_percentages(.9324, 'AI üretimi yönünde bulgu')
    assert res.ai_class_score_percent == 93.24
    assert res.altered_area_percent is None
    assert text_for_ai_percent(res.altered_area_percent) == 'Ölçülemedi'


def test_binary_mask_percent_is_area_not_score():
    m = np.zeros((100,100), np.uint8)
    m[:20,:25] = 255
    res = summarize_percentages(.9324, 'AI üretimi yönünde bulgu', m)
    assert res.ai_class_score_percent == 93.24
    assert res.altered_area_percent == 5.0
    assert res.altered_area_kind.startswith('Modelin şüpheli')


def test_valid_all_zero_mask_means_zero_area():
    res = summarize_percentages(None, None, np.zeros((64,64),dtype=np.uint8))
    assert res.altered_area_percent == 0.0
    assert res.ai_class_score_percent is None


def test_raw_prob_map_refused_as_binary_mask():
    with pytest.raises(ValueError):
        summarize_percentages(.8,'?',np.full((10,10), .7, dtype=np.float32))


def test_invalid_score_refused():
    with pytest.raises(ValueError):
        summarize_percentages(1.3, '?')


def test_region_summaries():
    regs = [{'x':30,'y':40,'width':10,'height':5,'area':50}]
    assert region_summaries(regs,1000)[0]['bolge_alani_yuzde'] == 5.0
