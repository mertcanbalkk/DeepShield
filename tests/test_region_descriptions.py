import numpy as np
import pytest
from forensic_engine import marked_regions
from presentation_metrics import position_label, region_summaries, summarize_percentages, text_report, describe_clone_candidates

@pytest.mark.parametrize('xywh,expected', [
    ((0,0,20,20),'üst sol köşede'),
    ((80,0,20,20),'üst sağ köşede'),
    ((0,80,20,20),'alt sol köşede'),
    ((80,80,20,20),'alt sağ köşede'),
    ((40,40,20,20),'merkezde'),
    ((0,40,20,20),'sol tarafta'),
    ((40,0,20,20),'üst ortada'),
])
def test_position_grid(xywh, expected):
    assert position_label(*xywh,100,100) == expected


def test_region_area_uses_mask_area_not_rectangle():
    r = region_summaries([{'x':0,'y':0,'width':20,'height':20,'area':100}],10000,100,100)
    assert r[0]['bolge_alani_yuzde'] == 1.0
    assert r[0]['konum'] == 'üst sol köşede'
    assert 'AI kullanımı kanıtı değildir' in r[0]['aciklama']


def test_no_localizer_is_unmeasured_not_unmodified():
    report = text_report('AI yönünde',93.24,None,[])
    assert 'belirlenemedi' in report
    assert 'Ölçülemedi' in report


def test_localizer_boxes_align_with_region_text():
    image = np.full((160,180,3), 120, dtype=np.uint8)
    score = np.zeros((160,180), dtype=np.float32)
    score[3:43,3:40] = .9
    score[120:150,138:169] = .9
    marked, mask, regions = marked_regions(image,score,threshold=.65,min_area_fraction=.001)
    summaries = region_summaries(regions,mask.size,180,160)
    assert len(summaries) == 2
    assert summaries[0]['konum'] == 'üst sol köşede'
    assert summaries[1]['konum'] == 'alt sağ köşede'
    assert np.any(marked != image)
    assert summarize_percentages(.9,'AI yönünde',mask).altered_area_percent > 0


def test_out_of_bounds_rejected():
    with pytest.raises(ValueError):
        position_label(99,0,20,20,100,100)


def test_clone_locations_are_described_without_attribution():
    candidate = [{'area_a':(0,0,20,20),'area_b':(80,80,100,100)}]
    desc = describe_clone_candidates(candidate,100,100)
    assert 'üst sol' in desc[0] and 'alt sağ' in desc[0]
    assert 'hangisinin kaynak/hedef olduğu bilinmez' in desc[0]
