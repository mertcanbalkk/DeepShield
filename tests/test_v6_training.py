import sys
from pathlib import Path
from PIL import Image, ImageDraw
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))
from decision import tune_thresholds, decide
from dataset_quality import audit, BKTree, hamming
from evaluate_external import summarize, discover


def test_bktree_search():
    t=BKTree()
    t.add(0, 'zero')
    t.add(15,'fifteen')
    t.add(15,'another')
    assert sorted(label for _,label in t.query(15,0))==['another','fifteen']
    assert sorted(label for _,label in t.query(1,1)) == ['zero']


def test_calibration_needs_real_data():
    out=tune_thresholds([.2,.8],[1,0])
    assert out['ready'] is False
    assert decide([.95],out)['decision'].startswith('Belirsiz')


def test_calibration_separated():
    values=[.9+i*.002 for i in range(30)] + [.05+i*.003 for i in range(30)]
    tags=[0]*30+[1]*30
    out=tune_thresholds(values,tags,max_error=0)
    assert out['ready'] and out['real_at_most'] < out['ai_at_least']
    assert decide([.9,.88,.86],out)['decision'].startswith('AI')
    assert decide([.05,.09,.12],out)['decision'].startswith('Gerçek')
    assert decide([.9,.10],out)['decision'].startswith('Belirsiz')


def test_calibration_no_perfect_reality_claim():
    v=[.65]*30+[.66]*30
    truth=[0]*30+[1]*30
    out=tune_thresholds(v,truth)
    assert out['ready']
    assert decide([.5],out)['decision'].startswith('Gerçek')
    assert decide([.655],out)['decision'].startswith('Belirsiz')


def test_calibration_invalid():
    with pytest.raises(ValueError):
        tune_thresholds([float('nan')],[0])
    with pytest.raises(ValueError):
        decide([1.2],{'ready':False})


def test_audit_reencoded_near_copy(tmp_path):
    for split in ('train','val','test'):
        for label in ('ai','real'):
            (tmp_path/split/label).mkdir(parents=True)
    im=Image.new('RGB',(120,100),(255,255,255))
    draw=ImageDraw.Draw(im)
    draw.rectangle((8,3,51,61),fill=(10,30,220))
    draw.ellipse((70,40,116,95),fill=(205,30,30))
    im.save(tmp_path/'train'/'ai'/'source.png')
    im.save(tmp_path/'test'/'real'/'reencoded.jpg',quality=83)
    out=audit(tmp_path,max_hamming=4)
    assert not out['valid']
    assert any(c['kind']=='near' for c in out['collisions'])


def test_audit_bad_image(tmp_path):
    for split in ('train','val','test'):
        for label in ('ai','real'):
            path=tmp_path/split/label
            path.mkdir(parents=True)
            Image.new('RGB',(30,30),(ord(split[0])*3,34,55)).save(path/'sample.png')
    out=audit(tmp_path)
    assert out['counts']['val/real']==1


def test_summary_counts_abstentions():
    stats=summarize([(0,'ai'),(0,'uncertain'),(0,'real'),(1,'real'),(1,'uncertain'),(1,'ai')])
    assert stats['ai_recall_over_all']==1/3
    assert stats['false_ai_rate_over_all_real']==1/3
    assert stats['coverage']==4/6
