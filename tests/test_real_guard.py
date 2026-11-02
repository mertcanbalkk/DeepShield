"""Decision-policy tests use synthetic scores, NOT ground-truth model accuracy."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.real_guard import (DEFAULT_AI_MIN_SCORE, MIN_CALIBRATION_REAL,
                                 ai_evidence, make_guard, load_guard,
                                 protect_decision, sha256_file)

SHA = 'a'*64
AI = 'AI üretimi yönünde bulgu'
REAL = 'Gerçek fotoğraf yönünde bulgu'


def test_fallback_downgrades_borderline_ai_without_faking_score():
    verdict, note = protect_decision(AI, [.93, .91, .94])
    assert verdict.startswith('Belirsiz')
    assert '.91' in note
    assert protect_decision(AI, [.999,.998,.999])[0] == AI
    assert protect_decision(REAL, [.95,.8,.9])[0] == REAL
    assert protect_decision('Belirsiz (görüntü)', [.95,.8,.9])[0].startswith('Belirsiz')
    assert protect_decision(AI,[.91,.94],enabled=False)[0] == AI


def test_guard_uses_most_ai_like_real_and_never_promises_probability():
    score_lists = [[i/60]*3 for i in range(MIN_CALIBRATION_REAL)]
    score_lists[-1] = [.999,.999,.999]
    guard=make_guard(score_lists,SHA)
    assert guard['ready']
    assert guard['n_calibration_real']==MIN_CALIBRATION_REAL
    assert guard['ai_min_score_must_exceed']==.999
    assert protect_decision(AI,[.999]*3,guard)[0].startswith('Belirsiz')
    assert protect_decision(AI,[.9999]*3,guard)[0] == AI


def test_guard_always_only_removes_ai_decisions():
    guard=make_guard([[i/70]*3 for i in range(50)],SHA)
    scores = [[x,x*.98, min(x*1.005,1)] for x in [0,.1,.25,.5,.85,.95,.975,.98,.99,.995,.999,1.]]
    for triplet in scores:
        for base in (AI, REAL, 'Belirsiz'):
            protected, _=protect_decision(base,triplet,guard)
            assert not protected.startswith('AI üretimi') or base.startswith('AI üretimi')


def test_bad_calibration_refused(tmp_path):
    with pytest.raises(ValueError):
        make_guard([[.5,.5,.5]]*10,SHA)
    with pytest.raises(ValueError):
        ai_evidence([float('nan')])
    weight=tmp_path/'w.pt'
    weight.write_bytes(b'weights-one')
    good_sha=sha256_file(weight)
    guard=make_guard([[.5]*3 for _ in range(50)],good_sha)
    guard_file=tmp_path/'guard.json'
    guard_file.write_text(json.dumps(guard),encoding='utf-8')
    assert load_guard(guard_file,weight)['ready']
    weight.write_bytes(b'weights-two')
    with pytest.raises(ValueError,match='değişmiş'):
        load_guard(guard_file,weight)
    assert load_guard(tmp_path/'missing.json',weight) is None


def test_all_real_calibration_examples_are_not_classified_as_ai():
    values = [.51 + .48*(i/49) for i in range(50)]
    guard=make_guard([[v,v,v] for v in values],SHA)
    predictions=[protect_decision(AI,[v]*3,guard)[0] for v in values]
    assert all(not p.startswith('AI üretimi') for p in predictions)
