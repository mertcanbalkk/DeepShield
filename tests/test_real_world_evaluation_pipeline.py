"""Offline integration checks with a toy predictor, not real model accuracy."""
from __future__ import annotations
import json
import types
from pathlib import Path
import numpy as np
from PIL import Image


def test_calibrate_and_evaluate_without_training_or_download(tmp_path, monkeypatch):
    from training.calibrate_real_guard import calibrate
    from training.evaluate_external import evaluate_folder
    from training.real_guard import load_guard

    real_dir = tmp_path/'real_calibration'
    real_dir.mkdir()
    for i in range(50):
        Image.new('RGB',(18,17),(20+i*4,)*3).save(real_dir/f'real_{i:02}.png')
    weight=tmp_path/'unchanged.pt'
    weight.write_bytes(b'dummy checkpoint identity; no model loaded')
    guard_file=tmp_path/'outputs'/'guard.json'

    class FakeClassifier:
        def __init__(self, rgb):
            # Strikingly wrong baseline: high AI scores for dark real pixels.
            self.ai_score = float(rgb.mean()/255)
            self.decision = ('AI üretimi yönünde bulgu' if self.ai_score >= .2
                             else 'Gerçek fotoğraf yönünde bulgu')
            self.labels = [
                {'label': 'Orijinal ham AI skoru','score':self.ai_score},
                {'label': 'JPEG90 ham AI skoru','score':self.ai_score},
                {'label': 'JPEG70 ham AI skoru','score':self.ai_score},
            ]

    fake=types.ModuleType('model_adapters')
    fake.load_custom_classifier=lambda _path: object()
    fake.classify_custom=lambda rgb, _bundle, conservative=False: FakeClassifier(rgb)
    monkeypatch.setitem(__import__('sys').modules, 'model_adapters', fake)

    result=calibrate(real_dir,weight,guard_file)
    assert result['n_calibration_real']==50
    guard=load_guard(guard_file,weight)
    assert len(guard['calibration_sha256'])==50

    external=tmp_path/'external'
    (external/'ai').mkdir(parents=True)
    (external/'real').mkdir(parents=True)
    Image.new('RGB',(18,17),(255,255,255)).save(external/'ai'/'ai.png')
    Image.new('RGB',(18,17),(121,121,121)).save(external/'real'/'real.png')
    # The calibration duplicate is NOT allowed into the external test.
    (external/'real'/'dup.png').write_bytes((real_dir/'real_00.png').read_bytes())
    report=evaluate_folder(external, weight,guard_path=guard_file)
    assert len(report['excluded_calibration_duplicates'])==1
    assert report['baseline']['overall']['real_false_ai']==1
    assert report['conservative']['overall']['real_false_ai']==0
    assert report['conservative']['overall']['real_uncertain']==1
    assert report['conservative']['overall']['ai_correct']==1
    assert report['conservative']['overall']['coverage']==.5
    assert len(report['per_image'])==2
