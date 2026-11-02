"""Validation-calibrated, conservative AI image decisions.

Positive label is class 0 ('ai'). Raw model outputs are NOT forensic
probabilities; thresholds have empirical meaning only on held-out data.
"""
from __future__ import annotations

import math
from typing import Sequence


def _clean(scores: Sequence[float], labels: Sequence[int]):
    if len(scores) != len(labels) or not scores:
        raise ValueError('Skorlar ve etiketler aynı sayıda olmalı, boş olamaz.')
    for score, label in zip(scores, labels):
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1 or label not in (0, 1):
            raise ValueError('Skor [0,1] arasında sonlu sayı; etiket 0 (AI) veya 1 (real) olmalı.')


def tune_thresholds(scores: Sequence[float], labels: Sequence[int], max_error: float = .05,
                    min_per_class: int = 20) -> dict:
    """Choose abstaining cutoffs from held-out validation, never test data.

    AI verdict if score >= ai_at_least. Real verdict if score <= real_at_most.
    No cutoffs are issued with insufficient validation observations.

    The empirical bounds are descriptive only (not statistical guarantees).
    """
    _clean(scores, labels)
    if not 0 <= max_error < .5:
        raise ValueError('max_error 0 ile 0.5 arasında olmalı.')
    ai = sorted(float(s) for s, y in zip(scores, labels) if y == 0)
    real = sorted(float(s) for s, y in zip(scores, labels) if y == 1)
    if len(ai) < min_per_class or len(real) < min_per_class:
        return {'ready': False, 'reason': 'Kalibrasyon için her sınıfta en az '
                f'{min_per_class} doğrulama örneği olmalı.', 'n_ai':len(ai), 'n_real':len(real)}
    # Choose the most permissive high threshold with empirical FPR <= max_error.
    candidates = sorted({0.0, 1.0, *(min(1., s + 1e-8) for s in real), *real})
    hi = next((t for t in candidates if sum(s >= t for s in real) / len(real) <= max_error), None)
    # Choose the most permissive low threshold with AI->real error <= max_error.
    candidates_low = sorted({0.0, 1.0, *(max(0., s - 1e-8) for s in ai), *ai}, reverse=True)
    lo = next((t for t in candidates_low if sum(s <= t for s in ai) / len(ai) <= max_error), None)
    if hi is None or lo is None:
        return {'ready':False, 'reason':'Kalibrasyon kümesi güvenli karar bantları üretmedi.',
                'n_ai':len(ai), 'n_real':len(real)}
    if lo >= hi:
        # A perfectly separated validation set can have lo >= hi. A single
        # decision boundary is safer than contradictory classifications.
        boundary = (lo + hi) / 2
        lo, hi = boundary - 1e-6, boundary + 1e-6
    result = {'ready':True,'ai_at_least':round(hi, 8),'real_at_most':round(lo, 8),
              'max_empirical_error':max_error, 'n_ai':len(ai),'n_real':len(real),
              'val_false_ai_rate':sum(s >= hi for s in real) / len(real),
              'val_false_real_rate':sum(s <= lo for s in ai) / len(ai),
              'warning':'Yalnızca bu doğrulama dağılımı için ampirik hata; gerçek dünya garantisi değildir.'}
    return result


def decide(scores: Sequence[float], thresholds: dict) -> dict:
    """Verdict is stable only if ALL image variants agree on the same class."""
    if not scores or any(not math.isfinite(float(s)) or not 0 <= s <= 1 for s in scores):
        raise ValueError('En az bir geçerli [0,1] skor gerekli.')
    lo, hi = min(scores), max(scores)
    if not thresholds.get('ready'):
        verdict = 'Belirsiz (kalibrasyon yok)'
    elif lo >= thresholds['ai_at_least']:
        verdict = 'AI üretimi yönünde bulgu'
    elif hi <= thresholds['real_at_most']:
        verdict = 'Gerçek fotoğraf yönünde bulgu'
    else:
        verdict = 'Belirsiz (görüntü varyantlarında tutarsızlık veya ara skor)'
    return {'decision':verdict, 'scores':list(scores), 'min_score':float(lo),
            'max_score':float(hi), 'mean_score':sum(scores)/len(scores),
            'number_of_variants':len(scores)}
