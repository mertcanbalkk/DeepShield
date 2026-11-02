"""Conservative, auditable real-photo false-AI protection.

This is a DECISION layer, not a new AI detector. It can only turn an AI
verdict into an abstention; it cannot fix a model's raw scores or increase
sensitivity. An independently collected camera-photo set is required to
calibrate a meaningful site-specific threshold.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

GUARD_VERSION = 1
# An intentionally conservative temporary cutoff, NOT a calibrated probability.
DEFAULT_AI_MIN_SCORE = 0.98
MIN_CALIBRATION_REAL = 50


def ai_evidence(scores: Sequence[float]) -> float:
    """All JPEG variants need to support 'AI': use the lowest AI score."""
    if not scores:
        raise ValueError('AI skor listesi boş.')
    values = [float(s) for s in scores]
    if any(not math.isfinite(s) or s < 0 or s > 1 for s in values):
        raise ValueError('AI skorları sonlu ve [0,1] arasında olmalı.')
    return min(values)


def make_guard(real_score_lists: Sequence[Sequence[float]], model_sha256: str,
               min_images: int = MIN_CALIBRATION_REAL) -> dict:
    """Use the most AI-like *known real photo* as an AI decision veto.

    Threshold is the maximum observed min-variant score among real images.
    On calibration examples, no real item can pass a STRICT greater-than
    comparison. Unseen-camera error rates are NOT guaranteed.
    """
    if len(real_score_lists) < min_images:
        raise ValueError(f'En az {min_images} bağımsız gerçek fotoğraf gerekli; şu an {len(real_score_lists)}.')
    if len(model_sha256) != 64 or any(c not in '0123456789abcdef' for c in model_sha256):
        raise ValueError('Model SHA-256 geçersiz.')
    evidence = [ai_evidence(scores) for scores in real_score_lists]
    cutoff = max(DEFAULT_AI_MIN_SCORE, max(evidence))
    return {
        'schema_version': GUARD_VERSION,
        'ready': True,
        'model_sha256': model_sha256,
        'ai_min_score_must_exceed': cutoff,
        'n_calibration_real': len(evidence),
        'max_real_ai_min_score': max(evidence),
        'fallback_floor': DEFAULT_AI_MIN_SCORE,
        'note': 'Yalnızca bu gerçek fotoğraflarda 0 AI alarmı olacak şekilde seçildi. '
                'Yeni cihaz ve kaynaklar için hata garantisi değildir; bağımsız test gerekir.',
    }


def sha256_file(weights_path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(weights_path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_guard(guard_path: str | Path, weights_path: str | Path) -> dict | None:
    """Fail closed for stale/malformed calibration; caller may show its warning."""
    path = Path(guard_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or data.get('schema_version') != GUARD_VERSION
            or data.get('ready') is not True
            or not isinstance(data.get('n_calibration_real'), int)
            or data['n_calibration_real'] < MIN_CALIBRATION_REAL):
        raise ValueError('Gerçek fotoğraf kalibrasyonu eksik veya uyumsuz.')
    cutoff = data.get('ai_min_score_must_exceed')
    if not isinstance(cutoff, (float, int)) or not math.isfinite(cutoff) or not DEFAULT_AI_MIN_SCORE <= cutoff <= 1:
        raise ValueError('Gerçek fotoğraf koruma eşiği geçersiz.')
    if data.get('model_sha256') != sha256_file(weights_path):
        raise ValueError('AI model ağırlığı değişmiş; gerçek fotoğraf koruma dosyası yeniden oluşturulmalı.')
    return data


def protect_decision(base_decision: str, scores: Sequence[float],
                     guard: dict | None = None, enabled: bool = True) -> tuple[str, str]:
    """AI-positive set is a SUBSET of the original classifier's AI-positive set.

    As a result, for any fixed evaluation set the count of real->AI false
    positives cannot increase (but AI recall / coverage may decrease).
    """
    if not enabled:
        return base_decision, 'Temkinli yanlış-alarm koruması kapalı.'
    if not base_decision.startswith('AI üretimi yönünde'):
        return base_decision, 'Mevcut sınıflandırıcı kararı korundu.'
    evidence = ai_evidence(scores)
    cutoff = DEFAULT_AI_MIN_SCORE
    if guard:
        if not guard.get('ready') or guard.get('schema_version') != GUARD_VERSION:
            raise ValueError('Geçersiz gerçek fotoğraf koruma dosyası.')
        cutoff = max(cutoff, float(guard['ai_min_score_must_exceed']))
    if evidence <= cutoff:
        suffix = ('bağımsız gerçek fotoğraflarla ayarlı koruma' if guard
                  else 'geçici temkinli eşik; harici kalibrasyon yapılmadı')
        return 'Belirsiz (gerçek fotoğrafa yanlış AI alarmını önleme)', (
            f'Minimum JPEG-varyant AI skoru {evidence:.4f}; AI kararı için '
            f'{cutoff:.4f} üstü gerekli ({suffix}). Ham skor değiştirilmedi.')
    suffix = 'Gerçek fotoğraf kalibrasyon eşiği kullanıldı.' if guard else 'Harici gerçek fotoğraf kalibrasyonu henüz yok.'
    return base_decision, ('Tüm JPEG varyantlarında AI kanıtı temkinli eşikten yüksek. ' + suffix)
