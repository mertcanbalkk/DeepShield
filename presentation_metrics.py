"""User-facing measurements for whole-image AI scores and tampering masks.

An AI class score is not a percentage of AI-authored pixels.
A forgery localization mask is not proof of use of AI rather than other editors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnalysisPercentages:
    ai_class_score_percent: float | None
    ai_decision: str
    altered_area_percent: float | None
    altered_area_kind: str


def summarize_percentages(ai_score: float | None, decision: str | None,
                          localized_mask: np.ndarray | None = None) -> AnalysisPercentages:
    pct = None
    if ai_score is not None:
        score = float(ai_score)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Geçersiz AI skoru; [0,1] aralığında olmalı.')
        pct = round(score * 100, 2)
    if localized_mask is None:
        area_pct = None
        kind = 'Yerelleştirme modeli çalışmadı'
    else:
        mask = np.asarray(localized_mask)
        if mask.ndim != 2 or mask.size == 0 or mask.dtype.kind not in 'biuf':
            raise ValueError('Yerelleştirme maskesi geçerli 2D sayısal dizi olmalı.')
        if not np.isfinite(mask).all() or not np.all(np.isin(mask, (0, 1, 255))):
            raise ValueError('Alan yüzdesi için eşiklenmiş 0/1/255 maske gerekir.')
        area_pct = round(100 * np.count_nonzero(mask) / mask.size, 2)
        kind = 'Modelin şüpheli işaretlediği piksel alanı (AI kanıtı değil)'
    return AnalysisPercentages(pct, decision or 'Belirsiz', area_pct, kind)


def text_for_ai_percent(value: float | None) -> str:
    return 'Ölçülemedi' if value is None else f'%{value:.2f}'


def position_label(x: int, y: int, width: int, height: int,
                   image_width: int, image_height: int) -> str:
    """Human-readable 3x3 grid location based on bounding-box center.

    This says WHERE the localizer flagged pixels, not WHAT object it found.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError('Görsel boyutları pozitif olmalı.')
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError('Bölge koordinatları görsel sınırları içinde olmalı.')
    u = (x + width / 2) / image_width
    v = (y + height / 2) / image_height
    horizontal = 'sol' if u < 1 / 3 else ('sağ' if u >= 2 / 3 else 'orta')
    vertical = 'üst' if v < 1 / 3 else ('alt' if v >= 2 / 3 else 'orta')
    if vertical == 'orta' and horizontal == 'orta':
        return 'merkezde'
    if vertical == 'orta':
        return f'{horizontal} tarafta'
    if horizontal == 'orta':
        return f'{vertical} ortada'
    return f'{vertical} {horizontal} köşede'


def region_summaries(regions: list[dict[str, Any]], total_pixels: int,
                     image_width: int | None = None, image_height: int | None = None) -> list[dict[str, Any]]:
    """Describe localizer components using real mask area, not bbox area.

    image width/height are optional to preserve compatibility with old callers;
    pass both for image-relative location descriptions.
    """
    if total_pixels <= 0:
        raise ValueError('Toplam piksel sayısı pozitif olmalı.')
    if (image_width is None) != (image_height is None):
        raise ValueError('Genişlik ve yükseklik birlikte verilmeli.')
    if image_width is not None and (image_width <= 0 or image_height <= 0 or image_width * image_height != total_pixels):
        raise ValueError('Görsel boyutları toplam piksel alanıyla uyuşmalı.')

    result = []
    for i, reg in enumerate(regions, start=1):
        x, y, width, height = (int(reg[key]) for key in ('x','y','width','height'))
        area = int(reg['area'])
        if area <= 0 or area > width * height or width <= 0 or height <= 0:
            raise ValueError('Bölge alanı geçersiz.')
        pos = (position_label(x, y, width, height, image_width, image_height)
               if image_width is not None else 'Konum belirtilmedi')
        result.append({
            'alan_no': i,
            'konum': pos,
            'aciklama': f'ALAN {i}: Görüntünün {pos} şüpheli bir bölge işaretlendi; bu bir manipülasyon tahminidir, AI kullanımı kanıtı değildir.',
            'x': x,
            'y': y,
            'genislik': width,
            'yukseklik': height,
            'bolge_alani_yuzde': round(100 * area / total_pixels, 2),
        })
    return result


def text_report(ai_decision: str, ai_score_pct: float | None, suspicious_area_pct: float | None,
                region_details: list[dict[str, Any]]) -> str:
    """Plain-text report; deliberately refuses to attribute edited regions to AI."""
    lines = [
        'DEEPSHIELD — TEK GÖRSEL ÖN İNCELEME',
        f'Bütün görüntü sınıflandırması: {ai_decision}',
        f'AI sınıf puanı: {text_for_ai_percent(ai_score_pct)} (piksel oranı veya kesinlik değildir)',
        f'Şüpheli manipülasyon alanı: {text_for_ai_percent(suspicious_area_pct)} (AI ile değiştirme oranı değildir)',
        '',
        'ŞÜPHELİ BÖLGELER',
    ]
    if suspicious_area_pct is None:
        lines.append('Bölgesel model çalışmadı: hangi yerin değiştiği belirlenemedi.')
    elif not region_details:
        lines.append('Seçili eşikte raporlanabilir şüpheli bölge bulunmadı. Bu, özgünlük kanıtı değildir.')
    else:
        for r in region_details:
            lines.append(f"ALAN {r['alan_no']}: {r['konum']}; x={r['x']}, y={r['y']}, "
                         f"boyut={r['genislik']}×{r['yukseklik']} piksel, "
                         f"görüntüde kaplanan maske alanı %{r['bolge_alani_yuzde']:.2f}.")
    lines += ['', 'NOT: Model şüphe tahmini yapar; bölgelerin AI ile değiştirildiğini veya manipülasyon türünü kanıtlamaz.']
    return '\n'.join(lines)


def describe_clone_candidates(clone_pairs: list[dict[str, Any]], image_width: int, image_height: int) -> list[str]:
    """Spatial descriptions only: cannot infer which similar area was copied.

    `area_a`/`area_b` contain (x1,y1,x2,y2) from forensic_engine.
    """
    descriptions = []
    for i, pair in enumerate(clone_pairs, start=1):
        parts = []
        for key in ('area_a', 'area_b'):
            x1, y1, x2, y2 = (int(v) for v in pair[key])
            loc = position_label(x1, y1, x2-x1, y2-y1, image_width, image_height)
            parts.append(f'{loc} (x={x1}–{x2}, y={y1}–{y2})')
        descriptions.append(f'Benzer alan çifti {i}: {parts[0]} ile {parts[1]}. '
                            'Bunlardan hangisinin kaynak/hedef olduğu bilinmez; doğal tekrarlanan desen de olabilir.')
    return descriptions
