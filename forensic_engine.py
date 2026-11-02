"""DeepShield 4: explainable forensic SIGNALS on a single image.

No score/map in this module is calibrated proof of manipulation or AI authorship.
Raw forensic evidence is always kept separate from learned model outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1800
MAX_DECODE_PIXELS = 36_000_000
Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS


@dataclass
class ForensicResult:
    rgb: np.ndarray
    filename: str
    sha256: str
    metadata: dict[str, str]
    maps: dict[str, np.ndarray]
    available: dict[str, bool]
    explanations: dict[str, str]
    clone_pairs: list[dict[str, Any]]
    hypotheses: list[str]
    warnings: list[str]


def read_photo(data: bytes) -> np.ndarray:
    if not data or len(data) > 35 * 1024 * 1024:
        raise ValueError('Görsel boş veya 35 MB boyut sınırını aşıyor.')
    with Image.open(BytesIO(data)) as img:
        img.verify()
    with Image.open(BytesIO(data)) as img:
        if img.width * img.height > MAX_DECODE_PIXELS:
            raise ValueError('Görüntü piksel sınırını aşıyor (en fazla 36 MP).')
        img = ImageOps.exif_transpose(img).convert('RGB')
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        return np.asarray(img).copy()


def get_metadata(data: bytes) -> dict[str, str]:
    with Image.open(BytesIO(data)) as im:
        exif = im.getexif()
        info = im.info
        result = {
            'format': str(im.format or '?'),
            'orijinal_çözünürlük': f'{im.width} × {im.height}',
            'exif': 'Var' if len(exif) else 'Yok',
            'kamera': str(exif.get(272, '—'))[:200],
            'yazılım': str(exif.get(305, '—'))[:200],
            'tarih': str(exif.get(36867, exif.get(306, '—')))[:200],
        }
        for k in ('parameters', 'prompt', 'Comment', 'comment', 'Software'):
            if k in info:
                result[f'ek:{k}'] = str(info[k])[:300]
        return result


def gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def robust_scale(x: np.ndarray, lo_q: float = 15, hi_q: float = 98) -> np.ndarray:
    """Display scaling ONLY. Never present normalized map as a probability."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0, posinf=0, neginf=0)
    low, high = np.percentile(x, (lo_q, hi_q))
    if high - low < 1e-7:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0, 1).astype(np.float32)


def ela_signal(rgb: np.ndarray, jpeg_quality: int = 90) -> np.ndarray:
    """Recompression error, appropriate mostly for JPEG; NOT a change detector."""
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format='JPEG', quality=jpeg_quality, subsampling=0)
    reencoded = np.asarray(Image.open(BytesIO(buffer.getvalue())).convert('RGB'), dtype=np.float32)
    e = np.mean(np.abs(rgb.astype(np.float32) - reencoded), axis=2)
    return cv2.GaussianBlur(e, (0, 0), 2)


def residual_signal(rgb: np.ndarray) -> np.ndarray:
    """Relative spatial noise residual; normalized against local texture/edges."""
    g = gray(rgb).astype(np.float32)
    background = cv2.GaussianBlur(g, (5, 5), 1.15)
    noise = g - background
    noise_energy = np.sqrt(cv2.GaussianBlur(noise * noise, (0, 0), 5) + 1e-6)
    grad_x = cv2.Sobel(background, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(background, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (0, 0), 5)
    # Texture-normalized local noise; naturally flat regions can still be anomalous.
    normalized = noise_energy / (3 + 0.22 * grad)
    scale = max(13, min(g.shape) // 25)
    local = cv2.GaussianBlur(normalized, (0, 0), scale)
    discrepancy = np.abs(normalized - local)
    return discrepancy


def block_signal(rgb: np.ndarray) -> np.ndarray:
    """Simple 8-pixel grid discontinuity visualization; content can cause peaks."""
    g = gray(rgb).astype(np.float32)
    h, w = g.shape
    if h < 40 or w < 40:
        return np.zeros_like(g)
    d = np.zeros_like(g)
    for x in range(8, w - 1, 8):
        jump = np.abs(g[:, x] - g[:, x - 1])
        smooth = 0.5 * (np.abs(g[:, x - 1] - g[:, x - 2]) + np.abs(g[:, x + 1] - g[:, x]))
        d[:, max(0, x - 2):x + 2] = np.maximum(d[:, max(0, x - 2):x + 2], np.maximum(jump - smooth, 0)[:, None])
    for y in range(8, h - 1, 8):
        jump = np.abs(g[y, :] - g[y - 1, :])
        smooth = 0.5 * (np.abs(g[y - 1, :] - g[y - 2, :]) + np.abs(g[y + 1, :] - g[y, :]))
        d[max(0, y - 2):y + 2, :] = np.maximum(d[max(0, y - 2):y + 2, :], np.maximum(jump - smooth, 0)[None, :])
    return cv2.GaussianBlur(d, (0, 0), 5)


def _candidate_clone_matches(g: np.ndarray, max_keypoints: int = 4000) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """ORB descriptor pairs far apart; repeated patterns can mimic clones."""
    h, w = g.shape
    orb = cv2.ORB_create(nfeatures=max_keypoints, fastThreshold=8, edgeThreshold=16)
    kp, desc = orb.detectAndCompute(g, None)
    if desc is None or len(kp) < 18:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Several nearby descriptor neighbors can be self-matches: exclude spatially local.
    knn = bf.knnMatch(desc, desc, k=min(12, len(desc)))
    min_spacing = max(28, min(h, w) * 0.09)
    matches = []
    seen = set()
    for i, neighbors in enumerate(knn):
        for match in neighbors:
            j = match.trainIdx
            if j == i or match.distance > 24:
                continue
            a, b = np.asarray(kp[i].pt), np.asarray(kp[j].pt)
            if np.linalg.norm(a - b) < min_spacing:
                continue
            # Confirm local photometric agreement, otherwise repetitive structures
            # (windows, road markings, text) can give a huge number of ORB matches.
            ax, ay, bx, by = round(a[0]), round(a[1]), round(b[0]), round(b[1])
            radius = 10
            if (min(ax, ay, bx, by) < radius or ax + radius >= w or bx + radius >= w
                    or ay + radius >= h or by + radius >= h):
                continue
            patch_a = g[ay-radius:ay+radius+1, ax-radius:ax+radius+1].astype(np.float32)
            patch_b = g[by-radius:by+radius+1, bx-radius:bx+radius+1].astype(np.float32)
            # Ignore low-entropy patterns such as identical ruled notebook lines.
            if min(float(patch_a.std()), float(patch_b.std())) < 17.0:
                continue
            if float(np.mean(np.abs(patch_a - patch_b))) > 8.0:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            matches.append((tuple(a), tuple(b), float(match.distance)))
            break
    return matches


def copy_move_signal(rgb: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rough copy-move candidate clusters; cannot prove clone/identify source vs destination."""
    g = gray(rgb)
    h, w = g.shape
    pairs = _candidate_clone_matches(g)
    # Quantize displacement 12 px; canonical sign makes reversed matches consistent.
    buckets: dict[tuple[int, int], list] = {}
    for a, b, distance in pairs:
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx < 0 or (dx == 0 and dy < 0):
            a, b, dx, dy = b, a, -dx, -dy
        key = (round(dx / 12), round(dy / 12))
        buckets.setdefault(key, []).append((a, b, distance))
    map_float = np.zeros((h, w), np.float32)
    clusters = []
    for key, matches in sorted(buckets.items(), key=lambda t: -len(t[1]))[:16]:
        # Unique spatial coverage avoids declaring repetitive high-density edges a copy.
        if len(matches) < 7:
            continue
        source_cells = {(int(a[0] // 22), int(a[1] // 22)) for a, _, _ in matches}
        dest_cells = {(int(b[0] // 22), int(b[1] // 22)) for _, b, _ in matches}
        if len(source_cells) < 5 or len(dest_cells) < 5:
            continue
        # Actual displacement variation, rather than quantized bucket alone.
        vectors = np.array([[b[0] - a[0], b[1] - a[1]] for a, b, _ in matches], np.float32)
        center = np.median(vectors, axis=0)
        robust = [m for m, vec in zip(matches, vectors) if np.linalg.norm(vec - center) < 15]
        if len(robust) < 7:
            continue
        def bbox(points):
            p = np.asarray(points, np.float32)
            x1, y1 = p.min(axis=0)
            x2, y2 = p.max(axis=0)
            pad = max(14, int(min(h, w) * 0.015))
            return (max(0, int(x1)-pad), max(0, int(y1)-pad), min(w, int(x2)+pad), min(h, int(y2)+pad))
        box_a = bbox([a for a, _, _ in robust])
        box_b = bbox([b for _, b, _ in robust])
        # Reject massive, sparse groupings: often repeated scene structure.
        if any((box[2]-box[0]) > .55*w or (box[3]-box[1]) > .55*h for box in (box_a,box_b)):
            continue
        for box in (box_a, box_b):
            x1, y1, x2, y2 = box
            cv2.rectangle(map_float, (x1, y1), (x2, y2), 1, -1)
        clusters.append({'pairs': len(robust), 'area_a': box_a, 'area_b': box_b,
                         'shift_pixels': (int(round(center[0])), int(round(center[1])))})
    if np.any(map_float):
        map_float = cv2.GaussianBlur(map_float, (0, 0), 3)
    return map_float, clusters[:4]


def _hint_score(map_float: np.ndarray) -> float:
    # Descriptive only: do not interpret as manipulation probability.
    if map_float.size < 1:
        return 0.0
    return float(np.percentile(map_float, 95))


def analyze_image(data: bytes, filename: str = 'gorsel') -> ForensicResult:
    rgb = read_photo(data)
    meta = get_metadata(data)
    h, w = rgb.shape[:2]
    small_image = min(h, w) < 192
    jpeg = meta['format'].upper() == 'JPEG'
    ela = robust_scale(ela_signal(rgb)) if jpeg else np.zeros((h, w), np.float32)
    noise = robust_scale(residual_signal(rgb))
    blocks = robust_scale(block_signal(rgb)) if jpeg else np.zeros((h, w), np.float32)
    clones, clone_pairs = copy_move_signal(rgb)
    maps = {'ela': ela, 'noise': noise, 'jpeg_blocks': blocks, 'copy_move': clones}
    available = {'ela': jpeg, 'noise': True, 'jpeg_blocks': jpeg, 'copy_move': True}
    hypotheses: list[str] = []
    if clone_pairs:
        hypotheses.append('Birbirine benzeyen ve yer değiştirmiş doku/özellik grupları bulundu (klonlama ihtimali).')
    # Software string can be written arbitrarily: record only, do not convict.
    software = meta.get('yazılım', '').lower()
    if software not in ('—', '', 'none'):
        hypotheses.append('Dosya metaverisinde yazılım bilgisi mevcut; bunun varlığı tek başına tahrifat kanıtı değildir.')
    warnings = ['Bu haritalar delil değil, insan tarafından incelenecek adli ipuçlarıdır.']
    if not jpeg:
        warnings.append('Kaynak JPEG olmadığı için ELA ve JPEG 8×8 blok kontrolü uygulanmadı.')
    if small_image:
        warnings.append('Görselin çözünürlüğü düşük; yerel analizler güvenilir olmayabilir.')
    if meta.get('exif') == 'Yok':
        warnings.append('EXIF yokluğu manipülasyon kanıtı değildir; sosyal medya/mesajlaşma EXIF silebilir.')
    return ForensicResult(
        rgb=rgb, filename=filename, sha256=sha256(data).hexdigest(), metadata=meta,
        maps=maps, available=available,
        explanations={
            'ela': 'JPEG tekrar sıkıştırma izleri: kaydetme, kalite ve doku nedeniyle değişebilir.',
            'noise': 'Bölgesel yüksek frekans/gürültü uyuşmazlığı: odak, ışık ve doku etkiler.',
            'jpeg_blocks': 'JPEG 8×8 blok sınırı anomalileri: JPEG kalitesi ve doğal detay etkiler.',
            'copy_move': 'Uzaktaki alanlarda benzer ORB özellikleri: tekrar eden dokular yanlış alarm oluşturabilir.'
        },
        clone_pairs=clone_pairs, hypotheses=hypotheses, warnings=warnings
    )


def heatmap(rgb: np.ndarray, score: np.ndarray, opacity: float = 0.47) -> np.ndarray:
    if score.shape != rgb.shape[:2]:
        score = cv2.resize(score, (rgb.shape[1], rgb.shape[0]))
    score = np.clip(score, 0, 1).astype(np.float32)
    cm = cv2.cvtColor(cv2.applyColorMap(np.uint8(score*255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - opacity)*rgb.astype(np.float32) + opacity*cm.astype(np.float32), 0, 255))


def marked_regions(rgb: np.ndarray, score: np.ndarray, threshold: float = .70,
                   min_area_fraction: float = .0008, max_regions: int = 12) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Draw user-visible tentative regions; score is 0..1 display or model output."""
    h, w = rgb.shape[:2]
    if score.shape != (h, w):
        score = cv2.resize(np.asarray(score, np.float32), (w, h))
    mask = (np.clip(score, 0, 1) >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    kept = np.zeros((h, w), np.uint8)
    for idx in range(1, n):
        x, y, bw, bh, area = map(int, stats[idx])
        if area >= max(50, int(h*w*min_area_fraction)):
            regions.append({'x':x,'y':y,'width':bw,'height':bh,'area':area,'label':idx})
    regions = sorted(regions, key=lambda r: -r['area'])[:max_regions]
    for r in regions:
        kept[labels == r['label']] = 255
    out = rgb.copy()
    # Do NOT dim original excessively: analyst must inspect its features.
    active = kept > 0
    out[active] = np.clip(rgb[active]*0.48 + np.array([255, 42, 40])*0.52, 0, 255).astype(np.uint8)
    cnt, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 238, 50), max(2, int(min(h,w)*.003)), cv2.LINE_AA)
    for i, r in enumerate(regions, start=1):
        x, y, bw, bh = r['x'], r['y'], r['width'], r['height']
        cv2.rectangle(out, (x, y), (x+bw, y+bh), (255, 54, 54), 2)
        cv2.putText(out, f'ALAN {i}', (x, max(20,y-8)), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 255, 70), 2, cv2.LINE_AA)
    return out, kept, [{k:v for k,v in r.items() if k!='label'} for r in regions]
