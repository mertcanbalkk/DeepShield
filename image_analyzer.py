"""DeepShield v2 - transparent image forensics, not an AI-origin oracle.

Modes:
 * with reference: geometric alignment + pixel/color difference map.
 * without reference: ELA and noise/texture visualization (non-conclusive).

Requires: numpy, opencv-python-headless, Pillow.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


MAX_SIDE = 2200


def load_rgb(data: bytes) -> np.ndarray:
    """Honor EXIF orientation, normalize RGB, limit very large images."""
    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        return np.array(im)


def bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_map(x: np.ndarray, low: float = 5, high: float = 99) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi - lo < 1e-6:
        return np.zeros(x.shape, np.uint8)
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def color_heatmap(rgb: np.ndarray, heat: np.ndarray, opacity: float = 0.46) -> np.ndarray:
    cm = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    base = bgr(rgb)
    overlay = cv2.addWeighted(base, 1 - opacity, cm, opacity, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def _remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == i] = 255
    return kept


def _bbox_regions(mask: np.ndarray, limit: int = 20) -> list[dict[str, int]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        result.append({"x": x, "y": y, "width": w, "height": h, "area": area})
    return sorted(result, key=lambda r: r["area"], reverse=True)[:limit]


def highlight_changes(rgb: np.ndarray, mask: np.ndarray,
                      regions: list[dict[str, int]]) -> np.ndarray:
    """Dramatically highlight measured differences, not guessed AI edits.

    Muted backdrop + solid translucent red mask + white/yellow contours
    makes changed areas legible even on small screens.
    """
    out = (rgb.astype(np.float32) * 0.63).astype(np.uint8)
    active = mask > 0
    red = np.array([255, 32, 45], dtype=np.float32)
    out[active] = (0.25 * rgb[active].astype(np.float32) + 0.75 * red).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 245, 75), 3, lineType=cv2.LINE_AA)
    for i, r in enumerate(regions, 1):
        x, y, w, h = (r[k] for k in ("x", "y", "width", "height"))
        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 55, 55), 3)
        px, py = x, max(y - 27, 0)
        label = f"FARK #{i}"
        cv2.rectangle(out, (px, py), (px + 112, py+24), (255, 55, 55), -1)
        cv2.putText(out, label, (px+5, py+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)
    return out


def red_difference_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black background, white text not included; red = changed pixels."""
    output = np.zeros_like(rgb)
    output[mask > 0] = (255, 45, 50)
    return output


def _register_reference(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Align original to suspect using ORB/homography only if needed.

    Valid mask prevents warped borders from being mistaken for edits.
    Homography is intentionally not used on same-sized inputs: for
    typical edited photos they already share pixel coordinates.
    """
    h, w = target.shape[:2]
    if reference.shape[:2] == (h, w):
        return reference.copy(), np.ones((h, w), np.uint8) * 255, "aynı çözünürlük, doğrudan hizalama"

    rh, rw = reference.shape[:2]
    if abs((rw / rh) - (w / h)) < 0.005:
        scaled = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LINEAR)
        return scaled, np.ones((h, w), np.uint8) * 255, "aynı en-boy oranında yeniden boyutlandırma"

    ref_g, tgt_g = gray(reference), gray(target)
    orb = cv2.ORB_create(nfeatures=5500)
    k1, d1 = orb.detectAndCompute(ref_g, None)
    k2, d2 = orb.detectAndCompute(tgt_g, None)
    if d1 is None or d2 is None or len(k1) < 15 or len(k2) < 15:
        raise ValueError("Referans ile inceleme görüntüsü hizalanamadı: yeterli ortak özellik yok.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.7*n.distance]
    if len(good) < 14:
        raise ValueError("İki görsel güvenilir biçimde eşleştirilemedi; aynı kadrajdaki orijinali yükleyin.")
    points1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)
    if H is None or inliers is None or inliers.mean() < 0.48:
        raise ValueError("Perspektif hizalaması güvenilir değil.")
    aligned = cv2.warpPerspective(reference, H, (w, h))
    valid = cv2.warpPerspective(np.full(reference.shape[:2], 255, np.uint8), H, (w, h),
                                flags=cv2.INTER_NEAREST)
    if np.count_nonzero(valid) / valid.size < 0.5:
        raise ValueError("Görsellerin ortak alanı çok küçük.")
    return aligned, valid, f"ORB homografi (%{inliers.mean()*100:.0f} eşleşme içte)"


def compare_to_original(suspect: np.ndarray, original: np.ndarray,
                        sensitivity: float = 1.0) -> dict[str, Any]:
    """Locate visually meaningful differences; cannot determine motive or AI origin."""
    aligned, valid, alignment = _register_reference(original, suspect)
    # Blur tiny JPEG artefacts, preserving meaningful localized edits.
    a = cv2.GaussianBlur(bgr(suspect), (3, 3), 0.7)
    b = cv2.GaussianBlur(bgr(aligned), (3, 3), 0.7)
    lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.linalg.norm(lab_a - lab_b, axis=2)
    usable = valid > 0
    values = delta[usable]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # Adaptive threshold plus a floor to reduce compression-only noise.
    threshold = max(13.0, median + (5.0 / sensitivity) * 1.4826 * mad)
    # A 3x3 morphological opening removes one-pixel outliers.
    initial = ((delta > threshold) & usable).astype(np.uint8) * 255
    opened = cv2.morphologyEx(initial, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = delta.shape
    mask = _remove_small_regions(closed, max(25, int(h*w*0.00008)))
    regions = _bbox_regions(mask)
    valid_n = int(np.count_nonzero(usable))
    changed_n = int(np.count_nonzero(mask))
    proportion = 100.0 * changed_n / max(valid_n, 1)
    heat = np.clip(delta * 255 / max(25.0, threshold*2.0), 0, 255).astype(np.uint8)
    heat[~usable] = 0
    return {
        "mode": "reference",
        "alignment": alignment,
        "threshold": round(threshold, 2),
        "difference_percent": round(proportion, 2),
        "regions": regions,
        "mask": mask,
        "heat": heat,
        "heat_overlay": color_heatmap(suspect, heat),
        "marked": highlight_changes(suspect, mask, regions),
        "mask_view": red_difference_mask(suspect, mask),
        "aligned_original": aligned,
        "valid_area_percent": round(100 * valid_n / (h*w), 2),
        "conclusion": (
            "Referans ile karşılaştırmada görsel farklar bulundu."
            if changed_n else "Bu eşiklerde anlamlı bir görsel fark saptanmadı."
        ),
        "caveat": "Bu sonuç, iki dosya arasındaki farklılıkları gösterir; AI kullanımını ya da kötü niyetli tahrifatı kanıtlamaz."
    }


def ela_map(rgb: np.ndarray, quality: int = 90) -> tuple[np.ndarray, np.ndarray]:
    """Error level analysis. Recompression patterns are circumstantial evidence only."""
    bio = BytesIO()
    Image.fromarray(rgb).save(bio, "JPEG", quality=quality, subsampling=0)
    bio.seek(0)
    re = np.array(Image.open(bio).convert("RGB")).astype(np.float32)
    raw = np.mean(np.abs(rgb.astype(np.float32) - re), axis=2)
    return raw, normalize_map(raw, 5, 99)


def noise_inconsistency_map(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """High-frequency residual variability map, not sensor fingerprint/PRNU."""
    g = gray(rgb).astype(np.float32)
    smoothed = cv2.GaussianBlur(g, (5, 5), 1.1)
    residual = g - smoothed
    local_energy = cv2.GaussianBlur(residual * residual, (0, 0), 6)
    local_grad = cv2.GaussianBlur(
        np.abs(cv2.Laplacian(smoothed, cv2.CV_32F)), (0, 0), 6
    )
    # Gradient normalization reduces, but does not remove, texture confounding.
    feature = np.sqrt(local_energy) / (2.0 + local_grad)
    typical = cv2.GaussianBlur(feature, (0, 0), max(15., min(rgb.shape[:2])/25))
    deviation = np.abs(feature - typical)
    return deviation, normalize_map(deviation, 5, 99)


def inspect_without_original(rgb: np.ndarray, filename: str = "") -> dict[str, Any]:
    """Display forensic indications, never verdict 'AI'/'authentic'."""
    h, w = rgb.shape[:2]
    _, ela = ela_map(rgb)
    _, noise = noise_inconsistency_map(rgb)
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    jpeg = ext in ("jpg", "jpeg")
    return {
        "mode": "indicators",
        "size": f"{w}×{h}",
        "ela": ela,
        "noise": noise,
        "ela_overlay": color_heatmap(rgb, ela),
        "noise_overlay": color_heatmap(rgb, noise),
        "jpeg_input": jpeg,
        "conclusion": "Tek görüntüyle kesin manipülasyon veya AI üretimi hükmü verilemez.",
        "caveat": ("ELA, JPEG sıkıştırması ve yeniden kaydetme geçmişinden etkilenir. "
                   "Gürültü farkları da doğal doku/ışık/odak değişimi nedeniyle oluşabilir. "
                   "Yüksek değerler kanıt veya kalibre edilmiş olasılık değildir.")
    }


def extract_metadata(data: bytes) -> dict[str, str]:
    with Image.open(BytesIO(data)) as img:
        exif = img.getexif()
        return {
            "format": str(img.format or "-"),
            "dimensions": f"{img.width}×{img.height}",
            "mode": img.mode,
            "exif_present": "Evet" if len(exif) else "Hayır",
            "software": str(exif.get(305, "-"))[:120],
            "camera_model": str(exif.get(272, "-"))[:120],
        }
