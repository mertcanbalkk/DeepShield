"""Offline false-AI alarm veto calibrated only from verified camera photos.

Keep these calibration photos DIFFERENT from training, model selection and
independent final test photos. Original images remain untouched. No downloads.

Example:
 python training/calibrate_real_guard.py --real-dir datasets/calibration_real \\
   --weights outputs/deepshield_ai_b0.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps
try:
    from .dataset_quality import EXTENSIONS
    from .real_guard import MIN_CALIBRATION_REAL, make_guard, sha256_file
except ImportError:
    from dataset_quality import EXTENSIONS
    from real_guard import MIN_CALIBRATION_REAL, make_guard, sha256_file


def collect_photos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError('Gerçek fotoğraf klasörü bulunamadı: ' + str(folder))
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def calibrate(folder: Path, weights: Path, out: Path, minimum: int = MIN_CALIBRATION_REAL) -> dict:
    import numpy as np
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from model_adapters import load_custom_classifier, classify_custom
    try:
        from .real_guard import ai_evidence
    except ImportError:
        from real_guard import ai_evidence

    paths = collect_photos(folder)
    if len(paths) < minimum:
        raise ValueError(f'En az {minimum} ayrı gerçek kamera fotoğrafı gerekli; klasörde {len(paths)} dosya var.')
    bundle = load_custom_classifier(str(weights))
    raw_scores = []
    fingerprints: set[str] = set()
    failures = []
    for i, path in enumerate(paths, 1):
        try:
            digest = sha256_file(path)
            if digest in fingerprints:
                print(f'ATLANDI: birebir tekrarlanan dosya: {path}')
                continue
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
            result = classify_custom(rgb, bundle, conservative=False)
            # labels[1:3] store JPEG90/JPEG70; raw_score is original.
            # For current adapter we also expose the original score as the
            # first element of the 3-variant prediction (see model_adapters.py).
            scores = [p['score'] for p in result.labels if p['label'].startswith(('Orijinal', 'JPEG90', 'JPEG70'))]
            if len(scores) != 3:
                raise ValueError('Orijinal/JPEG90/JPEG70 skorları bulunamadı; model sürümünü güncelle.')
            ai_evidence(scores)
            raw_scores.append(scores)
            fingerprints.add(digest)
        except Exception as exc:
            failures.append({'file': str(path), 'error': str(exc)[:350]})
        if i % 10 == 0 or i == len(paths):
            print(f'Kontrol: {i}/{len(paths)}, geçerli: {len(raw_scores)}, hatalı: {len(failures)}', flush=True)
    if len(raw_scores) < minimum:
        raise ValueError(f'Geçerli ve tekil gerçek fotoğraf sayısı {len(raw_scores)}; en az {minimum} gerekli. '
                         f'İlk hata: {failures[:1]}')
    result = make_guard(raw_scores, sha256_file(weights), min_images=minimum)
    result['calibration_folder'] = str(folder)
    result['calibration_sha256'] = sorted(fingerprints)
    result['n_invalid'] = len(failures)
    result['n_duplicate_skipped'] = len(paths) - len(raw_scores) - len(failures)
    result['independent_test_required'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Gerçek fotoğraflara yanlış AI alarmını azaltmak için çevrimdışı eşik')
    parser.add_argument('--real-dir', type=Path, required=True,
                        help='Modelin eğitim/val/test görüntülerinden farklı gerçek kamera fotoğrafları')
    parser.add_argument('--weights', type=Path, default=Path('outputs/deepshield_ai_b0.pt'))
    parser.add_argument('--out', type=Path, default=Path('outputs/deepshield_real_guard.json'))
    parser.add_argument('--min-real', type=int, default=MIN_CALIBRATION_REAL)
    args = parser.parse_args()
    if args.min_real < MIN_CALIBRATION_REAL:
        parser.error(f'--min-real en az {MIN_CALIBRATION_REAL} olmalı')
    result = calibrate(args.real_dir, args.weights, args.out, args.min_real)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('UYARI: Bu kalibrasyon kümesinde sıfır AI alarmı, bağımsız testte sıfır hata garantisi değildir.')


if __name__ == '__main__':
    main()
