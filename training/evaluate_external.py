"""Evaluate a trained detector on an independent, source-held-out dataset.

Expected: --data path with AI or real folders; optionally producer/ai, producer/real.
This script NEVER updates a model or thresholds. Report every result, incl. abstentions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image, ImageOps
try:
    from .dataset_quality import EXTENSIONS
except ImportError:
    from dataset_quality import EXTENSIONS


def summarize(items):
    """A truthful confusion table with abstention as a THIRD outcome."""
    count = defaultdict(int)
    for truth, result in items:
        count[(truth, result)] += 1
    real_n = sum(count[(1, x)] for x in ('ai','real','uncertain'))
    ai_n = sum(count[(0, x)] for x in ('ai','real','uncertain'))
    answer = {'count':len(items), 'ai_n':ai_n,'real_n':real_n,
              'ai_correct':count[(0,'ai')], 'ai_missed_as_real':count[(0,'real')],
              'ai_uncertain':count[(0,'uncertain')],
              'real_correct':count[(1,'real')], 'real_false_ai':count[(1,'ai')],
              'real_uncertain':count[(1,'uncertain')]}
    answer['ai_recall_over_all'] = count[(0,'ai')] / ai_n if ai_n else None
    answer['false_ai_rate_over_all_real'] = count[(1,'ai')] / real_n if real_n else None
    decided = (count[(0,'ai')] + count[(0,'real')] + count[(1,'ai')] + count[(1,'real')])
    correct = count[(0,'ai')] + count[(1,'real')]
    answer['coverage'] = decided / len(items) if items else None
    answer['accuracy_over_all_with_abstentions_not_correct'] = correct / len(items) if items else None
    answer['selective_accuracy_among_decided'] = correct / decided if decided else None
    answer['precision_ai_among_ai_decisions'] = (count[(0,'ai')] / (count[(0,'ai')]+count[(1,'ai')])
                                               if count[(0,'ai')]+count[(1,'ai')] else None)
    answer['real_recall_over_all'] = count[(1,'real')] / real_n if real_n else None
    answer['false_real_rate_over_all_ai'] = count[(0,'real')] / ai_n if ai_n else None
    return answer


def discover(root):
    groups = []
    # Either data/{ai,real} or data/<producer>/{ai,real}
    folders = [root] if any((root/l).is_dir() for l in ('ai','real')) else [d for d in root.iterdir() if d.is_dir()]
    for folder in folders:
        for label in ('ai','real'):
            if (folder/label).is_dir():
                for path in sorted((folder/label).rglob('*')):
                    if path.is_file() and path.suffix.lower() in EXTENSIONS:
                        groups.append((folder.name, 0 if label=='ai' else 1, path))
    if not groups:
        raise ValueError('Harici değerlendirme görselleri bulunamadı: <kök>/<üretici>/ai|real')
    return groups


def verdict_key(decision: str | None) -> str:
    if decision and decision.startswith('AI üretimi yönünde'):
        return 'ai'
    if decision and decision.startswith('Gerçek fotoğraf yönünde'):
        return 'real'
    return 'uncertain'


def evaluate_folder(root, weights, max_images=0, guard_path=None):
    """Compare previous versus conservative classifications on SAME held-out photos.

    A real image incorrectly identified as AI is a false alarm. Abstentions
    are counted separately, never disguised as correct answers.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_adapters import classify_custom, load_custom_classifier
    from training.real_guard import load_guard, protect_decision, sha256_file
    import numpy as np
    groups = discover(root)
    bundle = load_custom_classifier(str(weights))
    guard = load_guard(guard_path, weights) if guard_path else None
    results_base, results_safe = defaultdict(list), defaultdict(list)
    details = []
    errors = []
    skipped_overlap = []
    calibration_hashes = set(guard.get('calibration_sha256', [])) if guard else set()
    for producer, truth, path in groups[:max_images if max_images else None]:
        try:
            if calibration_hashes and sha256_file(path) in calibration_hashes:
                skipped_overlap.append(str(path))
                continue  # NEVER test on the images that selected a threshold.
            with Image.open(path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
            original = classify_custom(rgb, bundle, conservative=False)
            scores = [p['score'] for p in original.labels
                      if p['label'].startswith(('Orijinal', 'JPEG90', 'JPEG70'))]
            safe_decision, _ = protect_decision(original.decision, scores, guard=guard)
            base_verdict, safe_verdict = verdict_key(original.decision), verdict_key(safe_decision)
            results_base[producer].append((truth, base_verdict))
            results_safe[producer].append((truth, safe_verdict))
            details.append({'file': str(path), 'source': producer,
                            'truth': 'ai' if truth == 0 else 'real',
                            'raw_ai_score': original.ai_score,
                            'base_verdict': base_verdict, 'conservative_verdict': safe_verdict})
        except Exception as exc:
            errors.append({'file': str(path), 'error': str(exc)[:350]})
    base_items = [x for items in results_base.values() for x in items]
    safe_items = [x for items in results_safe.values() for x in items]
    return {
        'baseline': {'overall': summarize(base_items),
                     'by_source': {group:summarize(items) for group,items in results_base.items()}},
        'conservative': {'overall': summarize(safe_items),
                         'by_source': {group:summarize(items) for group,items in results_safe.items()}},
        'overall': summarize(safe_items),
        'by_source': {group:summarize(items) for group,items in results_safe.items()},
        'real_guard_used': bool(guard),
        'per_image': details,
        'errors': errors,
        'excluded_calibration_duplicates': skipped_overlap,
        'disclaimer': ('Test seti kalibrasyon/eğitim/doğrulamada kullanılmamalı. '
                       'Belirsiz yanıtlar doğru sayılmadı. Bağımsız üretici/cihaz '
                       'verisi olmadan gerçek dünya başarısı ölçülemez.')
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--guard', type=Path, default=None,
                        help='Opsiyonel outputs/deepshield_real_guard.json; aynı fotoğraflarla kalibre/test etmeyin')
    parser.add_argument('--out',type=Path,default=Path('outputs/external_eval.json'))
    parser.add_argument('--max-images',type=int,default=0)
    args=parser.parse_args()
    report=evaluate_folder(args.data,args.weights,args.max_images,args.guard)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'per_image'}, ensure_ascii=False,indent=2))
    print('Görsel bazında ham skorlar rapor dosyasında: ' + str(args.out))


if __name__=='__main__':
    main()
