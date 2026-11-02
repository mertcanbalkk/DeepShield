"""Turn extracted GenImage folders into DeepShield v6 train/val/test/ai|real.

GenImage's official train is subdivided into train+validation; official val is
reserved for test. Selected images are COPIED; originals are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
SPLITS = ('train', 'val', 'test')
LABELS = ('ai', 'real')


def discover(source: Path):
    """Find generator folders in an extracted GenImage tree."""
    roots = []
    candidate_dirs = [source] + [p for p in source.rglob('*') if p.is_dir() and p.name == 'train' and p.parent != source]
    # Only generator directories where train and val both contain ai and nature.
    for p in candidate_dirs:
        generator = p.parent if p.name == 'train' else p
        if all((generator / split / label).is_dir() for split in ('train', 'val') for label in ('ai', 'nature')):
            if generator not in roots:
                roots.append(generator)
    return sorted(roots)


def files(folder: Path):
    return sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTS), key=str)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def create(source: Path, output: Path, train_count: int, val_count: int, test_count: int,
           seed: int = 42, generator_filter: str | None = None):
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f'Kaynak klasör yok: {source}')
    if output == source or source in output.parents:
        raise ValueError('Çıktıyı GenImage kaynak klasörünün DIŞINA seçin.')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Çıktı klasörü boş olmalı: {output} (eski veriyi karıştırmayalım)')
    roots = discover(source)
    if generator_filter:
        roots = [p for p in roots if generator_filter.casefold() in p.name.casefold()]
    if not roots:
        raise ValueError('GenImage klasörleri bulunamadı. ZIP/7z arşivlerini açın; örnek: Midjourney/train/ai ve train/nature.')
    rng = random.Random(seed)
    print('Bulunan üreticiler:', ', '.join(r.name for r in roots), flush=True)
    requested = {'train': train_count, 'val': val_count, 'test': test_count}
    if min(requested.values()) < 1:
        raise ValueError('Her split için sınıf başına en az 1 görüntü seçin.')
    # Each producer receives a quota, but smaller pools can be compensated by others.
    pool = {}
    for g in roots:
        for label in LABELS:
            sub = 'ai' if label == 'ai' else 'nature'
            for source_split in ('train', 'val'):
                samples = files(g / source_split / sub)
                rng.shuffle(samples)
                pool[(g.name, source_split, label)] = samples
    counts = {s: {l: 0 for l in LABELS} for s in SPLITS}
    by_generator = {g.name: {s: {l: 0 for l in LABELS} for s in SPLITS} for g in roots}
    seen_paths = set()
    seen_hashes = set()
    # Test FIRST: preserve official GenImage val images as never-used test samples.
    for split in ('test', 'val', 'train'):
        origin = 'val' if split == 'test' else 'train'
        for label in LABELS:
            target = requested[split]
            names = [g.name for g in roots]
            # Fair round-robin allocation across generators.
            index = 0
            while counts[split][label] < target:
                did_copy = False
                for i in range(len(names)):
                    gen = names[(index + i) % len(names)]
                    samples = pool[(gen, origin, label)]
                    while samples:
                        path = samples.pop()
                        if str(path) in seen_paths:
                            continue
                        seen_paths.add(str(path))
                        digest = sha256_file(path)
                        if digest in seen_hashes:
                            continue
                        seen_hashes.add(digest)
                        out_folder = output / split / label
                        out_folder.mkdir(parents=True, exist_ok=True)
                        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', gen)[:55]
                        dest = out_folder / f'{safe}_{digest[:20]}{path.suffix.lower()}'
                        shutil.copy2(path, dest)
                        counts[split][label] += 1
                        by_generator[gen][split][label] += 1
                        did_copy = True
                        if counts[split][label] % 500 == 0:
                            print(f'{split}/{label}: {counts[split][label]}/{target}', flush=True)
                        break
                    if did_copy:
                        index = (index + i + 1) % len(names)
                        break
                if not did_copy:
                    raise ValueError(f'{split}/{label} için {target} benzersiz görüntü bulunamadı. '
                                     f'Bulunan: {counts[split][label]}. Daha küçük sayı seçin veya başka üreticiler ekleyin. '
                                     'Eksik hazırlanan çıktı klasörünü silip tekrar deneyin.')
    report = {'source': str(source), 'output': str(output), 'seed': seed,
              'generator_names': [g.name for g in roots], 'counts': counts,
              'by_generator': by_generator,
              'note': 'Original GenImage val was used ONLY for test; GenImage train split into DeepShield train/val. '
                      'Exact SHA-256 duplicate images excluded. Perceptual near-duplicates must be audited separately.'}
    (output / 'dataset_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='GenImage → DeepShield v6 training data converter')
    parser.add_argument('--source', type=Path, required=True, help='Arşivleri açılmış GenImage klasörü')
    parser.add_argument('--output', type=Path, default=Path('datasets/ai_origin'))
    parser.add_argument('--train-per-class', type=int, default=8000)
    parser.add_argument('--val-per-class', type=int, default=1000)
    parser.add_argument('--test-per-class', type=int, default=1000)
    parser.add_argument('--generator', default=None, help='İsteğe bağlı üretici adı filtresi')
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    try:
        report = create(a.source, a.output, a.train_per_class, a.val_per_class,
                        a.test_per_class, a.seed, a.generator)
    except (ValueError, OSError) as e:
        print('HATA:', e, file=sys.stderr)
        sys.exit(1)
    print('TAMAMLANDI:', json.dumps(report['counts'], ensure_ascii=False), flush=True)
    print('Rapor:', a.output / 'dataset_report.json')

if __name__ == '__main__':
    main()
