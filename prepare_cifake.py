"""Prepare CIFAKE-style train/FAKE,train/REAL,test/FAKE,test/REAL
for DeepShield v6 train/val/test/{ai,real}.

Does not modify source data. Keeps source test untouched and uses only the source
training split for train and validation. Deterministic, avoids byte-identical
samples across splits, and never silently overwrites an existing destination.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
import random
import shutil
import sys

EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
LABELS = {'FAKE':'ai', 'REAL':'real'}


def image_paths(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f'Kaynak klasor bulunamadi: {folder}')
    paths = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXT]
    if not paths:
        raise ValueError(f'Gorsel yok: {folder}')
    return sorted(paths)


def choose_unique(candidates, n, rng, known):
    rng.shuffle(candidates)
    chosen = []
    for path in candidates:
        digest = sha256(path.read_bytes()).digest()
        if digest in known:
            continue
        chosen.append((path, digest))
        known.add(digest)
        if len(chosen) == n:
            return chosen
    raise ValueError(f'Yalnizca {len(chosen)} benzersiz ornek bulundu; {n} gerekli. Daha kucuk sayi secin.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path(r'C:\GenImage'))
    p.add_argument('--output', type=Path, default=Path('datasets/ai_origin'))
    p.add_argument('--train-per-class', type=int, default=1000)
    p.add_argument('--val-per-class', type=int, default=200)
    p.add_argument('--test-per-class', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.train_per_class,args.val_per_class,args.test_per_class) < 1:
        p.error('Butun ornek sayilari pozitif olmali')
    if args.output.exists() and any(args.output.iterdir()):
        p.error(f'Hedef klasor bos degil: {args.output}. Once farkli --output verin.')
    rng = random.Random(args.seed)
    plan = {}
    known = set()
    # Use source TEST only for test, and source TRAIN only for train/val.
    for source_label, target_label in LABELS.items():
        train_items = image_paths(args.source/'train'/source_label)
        test_items = image_paths(args.source/'test'/source_label)
        train_chosen = choose_unique(train_items, args.train_per_class, rng, known)
        val_chosen = choose_unique(train_items, args.val_per_class, rng, known)
        test_chosen = choose_unique(test_items, args.test_per_class, rng, known)
        plan['train',target_label] = train_chosen
        plan['val',target_label] = val_chosen
        plan['test',target_label] = test_chosen
    counts = Counter()
    for (split,label), files in plan.items():
        dest = args.output/split/label
        dest.mkdir(parents=True, exist_ok=True)
        for i,(src,digest) in enumerate(files,1):
            filename = f'{i:06d}_{digest.hex()[:10]}{src.suffix.lower()}'
            shutil.copy2(src,dest/filename)
            counts[f'{split}/{label}'] += 1
    print('BASARILI! DeepShield verisi hazir: ', args.output.resolve())
    for key in sorted(counts):
        print(f'{key}: {counts[key]}')
    print('Not: 32x32 CIFAKE sadece ilk egitim denemesidir; genel AI tespiti dogrulugu garanti edilmez.')

if __name__ == '__main__':
    try: main()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print('HATA:', exc, file=sys.stderr)
        sys.exit(1)
