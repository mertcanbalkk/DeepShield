"""Detect byte-exact AND visually near-duplicate leakage across dataset splits.

Supports train/val/test/{ai,real} folders. Perceptual hashes are advisory:
images with uniform color or repeated patterns can collide.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from PIL import Image, ImageOps

EXTENSIONS = {'.jpg','.jpeg','.png','.webp','.tif','.tiff'}


def difference_hash(path: Path) -> int:
    with Image.open(path) as src:
        gray = ImageOps.exif_transpose(src).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pix = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pix[y*9+x] > pix[y*9+x+1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    """Metric tree avoids quadratic image comparisons in typical datasets."""
    def __init__(self):
        self.root = None

    def add(self, value, payload):
        if self.root is None:
            self.root = [value, [payload], {}]
            return
        node = self.root
        while True:
            d = hamming(value, node[0])
            if d == 0:
                node[1].append(payload)
                return
            if d not in node[2]:
                node[2][d] = [value, [payload], {}]
                return
            node = node[2][d]

    def query(self, value, radius):
        matches = []
        if self.root is None:
            return matches
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming(value, node[0])
            if dist <= radius:
                matches.extend((dist, payload) for payload in node[1])
            stack.extend(child for key,child in node[2].items() if dist-radius <= key <= dist+radius)
        return matches


def audit(root: Path, max_hamming: int = 4, max_examples: int = 30) -> dict:
    """Audit near duplicates across train/val/test; raises on missing folders."""
    if not 0 <= max_hamming <= 64:
        raise ValueError('Hamming eşiği 0..64 arasında olmalı.')
    tree = BKTree()
    known_exact = {}
    counts, collisions, invalid = {}, [], []
    for split in ('train','val','test'):
        for label in ('ai','real'):
            folder = root / split / label
            if not folder.is_dir():
                raise ValueError('Eksik klasör: ' + str(folder))
            paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            counts[f'{split}/{label}'] = len(paths)
            for path in paths:
                try:
                    digest = sha256(path.read_bytes()).hexdigest()
                    dhash = difference_hash(path)
                except (OSError, ValueError) as exc:
                    invalid.append({'file':str(path), 'error':str(exc)[:200]})
                    continue
                if digest in known_exact and known_exact[digest][0] != split and len(collisions) < max_examples:
                    collisions.append({'kind':'exact','a':known_exact[digest][2], 'b':str(path),'distance':0})
                for dist,(old_split, old_label, old_path, old_digest) in tree.query(dhash,max_hamming):
                    if old_split != split and old_digest != digest:
                        if len(collisions) < max_examples:
                            collisions.append({'kind':'near','a':old_path,'b':str(path),'distance':dist})
                        break
                known_exact.setdefault(digest, (split,label,str(path),digest))
                tree.add(dhash,(split,label,str(path),digest))
    return {'counts':counts, 'collisions':collisions, 'invalid':invalid,
            'valid':not collisions and not invalid and all(counts.values()),
            'note':'dHash şüpheli yakın-kopyaları gösterir; yanlış alarmlar elle doğrulanmalıdır. Kaynak/üretici grubu ayrımı ayrıca gerekir.'}


if __name__ == '__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description='Eğitim/test veri sızıntısı kontrolü')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--hamming',type=int,default=4)
    args=parser.parse_args()
    report=audit(args.data,max_hamming=args.hamming)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['valid'] else 1)
