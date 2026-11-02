from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))
from train_classifier import scan_dataset, metrics_from_counts


def make_dataset(root):
    for split in ('train','val','test'):
        for cls in ('ai','real'):
            path = root / split / cls
            path.mkdir(parents=True)
            (path/'x.jpg').write_bytes(f'{split}_{cls}'.encode())


def test_dataset_structure_and_counts(tmp_path):
    make_dataset(tmp_path)
    result = scan_dataset(tmp_path)
    assert result['train/ai'] == 1
    assert len(result) == 6


def test_cross_split_file_duplicate_rejected(tmp_path):
    make_dataset(tmp_path)
    (tmp_path / 'test/ai/x.jpg').write_bytes(b'train_ai')
    with pytest.raises(ValueError, match='tekrarlanan'):
        scan_dataset(tmp_path)


def test_metric_calculation_positive_class_ai():
    m = metrics_from_counts(7, 2, 3, 8)
    assert m['accuracy'] == .75
    assert m['false_positive_rate'] == .2
    assert abs(m['recall_ai'] - .7) < 1e-10
