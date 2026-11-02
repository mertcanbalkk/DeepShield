"""Read-only, network-free DeepShield installation check."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def check():
    print('DeepShield 6.5 — çevrimdışı kontrol')
    print('Python:', sys.version.split()[0], '| Yorumlayıcı:', sys.executable)
    for module in ('streamlit', 'torch', 'torchvision', 'cv2', 'numpy', 'PIL'):
        ok = importlib.util.find_spec(module) is not None
        print(('OK ' if ok else 'EKSİK ')+module)
    weights = ROOT / 'outputs' / 'deepshield_ai_b0.pt'
    trufor = ROOT / 'models' / 'TruFor' / 'TruFor_train_test' / 'test.py'
    trufor_weights = ROOT / 'models' / 'TruFor' / 'TruFor_train_test' / 'pretrained_models' / 'trufor.pth.tar'
    for label, path in [('AI model ağırlığı',weights),('TruFor kaynak',trufor),('TruFor ağırlığı',trufor_weights)]:
        print(('OK ' if path.is_file() else 'BULUNAMADI ')+label+': '+str(path))
    try:
        import torch
        print('PyTorch:',torch.__version__,'| CUDA:',torch.cuda.is_available())
    except (ImportError, OSError) as e:
        print('PyTorch kontrol hatası:', str(e))
    print('Bu kontrol model doğruluğu ölçmez; paket indirmez.')


if __name__ == '__main__':
    check()
