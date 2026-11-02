"""Optional real trained-model adapters. Fail visibly, never silently fabricate results."""
from __future__ import annotations

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

DEFAULT_CLASSIFIER = 'wkaandemir/ai-image-detector'
LEGACY_CLASSIFIER = 'dima806/ai_vs_human_generated_image_detection'


@dataclass
class ClassifierResult:
    model: str
    labels: list[dict[str, Any]]
    ai_score: float | None
    note: str
    decision: str | None = None
    raw_real_score: float | None = None


def classify_ai(rgb: np.ndarray, pipeline_obj: Any, model_id: str = DEFAULT_CLASSIFIER) -> ClassifierResult:
    """Predict whole-image AI origin; NOT whether a part was AI inpainted."""
    preds = pipeline_obj(Image.fromarray(rgb), top_k=None)
    labels = [{'label': str(p['label']), 'score': float(p['score'])} for p in preds]
    ai_labels = {'ai-generated', 'ai_generated', 'ai', 'artificial', 'synthetic', 'fake', 'generated'}
    human_labels = {'human', 'real', 'authentic', 'human-generated', 'human_generated'}
    found_ai = [p['score'] for p in labels if p['label'].lower().strip() in ai_labels]
    found_human = [p['score'] for p in labels if p['label'].lower().strip() in human_labels]
    if found_ai:
        ai = max(found_ai)
    elif found_human and len(labels) == 2:
        ai = 1 - max(found_human)
    else:
        ai = None  # Do not assume LABEL_0 means AI.
    return ClassifierResult(
        model=model_id, labels=labels, ai_score=ai,
        note=('Modelin ham sınıflandırma skoru; güncel üreticilerde ve yeniden sıkıştırılmış '
              'görsellerde kalibre edilmiş gerçeklik/kanıt olasılığı değildir. '
              'Bu model bölgesel AI düzenlemesini yerelleştirmez.'))


def load_modern_model():
    """Load a documented 2026 CLIP image-classification model from HF.

    Checkpoint is not shipped. Dependencies/download needed on the user's PC.
    """
    import timm
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from torchvision import transforms

    model_file = hf_hub_download(repo_id=DEFAULT_CLASSIFIER, filename='model.safetensors')
    model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=False, num_classes=1, img_size=256)
    state = load_file(model_file)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Refuse silent usage if checkpoint did not match architecture.
    total = len(model.state_dict())
    if total < 1 or len(missing) > total * .15 or len(unexpected) > total * .15:
        raise RuntimeError(f'Model ve ağırlıklar uyumsuz: eksik={len(missing)} fazla={len(unexpected)}')
    model.eval()
    prep = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481,0.458,0.408], std=[0.269,0.261,0.276])
    ])
    return model, prep


def classify_ai_modern(rgb: np.ndarray, model_bundle) -> ClassifierResult:
    """Run the model author's inference and decision band on a single image."""
    import torch
    model, prep = model_bundle
    x = prep(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        real = float(torch.sigmoid(model(x)).item())
    real = min(1., max(0., real))
    decision = ('Model: AI üretimi yönünde' if real < .91 else
                'Model: gerçek fotoğraf yönünde' if real >= .93 else
                'Model: belirsiz aralıkta')
    return ClassifierResult(
        model=DEFAULT_CLASSIFIER,
        labels=[{'label':'real','score':real},{'label':'fake','score':1-real}],
        ai_score=1-real, raw_real_score=real, decision=decision,
        note=('2026 CLIP modelinin üreticisinin belirlediği p(real) <0.91 / 0.91–0.93 / >=0.93 '
              'karar bantları uygulanır. Değer gerçek dünya kesin olasılığı değildir. '
              'Bölgesel AI inpainting yerini bulmaz. Başka veri dağılımlarında yanılabilir.')
    )


@dataclass
class LocalizationResult:
    model: str
    mask_scores: np.ndarray
    conf_scores: np.ndarray | None
    image_score: float | None
    note: str


def trufor_available(repo_dir: str, weights: str) -> bool:
    repo = Path(repo_dir).expanduser()
    return (repo / 'TruFor_train_test' / 'test.py').is_file() and Path(weights).expanduser().is_file()


def run_trufor(rgb: np.ndarray, repo_dir: str, weights: str, use_gpu: bool = False,
               timeout_seconds: int = 420, original_bytes: bytes | None = None,
               original_format: str | None = None,
               trust_official_checkpoint: bool = False) -> LocalizationResult:
    """Run official TruFor on one photo without rewriting its repository.

    TruFor's test.py supports an explicit .npz filename for -out. We use that
    instead of a directory: on Windows, its directory mode can turn an
    absolute input path into C:\\suspect.png.npz (permission denied).

    The weights and the TruFor repository stay untouched. No downloads occur.
    """
    repo = Path(repo_dir).expanduser().resolve()
    weights_path = Path(weights).expanduser().resolve()
    runner = repo / 'TruFor_train_test' / 'test.py'
    if not runner.is_file():
        raise FileNotFoundError('TruFor test.py bulunamadı: ' + str(runner))
    if not weights_path.is_file():
        raise FileNotFoundError('TruFor ağırlık dosyası bulunamadı: ' + str(weights_path))
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('TruFor için H×W×3 boyutunda uint8 RGB görüntüsü gerekir.')
    python_exe = os.environ.get('DEEPSHIELD_TRUFOR_PYTHON', sys.executable)
    if use_gpu and not os.environ.get('DEEPSHIELD_TRUFOR_PYTHON'):
        # The project's current Python environment is CPU-only for many users.
        try:
            import torch
            cuda_ready = torch.cuda.is_available()
        except (ImportError, OSError):
            cuda_ready = False
        if not cuda_ready:
            raise RuntimeError('Bu Python ortamında CUDA yok. Panelde TruFor GPU seçeneğini kapatıp CPU ile dene.')

    with tempfile.TemporaryDirectory(prefix='deepshield-trufor-') as td:
        temp = Path(td)
        fmt = (original_format or '').upper()
        preserve = original_bytes is not None and fmt in ('JPEG', 'PNG')
        in_file = temp / ('suspect.jpg' if preserve and fmt == 'JPEG' else 'suspect.png')
        if preserve:
            # Only pass bytes through when they really represent the displayed
            # orientation/size. Otherwise encode the normalized pixels as PNG.
            from io import BytesIO
            from PIL import ImageOps
            with Image.open(BytesIO(original_bytes)) as source:
                normalized = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
            if normalized.shape == rgb.shape and np.array_equal(normalized, rgb):
                in_file.write_bytes(original_bytes)
            else:
                Image.fromarray(rgb).save(in_file, format='PNG')
                if in_file.suffix != '.png':
                    converted = in_file.with_suffix('.png')
                    in_file.replace(converted)
                    in_file = converted
        else:
            Image.fromarray(rgb).save(in_file, format='PNG')

        out_dir = temp / 'model_outputs'
        out_dir.mkdir()
        # IMPORTANT: official test.py accepts a full .npz filename. A directory
        # with a Windows absolute -in path can mistakenly target the C: root.
        output_file = out_dir / 'result.npz'
        cmd = [python_exe, str(runner), '-g', '0' if use_gpu else '-1',
               '-in', str(in_file), '-out', str(output_file), '-exp', 'trufor_ph3',
               'TEST.MODEL_FILE', str(weights_path)]
        # Newer PyTorch blocks older pickle checkpoints by default. Only for
        # user-confirmed trusted official weights, opt in to legacy loading.
        # This is potentially unsafe for an untrusted .pth.tar file.
        env = os.environ.copy()
        if trust_official_checkpoint:
            env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        try:
            result = subprocess.run(cmd, cwd=str(runner.parent), capture_output=True,
                                    text=True, timeout=timeout_seconds, check=False, env=env)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError('TruFor zaman aşımı (' + str(timeout_seconds) + ' saniye).') from e
        except OSError as e:
            raise RuntimeError('TruFor Python işlemi başlatılamadı: ' + str(e)) from e

        # TruFor's original test.py catches inference exceptions and sometimes
        # exits with return code 0. The presence of the output file is required.
        stderr_tail = (result.stderr or '')[-6000:]
        stdout_tail = (result.stdout or '')[-1000:]
        if result.returncode != 0 or not output_file.is_file():
            reason = 'çıkış kodu ' + str(result.returncode) if result.returncode != 0 else 'sonuç dosyası oluşmadı'
            raise RuntimeError('TruFor başarısız (' + reason + ').\n'
                               + ('GERÇEK HATA:\n' + stderr_tail + '\n' if stderr_tail else '')
                               + ('ÇIKTI:\n' + stdout_tail if stdout_tail else ''))
        try:
            with np.load(output_file, allow_pickle=False) as data:
                if 'map' not in data:
                    raise ValueError('TruFor çıktısında map alanı yok.')
                prob = np.asarray(data['map'], np.float32).squeeze()
                conf = np.asarray(data['conf'], np.float32).squeeze() if 'conf' in data else None
                image_score = float(np.asarray(data['score']).item()) if 'score' in data else None
        except (OSError, ValueError, KeyError) as e:
            raise RuntimeError('TruFor sonuç dosyası bozuk ya da okunamadı: ' + str(e)) from e

    if prob.ndim != 2 or not np.isfinite(prob).all():
        raise ValueError('TruFor geçerli iki boyutlu anomali haritası üretmedi.')
    h, w = rgb.shape[:2]
    if prob.shape != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    if conf is not None:
        if conf.ndim == 2 and np.isfinite(conf).all():
            if conf.shape != (h, w):
                conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            conf = None
    if image_score is not None and not np.isfinite(image_score):
        image_score = None
    return LocalizationResult(
        model='TruFor (CVPR 2023)', mask_scores=np.clip(prob, 0, 1),
        conf_scores=np.clip(conf, 0, 1) if conf is not None else None,
        image_score=image_score,
        note='TruFor tahmin haritası; tek başına tahrifatın kanıtı değildir.'
    )


def load_custom_classifier(weights_path: str):
    """Load fine-tuned weights AND the validation-derived decision thresholds."""
    import torch
    from torchvision import models, transforms
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'efficientnet_b0' or checkpoint.get('classes') != ['ai','real']:
        raise ValueError('Beklenen DeepShield EfficientNet checkpoint biçimi bulunamadı.')
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features,2)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(checkpoint['mean'],checkpoint['std'])
    ])
    return model, preprocess, checkpoint.get('calibration', {'ready':False})


def classify_custom(rgb: np.ndarray, loaded_bundle, *, conservative: bool = True,
                    real_guard: dict | None = None) -> ClassifierResult:
    """Evaluate native/recompressed variants; abstain on conflicts.

    Compared to a single-score threshold this intentionally sacrifices some
    coverage to avoid confident claims when compression changes the verdict.
    """
    import torch
    from io import BytesIO
    from training.decision import decide
    from training.real_guard import protect_decision
    model, preprocess, thresholds = loaded_bundle
    source = Image.fromarray(rgb)
    variants = [source]
    for quality in (90, 70):
        mem = BytesIO()
        source.save(mem, format='JPEG', quality=quality)
        mem.seek(0)
        with Image.open(mem) as picture:
            variants.append(picture.convert('RGB'))
    scores=[]
    with torch.inference_mode():
        for variant in variants:
            score=float(torch.softmax(model(preprocess(variant).unsqueeze(0)),dim=1)[0,0].item())
            scores.append(score)
    assessment = decide(scores, thresholds)
    decision, guard_note = protect_decision(
        assessment['decision'], scores, guard=real_guard, enabled=conservative)
    ai = assessment['mean_score']
    return ClassifierResult(model='DeepShield EfficientNet-B0 (kalibrasyon + sıkıştırma testi)',
                            labels=[{'label':'Orijinal ham AI skoru','score':scores[0]},
                                    {'label':'JPEG90 ham AI skoru','score':scores[1]},
                                    {'label':'JPEG70 ham AI skoru','score':scores[2]},
                                    {'label':'Ham AI ortalama skoru','score':ai}],
                            ai_score=ai, decision=decision,
                            note=('3 görüntü varyantı birlikte değerlendirildi. '+
                                  ('Doğrulama kümesi tabanlı eşikler kullanıldı. ' if thresholds.get('ready') else
                                   'Kalibrasyon yok; kesin sınıflandırma yapmıyoruz. ')+
                                  'Ham skor kesin gerçeklik olasılığı değildir. Model yalnızca TAM AI üretimini sınıflandırır. ' + guard_note))
