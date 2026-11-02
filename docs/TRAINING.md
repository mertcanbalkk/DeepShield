# Eğitim ve doğruluk ölçümü

**Model ağırlıkları bu repoda yoktur.** Eğitim ve test verileri `datasets/`, ağırlıklar ve raporlar `outputs/` altında yerel tutulur; GitHub'a gönderilmez. AI kurulumu (PyTorch vb.) büyük indirmeler gerektirebilir; `requirements-ai.txt` isteğe bağlıdır.

## 1. Veri hazırlama (GenImage örneği)

İndirilmiş ve arşivden çıkarılmış GenImage klasöründe `train/ai`, `train/nature`, `val/ai` ve `val/nature` bulunmalıdır. `nature` gerçek fotoğrafları ifade eder. GenImage veri kümesinin lisansını ayrıca kontrol edin.

```powershell
python prepare_genimage.py --source "D:\GenImage" --output datasets/ai_origin --train-per-class 1000 --val-per-class 200 --test-per-class 200
python -m training.dataset_quality --data datasets/ai_origin
python training/train_classifier.py --data datasets/ai_origin --epochs 3 --batch-size 8
```

Hazırlayıcı, GenImage `val` örneklerini DeepShield `test` kümesine ayırır; GenImage `train` örneklerinden `train` ve `val` oluşturur. Birebir SHA-256 tekrarlarını eler, ancak yakın kopyaları gözle incelemek gerekir. Çıktı klasörü boş olmalıdır; başka verilerle karıştırmayın. Eğitimden sonra uyumlu checkpoint varsayılan olarak `outputs/deepshield_ai_b0.pt` konumundadır.

## 2. Yanlış AI alarmını azaltmak için kalibrasyon

Eğitim, doğrulama ve testte kullanılmamış, **kesin gerçek** ve birbirinden bağımsız en az 50 kamera fotoğrafını `datasets/calibration_real/` içine koyun (farklı cihazlar/ortamlar idealdir).

```powershell
python training/calibrate_real_guard.py --real-dir datasets/calibration_real --weights outputs/deepshield_ai_b0.pt --out outputs/deepshield_real_guard.json
```

Bunun amacı gerçek fotoğraflara verilen yanlış AI kararını azaltmaktır; modelin ham skoru değişmez. Başlangıçtaki temkinli eşik `0.98`, **evrensel veya kalibre edilmiş olasılık** değildir. Bazı AI görüntüleri de belirsiz kalabilir.

## 3. Bağımsız test

Etiketleri gerçekten bilinen, eğitim/kalibrasyonda **hiç kullanılmamış** örnekleri `datasets/external_eval/real/` ve `datasets/external_eval/ai/` klasörlerine ayırın (mümkünse her sınıf için 100+, farklı kaynaklardan). Aynı görselin farklı biçimde kaydedilmiş kopyalarını bölümler arasında kullanmayın.

```powershell
python training/evaluate_external.py --data datasets/external_eval --weights outputs/deepshield_ai_b0.pt --guard outputs/deepshield_real_guard.json --out outputs/external_eval.json
```

Kalibrasyon yapılmadıysa `--guard ...` seçeneğini çıkarın. Raporun `baseline` ve `conservative` alanlarında `real_false_ai` (gerçeğe AI denmesi), `ai_correct`, `ai_uncertain`, `real_uncertain`, `coverage` ve karar verilenlerdeki doğruluğu birlikte değerlendirin. **Belirsizler doğru sayılmaz.** Aynı kümede eşik ayarlayıp başarı raporlamak bağımsız değerlendirme değildir.

## 4. Bölgesel analiz

TruFor ayrı model ve ağırlık gerektirir. Kurulum: [TRUFOR.md](TRUFOR.md). Şüpheli maske alanı **AI ile oluşturulan piksel yüzdesi değildir**. Gerçek bölge doğruluğu için maskesi bilinen manipülasyon örnekleriyle ayrıca değerlendirme gereklidir.
