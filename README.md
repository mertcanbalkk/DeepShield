# DeepShield — Görsel adli analiz paneli

**Tek fotoğraf yükle → tüm görüntü için AI sınıf skoru → varsa TruFor ile
şüpheli bölgeler → SHA-256, EXIF ve teknik görüntü haritaları.**

> **Önemli:** Bu repo eğitilmiş `.pt` ağırlıklarını veya TruFor modelini **içermez**.
> Sadece kodu indirip başlattığınızda eğitilmiş sınıflandırıcı **hazır olmaz**.
> Ağırlıkları ayrı hazırlamalı veya arayüzden dış model seçmeyi açıkça
> onaylamalısınız (dış model internetten büyük dosya indirebilir).
> **%100 güvenilir AI tespiti iddiası yoktur.** Tam görüntü skoru, resimdeki
> AI piksel yüzdesi veya doğrulanmış olasılık değildir.

## Özellikler

- **Streamlit paneli:** JPEG, PNG, WebP ve TIFF görüntüleri yerel bilgisayarda inceleme.
- **AI tam görüntü sınıflandırma:** İsteğe bağlı, kullanıcı eğitimiyle elde
  edilen EfficientNet-B0 `.pt`, ya da isteğe bağlı harici CLIP/ViT modeli.
- **Temkinli karar:** Kaynağına göre değişen yanlış AI alarmları için
  kalibrasyona açık *AI / Gerçek / Belirsiz* kararları. Kalibrasyonsuz
  başlangıç eşiği herhangi bir doğruluk garantisi değildir.
- **TruFor bölgesel harita:** Ayrı resmî repo ve güvenilir model ağırlıkları
  gerektiğinde kullanılabilir; otomatik olarak indirilmez.
- **Teknik inceleme:** SHA-256, EXIF, kaynak JPEG için ELA, gürültü izleri,
  kaba kopyalama benzerliği, isteğe bağlı `c2patool`.
- **Değerlendirme:** Etiketli gerçek/AI veri üzerinde gerçek yanlış alarm,
  AI yakalama, belirsiz karar ve kapsama metrikleri.

## Hızlı kurulum — Windows

1. [Python 3.11 veya 3.12](https://www.python.org/downloads/) yükleyin.
2. ZIP'i açın veya repoyu `git clone` ile indirin.
3. Proje dizininde **PowerShell** açıp aşağıdaki komutu uygulayın:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\KUR_WINDOWS.ps1
```

Bu komut yalnızca temel panel paketlerini kurar. Fotoğraf önizlemesi, teknik
haritalar ve dosya incelemesi için yeterlidir; AI modeli dahil değildir.

**AI sınıflandırma / eğitim için isteğe bağlı büyük kurulum:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\KUR_WINDOWS.ps1 -AI
```

Bu seçenek büyük PyTorch/model indirmeleri yapabileceği için onay ister.

**Başlat:** `BASLAT_WINDOWS.bat` dosyasına çift tıklayın, ardından
`http://localhost:8501` adresine gidin. Panel **varsayılan olarak sadece
127.0.0.1 üzerinden yerel erişime** ayarlanır.

## Hızlı kurulum — Linux / macOS

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

AI için `python -m pip install -r requirements-ai.txt` isteğe bağlıdır.
Torch/torchvision sürümlerini cihazın CPU/CUDA sistemine göre uyumlu kurun.
TruFor'un bağımlılıklarının ayrıca farklı sürüm gerektirebileceğini unutmayın.

## AI model ağırlığı nasıl eklenir?

1. Kendi hakkına sahip olduğunuz `.pt` checkpoint'i, eğitimden sonra
   **yerel olarak** `outputs/deepshield_ai_b0.pt` konumuna koyun. Bu
   checkpoint'in `training/train_classifier.py` biçimiyle uyumlu olması gerekir.
2. Modeli oluşturmak için [Eğitim ve değerlendirme kılavuzunu](docs/TRAINING.md) izleyin.
3. Başka kişilerin de doğrudan AI analizi yapmasını istiyorsanız, **haklarını
   doğruladığınız** ağırlıkları GitHub **Releases** altında ayrı paylaşabilir,
   README'ye sürüm/SHA-256 ile bağlantı ekleyebilirsiniz. Bu repo, model
   dosyasını otomatik indirmez; önceden verilmiş bir model ağırlığı olmadığından
   burada bir indirme URL'si yoktur.
4. Fotoğrafın gerçekten AI olup olmadığına ilişkin güvenilir ölçüm için
   eğitimde görülmemiş, farklı cihaz ve üreticileri kapsayan bir test kümesi
   gerekir; mevcut başarının bağımsız olarak doğrulandığı iddia edilmez.

## TruFor kurulumu (isteğe bağlı)

Resmî repo: https://github.com/grip-unina/TruFor

Resmî lisansı ve model ağırlığı koşullarını okuyun. TruFor'un kendi lisansında
**bilgilendirme ve kâr amacı gütmeyen kullanım** şartı vardır; MIT lisansı
ona uygulanmaz. TruFor'u `models/TruFor/` altına **ayrıca** kurun;
`.gitignore` bu klasörü GitHub'a göndermez. Yerel ağırlık ve Python
uyumluluk ayrıntıları için [TruFor kurulumunu](docs/TRUFOR.md) okuyun.

## Kullanım ve sonuçların anlamı

- Panelden fotoğrafı yükleyip **Analiz et** seçin. Başka kullanıcılar dosyayı
  elle klasöre kopyalamaz.
- **Ham AI sınıf skoru:** Model çıktısı; *görselin yüzde kaçının AI olduğu*
  veya *sonucun doğruluk olasılığı* değildir.
- **TruFor şüpheli alan yüzdesi:** Maske alanı; yalnız AI ile değiştirilmiş
  piksellerin yüzdesi değildir. Photoshop, sıkıştırma, doğal farklılıklar
  yanlış alarm üretebilir.
- **Belirsiz** sonuçları yanlış/eksik performans raporundan saklamayın.
- **TruFor olmadan** 'değiştirilen nesneler kesin saptandı' denmez.

## Test ve doğruluk ölçümü

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python scripts/check_release.py
```

Bu testler **kod davranışını** inceler, modelinizin gerçek dünyadaki başarısını
ölçmez. Ek PyTorch testleri, PyTorch kurulmadıysa atlanır.

Bağımsız test kümeniz hazırsa:

```bash
python training/evaluate_external.py --data datasets/external_eval --weights outputs/deepshield_ai_b0.pt --out outputs/external_eval.json
```

Kalibrasyon ve bağımsız test: [Eğitim ve doğruluk kılavuzu](docs/TRAINING.md).

## GitHub'a yayınlama

Yeni, boş bir GitHub deposu oluşturun. Proje kökünde:

```bash
git init
git branch -M main
git add .
git status
git diff --cached --name-only
git commit -m "Initial DeepShield release"
git remote add origin https://github.com/KULLANICI_ADIN/DeepShield.git
git push -u origin main
```

`KULLANICI_ADIN` yerine kendi hesabınızı yazın. **Push öncesinde** sahneye alınan dosyaların arasında kişisel fotoğraf, `.env`, veri kümesi ve model bulunmadığını kontrol edin. Var olan bir depoya yayınlıyorsanız `git init` ve `git remote add` komutlarını tekrar çalıştırmayın.

Uygulamanın gerçek doğruluk metriklerini yalnızca bağımsız testten sonra paylaşın. Üçüncü taraf kod, veri ve ağırlıklar için kullanım/dağıtım haklarını ayrıca inceleyin.

## GitHub'a yüklemeden önce

- `.gitignore` sayesinde `outputs/`, `datasets/`, `models/`, `*.pt`, `.env`
  ve kullanıcı fotoğrafları sürüm kontrolünün dışında tutulur.
- Önceden Git tarafından **izlenen** bir gizli dosya varsa `.gitignore` onu
  kaldırmaz. `git ls-files` ile kontrol edin; gerektiğinde
  `git rm --cached <dosya>` kullanın. Gerçek bir sır daha önce pushlandıysa
  gizli anahtarı iptal edip yenileyin.
- Kendi görüntülerinizin yayınlanmasına izin veren haklarınız olup olmadığını
  kontrol edin; `examples/` içindeki örnek görüntünün dağıtım hakkını yayın öncesinde doğrulayın.
- `SECURITY.md`, `PRIVACY.md`, `THIRD_PARTY_NOTICES.md` ve `LICENSE`
  dosyalarını okuyun; seçtiğiniz lisansı yayınlamadan önce onaylayın.
- GitHub Actions testleri `.github/workflows/ci.yml` ile her push/PR'da
  çalıştırılır. CI'ın yeşil olması *yüksek model doğruluğu* garantisi değildir.

## Kamuya açık sunucuya kurulum

**Bu kod bir demo/araştırma panelidir, üretim SaaS hizmeti değildir.**
Anonim ziyaretçilerin fotoğraf yüklediği hizmet için TLS, erişim kontrolü,
kullanıcı onayı, silme/saklama politikası, yük/süre kota sistemi ve çok-kullanıcılı
izolasyon gerekir. Sadece Streamlit portunu internete açmak yeterli değildir.

Varsayılan `DEEPSHIELD_LOCAL_ADMIN=0` güvenlik amaçlıdır. Gelişmiş yerel
yönetici dosya yollarını görmek için **yalnız güvenilir yerel ortamda**
`DEEPSHIELD_LOCAL_ADMIN=1` belirlenebilir. Bunu herkese açık sunucuda açmayın.

## Lisans ve sorumluluk

Orijinal DeepShield kaynak kodu: `LICENSE` (MIT). TruFor, hazır ağırlıklar,
veri kümeleri ve bağımlılıklar: **kendi ayrı koşulları** (`THIRD_PARTY_NOTICES.md`).
Kaynak, bölge tespiti ve sınıflandırma sonuçları kesin adli delil değildir.
