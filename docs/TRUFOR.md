# TruFor — isteğe bağlı bölgesel analiz kurulumu

> **Bu ZIP içinde TruFor model ağırlıkları yoktur.** Veri seti indirmen gerekmiyor, ancak bölgesel yerelleştirme için **önceden eğitilmiş TruFor ağırlıklarını** kurman gerekiyor. TruFor, *şüpheli manipülasyonları* (AI veya geleneksel) bulmayı dener, **hangi kısmın AI tarafından üretildiğini kesin olarak belirlemez.** Lisansı bilgi amaçlı ve kâr amacı gütmeyen kullanımla sınırlıdır; kullanmadan önce resmî lisansı oku.

**Resmî kaynaklar:**
- https://github.com/grip-unina/TruFor
- https://github.com/grip-unina/TruFor/blob/main/TruFor_train_test/README.md
- Eğitimli ağırlıklar: Yukarıdaki ikinci linkteki *Inference using our provided weights* → **weights** bağlantısı (MD5: `7bee48f3476c75616c3c5721ab256ff8`).

## 1 — TruFor deposunu indir (Windows PowerShell)

```powershell
cd C:\DeepShield
git clone https://github.com/grip-unina/TruFor.git models/TruFor
```

`git` komutu bulunmazsa Git for Windows kurulumu gerekir. TruFor'un kendi bağımlılıkları, DeepShield'ın `requirements-ai.txt` paketinden ayrıdır. Resmî `trufor_conda.yaml` ile **Conda** ortamı oluşturulması öneriliyor. Windows yerel kurulumunda hata olursa WSL2 Ubuntu veya Docker/Linux kullan.

## 2 — Eğitilmiş ağırlıkları indir

TruFor README'deki **weights** dosyasını indir, arşivden çıkar ve şu dosyanın bulunduğunu doğrula:

```
C:\DeepShield\models\TruFor\TruFor_train_test\pretrained_models\trufor.pth.tar
```

İndirme linki ve dosya adı sürüme göre değişirse README'deki güncel yönergeyi takip et. TruFor resmî belgeleri MD5 değerini `7bee48f3476c75616c3c5721ab256ff8` olarak bildiriyor. **Bu MD5 değerinin hangi indirilen dosyaya ait olduğunu resmî kaynaktan doğrula; ZIP ve çıkarılmış modelin özeti farklıdır.** Çalıştırmadan önce depo lisansını kontrol et.

## 3 — Ortam bağımlılıkları

Resmî `TruFor_train_test/README.md` içindeki `conda env create -f trufor_conda.yaml` komutunu, deponun ilgili dizininden çalıştır. TruFor'un ortamını **ayrı** tutmak gereklidir. `model_adapters.py` TruFor subprocess çağrısında `sys.executable` kullandığından, TruFor farklı bir Conda ortamında kurulmuşsa bunu `DEEPSHIELD_TRUFOR_PYTHON` ortam değişkeniyle seçebilirsin (mevcut sürümde desteklenir).

Örnek:

```powershell
$env:DEEPSHIELD_TRUFOR_PYTHON = 'C:\path\to\trufor-env\python.exe'
$env:DEEPSHIELD_TRUFOR_DIR = 'C:\DeepShield\models\TruFor'
$env:DEEPSHIELD_TRUFOR_WEIGHTS = 'C:\DeepShield\models\TruFor\TruFor_train_test\pretrained_models\trufor.pth.tar'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Eğer tüm TruFor bağımlılıkları DeepShield'ın `.venv` ortamında da kurulmuşsa ayrı Python ayarına ihtiyaç yoktur.

## 4 — Kullanım

Gelişmiş model ayarlarında TruFor açık, repo ve ağırlık yolları doğru olsun. Sonuç ekranındaki iki sayı **farklıdır**:
- **AI sınıf skoru %**: tüm görselin AI/gerçek sınıflandırıcısının skoru; kalibre edilmiş olasılık değildir.
- **Şüpheli alan %**: TruFor maskesinde eşik üzerinde işaretli alanın toplam piksel alanına oranı; AI ile değiştirme yüzdesi değildir.

## Durum

Entegrasyon kodu otomatik test edildi, **bu pakette gerçek TruFor ağırlıklarıyla uçtan uca tespit testi yapılmadı**. Alan maskesinin gerçekten doğru olduğunu ayrıca gerçek fotoğraflar ve doğrulanmış değişiklik maskeleriyle ölçmek gerekir.
