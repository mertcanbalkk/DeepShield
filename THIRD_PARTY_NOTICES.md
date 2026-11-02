# Üçüncü taraf yazılım, ağırlık ve veri hakları

Bu depodaki **orijinal DeepShield kaynak kodu** için kök `LICENSE` (MIT) geçerlidir.
Bu lisans başka projelerin kütüphanelerini, ön eğitimli model ağırlıklarını,
veri setlerini veya kullanıcıların yüklediği fotoğrafları kapsamaz.

## TruFor

- Resmî kaynak: https://github.com/grip-unina/TruFor
- TruFor'un resmî README'si kullanımı **bilgilendirme ve kâr amacı gütmeyen**
  amaçlarla sınırlandırmaktadır. Kod ve ağırlıklar bu GitHub paketine dahil değildir.
- TruFor'u ticari hizmete dahil etmeden önce hak sahiplerinden gerekli izinleri alın.
- Koduna, ağırlığına veya lisansına otomatik olarak MIT uygulamayın.

## Açık kaynak Python bağımlılıkları

Streamlit, PyTorch/torchvision, OpenCV, Pillow, NumPy, transformers,
timm, safetensors, huggingface_hub vb. bağımlılıklar kendi lisanslarına
sahiptir. Paket sürümleri `requirements*.txt` dosyalarında açıklanmıştır;
kendi lisans dosyalarını kurulumdan sonra kontrol edin.

## Haricî AI modelleri ve veri kümeleri

Kodda bahsedilen Hugging Face model depoları, CIFAKE ve GenImage gibi veri
kümeleri için **her birinin lisansını, izin koşullarını ve kişisel veri durumunu**
ayrıca doğrulayın. Ağırlıklar ve veri kümeleri burada paylaşılmıyor.
DeepShield kendi eğitilmiş modelinin performansını ispat etmiş sayılmaz.

## Demo görüntüler

`examples/` içindeki şekiller, bu paket için prosedürel olarak oluşturulmuş
**sentetik arayüz örnekleridir**. Gerçek/AI sınıflandırıcı doğruluk testi
olarak kullanılamaz.
