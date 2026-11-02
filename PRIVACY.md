# Görüntü gizliliği ve veri saklama

DeepShield'ın mevcut Streamlit uygulaması görselleri analiz için çalıştığı
makinenin belleğine alır. Orijinal görüntü ile TruFor/C2PA geçici dosyaları
Python `TemporaryDirectory` kapsamında tutulur ve normal çıkışta silinir.
Streamlit cache'inde analiz sonuçları sunucu belleğinde bir süre kalabilir.
Uygulama kendiliğinden görüntüleri eğitim veri kümesine eklemez veya harici
bir AI API'sine **görsel yüklemek için** yazılmış değildir. Hazır model
seçenekleri Hugging Face üzerinden **model dosyaları indirebilir**.

Önemli sınırlar:

- Uygulama internete açık bir sunucuda çalıştırılırsa **kullanıcının görüntüsü
  bu sunucuya gönderilir**. Sunucu yöneticisi ağ/günlük/kopya/yedek
  politikalarından sorumludur.
- Bu paket kullanıcı hesabı, güvenli depolama, saklama süresi garantisi,
  kullanıcı verisi silme iş akışı, tarama/antivirüs, hız sınırı veya
  hukuki gizlilik onay mekanizması içermez.
- Analiz raporlarında EXIF, SHA-256, dosya adı ve kullanıcı tarafından verilen
  gerçeklik etiketi bulunabilir. Bunları paylaşmadan önce kontrol edin.
- Kişisel fotoğraf ve test dosyalarını **GitHub deposuna koymayın**.
- Kamusal bir hizmete açmadan önce HTTPS, ters proxy, kaynak/boyut sınırı,
  erişim kontrolü, uygun gizlilik bildirimi ve yükleme/silme politikasını
  ayrıca kurun. Streamlit demo paneli tek başına üretim sunucusu değildir.
