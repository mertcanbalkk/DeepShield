# Güvenlik politikası

## Bildirim

Açık güvenlik sorunlarını herkese açık issue ile yaymadan önce depo sahibine
GitHub **Security → Advisories → Report a vulnerability** üzerinden özel
bildirim yapın (depo sahibi bu özelliği açmalıdır).

## Güvenli kullanım

- Varsayılan olarak `DEEPSHIELD_LOCAL_ADMIN=0`; web ziyaretçileri yerel
  model dosya yolunu değiştiremez ve iç hata dosyası yollarını göremez.
- `DEEPSHIELD_LOCAL_ADMIN=1` yalnızca kendi bilgisayarındaki, internete
  açık **olmayan** geliştirme ortamına mahsustur.
- Eski PyTorch checkpoint'lerini `weights_only=False` ile açmak zararlı
  pickle kodu çalıştırabilir. Yalnızca **resmî, doğrulanmış, güvenilir**
  TruFor ağırlıklarını bu yöntemle kullanın. Model dosyalarını anonim
  kullanıcılardan kabul etmeyin.
- `c2patool` harici çalıştırılabilir dosyadır; resmî ve güvenilir kaynaktan
  kurun. İhtiyaç yoksa kurmayın.
- GitHub'a `.env`, `.pt`, `.pth`, `datasets/`, `outputs/`, `models/`,
  rapor, özel görüntü veya erişim token'ı yüklemeyin.
- `git status`, `git ls-files` ve GitHub'ın secret scanning özellikleriyle
  yüklemeden önce depo içeriğini inceleyin. `.gitignore` daha önce izlenen
  dosyaları otomatik olarak silmez.
- Tek kullanıcı için geliştirilmiş bu Streamlit panelini herhangi bir
  kimlik doğrulaması/izolasyon eklemeden doğrudan internete açmayın.

## Kısıtlamalar

Dosya boyutu ve piksel sınırı bulunur; ancak antivirüs, kota, çok-kullanıcılı
izolasyon, merkezi güvenlik günlükleri veya kapsamlı zararlı dosya taraması
bulunmaz. Sınırlar bir üretim güvenlik denetimi yerine geçmez.
