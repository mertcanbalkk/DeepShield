# DeepShield'ı GitHub'da yayınlama kontrol listesi

## 1. Yayın öncesi incele

- [ ] `LICENSE` MIT koşullarını, repo sahibi olarak kabul ediyor musun? Başka yazarlardan alınan kodlar için yeterli hak var mı?
- [ ] `THIRD_PARTY_NOTICES.md`: TruFor **ayrı ve kâr amacı gütmeyen** şartlara bağlı, bu repoya eklenmeyecek.
- [ ] Repo içindeki örnek resimler sentetik; kendi/başkalarının özel fotoğrafları yok.
- [ ] `outputs/`, `datasets/`, `models/`, `.env`, `.venv/`, kişisel raporlar ve `.pt` dosyaları git'e girmedi.
- [ ] En az bir Windows bilgisayarda yerel kurulum ve gerçek fotoğraf testi yapıldı; bunun sonuçlarını README'ye doğrulanmış metriklerle ekle.
- [ ] Eğitilmiş ağırlıkların **repo içinde olmadığını** ve ayrı paylaşım varsa kullanım haklarının doğrulandığını açıkça belirttin.
- [ ] Bir GitHub Releases ağırlığı yayımlıyorsan eğitim kaynak lisansı, SHA-256, model kartı ve bağımsız test metrikleri ekle.
- [ ] GitHub Actions testleri yeşil (bunlar model doğruluğunu doğrulamaz).

## 2. İlk kez yeni GitHub deposuna yükle

GitHub'da boş bir depo oluştur. ZIP'ten çıkardığın DeepShield klasöründe:

```powershell
git init
git branch -M main
git add .
git status
git diff --cached --name-only
git commit -m "Prepare DeepShield public research release"
git remote add origin https://github.com/KULLANICI_ADIN/DeepShield.git
git push -u origin main
```

`KULLANICI_ADIN` yerine kendi GitHub kullanıcı adını yaz. Kodu göndermeden önce `git diff --cached --name-only` çıktısını **mutlaka** kontrol et.

**Var olan repoya güncelleme yapıyorsan** `git init`, `git remote add origin` adımlarını tekrar etme; önce mevcut branch ve uzaktaki değişiklikleri kontrol et.

## 3. Bilgisayarındaki eski proje ile karşılaştır

Yeni ZIP bir **GitHub dağıtım kopyasıdır**, yerel `C:\Users\...\DeepShield` klasöründeki `.pt`, `datasets` veya TruFor ağırlıklarını içermez. Yayın kopyasını eski projeyle rastgele üzerine yazmak yerine `git diff` veya dosya karşılaştırmasıyla değişiklikleri aktar. Windows modellerini yayına ekleme.

## 4. Gerçek kullanıcılar için sonraki adım

Bu sürüm bir araştırma demosudur; kimlik doğrulama, çok kullanıcılı görev kuyruğu, saklama/silme politikası, sunucu kota yönetimi veya yetkili model dağıtımı değildir. İleride kamuya açık hizmet vermeden bunları ekle.
