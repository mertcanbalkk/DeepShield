# Katkıda bulunma

1. Yeni özellikleri ayrı branch/PR'de önerin ve hangi sorunu çözdüğünü anlatın.
2. `python -m pytest -q tests` çalıştırın; ağır PyTorch modelleri ayrıca kurulmuşsa
   ilgili testler de çalışır.
3. Yeni bir AI doğruluk iddiası ekliyorsanız veri kaynağı, split, örnek sayısı,
   yanlış pozitif oranı, AI recall ve **belirsiz** kararların sayısını açıklayın.
4. Hiçbir gerçek kullanıcı görüntüsünü, kişisel veri setini, token'ı, `.pt`
   ağırlığını ve üçüncü taraf ticari lisanslı modeli PR'a dahil etmeyin.
5. Bir hata düzeltmesini mümkünse küçük bir otomatik testle doğrulayın.
6. TruFor ağırlıklarını veya başka model dosyalarını bu MIT repoya eklemeyin.
