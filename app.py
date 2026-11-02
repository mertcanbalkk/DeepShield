"""DeepShield v6.5 — stable panel, genuine model outputs and transparent diagnostics."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from forensic_engine import analyze_image, heatmap, marked_regions
from model_adapters import DEFAULT_CLASSIFIER, LEGACY_CLASSIFIER, classify_ai, classify_ai_modern, load_modern_model, run_trufor, trufor_available, load_custom_classifier, classify_custom
from reference_engine import compare_to_original, load_rgb as read_optional_reference
from provenance import inspect_provenance
from presentation_metrics import summarize_percentages, text_for_ai_percent, region_summaries, text_report, describe_clone_candidates
from training.real_guard import DEFAULT_AI_MIN_SCORE, load_guard
from upload_safety import validate_upload

PROJECT_ROOT = Path(__file__).resolve().parent
# Public-safe defaults. Only the deployment owner should enable LOCAL_ADMIN.
LOCAL_ADMIN = os.environ.get('DEEPSHIELD_LOCAL_ADMIN', '').strip() == '1'


def as_project_path(value: str) -> str:
    """Resolve user-supplied relative paths against the app, not process cwd."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


st.set_page_config(page_title='DeepShield 6.6 | AI Skoru ve Bölge Açıklaması', page_icon='🛡️', layout='wide')
st.title('🛡️ DeepShield 6.6 — Temkinli AI Tespiti ve Bölge Analizi')
st.caption('AI üretimi · AI ile bölgesel düzenleme · Photoshop · silme/ekleme · klonlama: ayrı ipuçları ve modellerle araştırılır.')
st.info('**Tek fotoğraf yeterli.** Bütün görüntünün AI sınıf puanı, şüpheli alan yüzdesi ve bölge konumları ayrı raporlanır. Bu paket model ağırlığı içermez: mevcut `outputs/deepshield_ai_b0.pt` korunmalı; bölge işaretlemesi için eğitilmiş TruFor ayrıca kurulmalıdır.')


def png(data: np.ndarray) -> bytes:
    out = BytesIO()
    Image.fromarray(data).save(out, 'PNG')
    return out.getvalue()


def categorical_ai(score: float | None) -> str:
    if score is None:
        return 'Sınıflandırma etiketi bilinmiyor'
    if score >= .85:
        return 'Ham skor yüksek, ancak bu hazır model için doğrulanmış eşik yok'
    if score <= .15:
        return 'Ham skor düşük, ancak bu hazır model için doğrulanmış eşik yok'
    return 'Karar için bağımsız kalibrasyon gerekli'


@st.cache_resource(show_spinner='Eğitilmiş AI sınıflandırma modeli yükleniyor…')
def load_classifier_old():
    from transformers import pipeline
    return pipeline('image-classification', model=LEGACY_CLASSIFIER, device=-1)

@st.cache_resource(show_spinner='2026 CLIP modeli indiriliyor/yükleniyor…')
def load_classifier_modern_cached():
    return load_modern_model()


@st.cache_resource(show_spinner='Kendi DeepShield modelin yükleniyor…')
def load_custom_cached(path: str):
    return load_custom_classifier(path)

@st.cache_data(show_spinner=False, max_entries=8)
def analyze_cached(content: bytes, name: str):
    return analyze_image(content, name)


with st.sidebar:
    st.header('Analiz araçları')
    mode = st.radio('İnceleme yöntemi', ['Tek görsel — ana analiz', 'Gelişmiş model ayarları'], index=0)
    st.divider()
    use_classifier = st.checkbox('Eğitilmiş AI üretimi sınıflandırıcısı', value=True,
                                 help='İlk kullanımda ayrıca model indirmesi gerekebilir. Tamamen AI üretimi için eğitilmiştir, bölgesel değişiklik bulmaz.')
    default_weights = as_project_path(os.environ.get('DEEPSHIELD_CUSTOM_WEIGHTS', 'outputs/deepshield_ai_b0.pt'))
    model_choice = st.selectbox('Sınıflandırma modeli', ['Kendi eğittiğim DeepShield modeli', '2026 CLIP · güncel üreticiler', 'ViT · eski temel model'],
                                index=0, help='Bu modeller yalnızca tüm görseli sınıflandırır. Kendi modelinde eğitim sonrasında doğrulama eşikleri ve JPEG dayanıklılık kontrolü vardır.')
    custom_path = (st.text_input('Kendi model ağırlıkları (.pt)', value=default_weights,
                                 help='Yalnızca yerel yönetici modunda dosya yolu düzenlenebilir.')
                   if LOCAL_ADMIN else default_weights)
    if not LOCAL_ADMIN:
        st.caption('Model yollarını yalnızca sunucu yöneticisi ortam değişkenleriyle belirler.')
    conservative_mode = st.checkbox('Gerçek fotoğraflara yanlış AI alarmını azalt (temkinli mod)', value=True,
                                    help='AI kararı için JPEG varyantları ve daha sıkı eşik gerekir. Yanlış AI alarmı sayısını artırmaz, ancak bazı AI fotoğrafları BELİRSİZ yapabilir.')
    guard_path = as_project_path(os.environ.get('DEEPSHIELD_REAL_GUARD', 'outputs/deepshield_real_guard.json'))
    second_opinion = st.checkbox('İkinci AI modelinin görüşünü de al (eski ViT; ekstra indirme)', value=False, help='İki model aynı kararı verse bile bu kanıt değildir. İkisi ayrışırsa belirsiz raporlanır.')
    repo_dir = as_project_path(os.environ.get('DEEPSHIELD_TRUFOR_DIR', 'models/TruFor'))
    weights = as_project_path(os.environ.get('DEEPSHIELD_TRUFOR_WEIGHTS', 'models/TruFor/TruFor_train_test/pretrained_models/trufor.pth.tar'))
    trufor_installed = trufor_available(repo_dir, weights)
    run_localizer = st.checkbox('TruFor ile şüpheli alan yerelleştirme', value=trufor_installed,
                               help='TruFor ayrı kurulur. Yüklü değilse analiz yalnızca diğer yöntemlerle yapılır.')
    gpu = False
    trust_checkpoint = False
    threshold = .55
    if mode == 'Gelişmiş model ayarları':
        if LOCAL_ADMIN:
            repo_dir = st.text_input('TruFor depo klasörü', value=repo_dir)
            weights = st.text_input('TruFor .pth.tar ağırlık dosyası', value=weights)
        else:
            st.caption('TruFor model dosyalarının konumunu sunucu yöneticisi ayarlar.')
        try:
            import torch
            cuda_ready = bool(torch.cuda.is_available())
        except (ImportError, OSError):
            cuda_ready = False
        gpu = st.checkbox('TruFor GPU (CUDA:0)', value=False, disabled=not cuda_ready)
        if not cuda_ready:
            st.caption('Bu Python ortamı yalnızca CPU kullanıyor; TruFor GPU seçeneği kapalı. Docker gerekli değil.')
        # Never give remote visitors control over unsafe pickle loading.
        if LOCAL_ADMIN:
            trust_checkpoint = st.checkbox('Resmî TruFor ağırlığına güveniyorum (eski checkpoint yüklemesine izin ver)',
                                           value=False, help='Yalnızca doğruladığın resmî ağırlık için kullan. '
                                            'Güvenilmeyen PyTorch checkpoint dosyası zararlı kod çalıştırabilir.')
        threshold = st.slider('Yerelleştirme modeli işaretleme eşiği', .2, .9, .55, .05)
    known_hash = st.text_input('Varsa daha önceden güvenilir kayıtta bulunan SHA-256 (isteğe bağlı)', value='', help='Şu anda hesaplanan hash ile karşılaştırılır. Kendiliğinden dosyanın geçmişini göstermez.')
    st.divider()
    st.caption('AI % skoru = sınıflandırıcının ham puanı (fotoğrafın kaç yüzdesinin AI olduğu değil). Bölge % = TruFor maskesinin kapladığı alan (AI tarafından değiştirildiği kanıtı değil).')
    if model_choice.startswith('Kendi') and not Path(as_project_path(custom_path)).is_file():
        st.warning('Kendi eğitilmiş modelin henüz bulunmuyor. Önce README’deki eğitim adımlarını tamamlamalısın. Hazır CLIP modeli yalnızca ayrı bir tahmin verir.')

uploaded = st.file_uploader('📁 İncelenecek tek fotoğrafı yükle', type=['jpg','jpeg','png','webp','tif','tiff'], accept_multiple_files=False)
example = st.checkbox('Örnek görseli göster (tek-görsel analiz)', value=False)
with st.expander('📎 Orijinal fotoğrafım da var (isteğe bağlı, zorunlu değil)'):
    reference_upload = st.file_uploader('Varsa karşılaştırılacak orijinal fotoğrafı seç', type=['jpg','jpeg','png','webp'], key='optional_original')

truth_label = st.selectbox('Test için: Bu fotoğrafın gerçek durumunu biliyor musun? (isteğe bağlı)', ['Bilmiyorum', 'Tamamen AI üretimi', 'Gerçek kamera fotoğrafı', 'Kısmen değiştirilmiş fotoğraf'], index=0, help='Bu bilgi analizin sonucunu değiştirmez. Modelin hatalarını ölçmek içindir.')
if uploaded is None and not example:
    st.markdown('**Kullanım:** Fotoğraf yükle → **Analiz et** → sonuç üstte, işaretli alanlar ve detaylı adli haritalar altta.')
    st.caption('Başlamak için yalnızca incelenecek fotoğraf gerekli; referans veya orijinal dosya istemiyoruz.')
    st.stop()
if uploaded is not None:
    content, name = uploaded.getvalue(), uploaded.name
else:
    fp = Path(__file__).parent / 'examples' / 'tek_gorsel_klonlama_ornegi.png'
    content, name = fp.read_bytes(), fp.name

# Validate contents BEFORE preview. The extension alone is not trustworthy.
try:
    validate_upload(content, name)
except (ValueError, OSError) as upload_error:
    st.error('Geçersiz görüntü: ' + str(upload_error))
    st.stop()
st.image(content, caption='İncelenecek görüntü (analiz öncesi)', width=520)
if not st.button('🔬 ANALİZ ET — TEK GÖRSEL', type='primary', width='stretch'):
    st.stop()

try:
    base = analyze_cached(content, name)
except Exception as e:
    st.error('Dosya okunamadı: ' + str(e))
    st.stop()

ai_result = None
ai_status = 'Çalıştırılmadı'
if use_classifier:
    with st.spinner('Eğitilmiş AI modelinden bütün-görüntü tahmini alınıyor…'):
        try:
            if model_choice.startswith('Kendi'):
                local_guard = None
                if conservative_mode:
                    try:
                        local_guard = load_guard(guard_path, as_project_path(custom_path))
                    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as guard_error:
                        st.warning('Gerçek fotoğraf kalibrasyon dosyası kullanılamıyor; geçici temkinli mod etkin. ' + str(guard_error)[:300])
                ai_result = classify_custom(base.rgb,
                                            load_custom_cached(as_project_path(custom_path)),
                                            conservative=conservative_mode,
                                            real_guard=local_guard)
            elif model_choice.startswith('2026'):
                ai_result = classify_ai_modern(base.rgb, load_classifier_modern_cached())
            else:
                ai_result = classify_ai(base.rgb, load_classifier_old(), model_id=LEGACY_CLASSIFIER)
            ai_status = 'Çalıştı (doğruluk bağımsız ölçülmedi)'
        except (Exception) as e:
            ai_status = 'Kullanılamıyor'
            st.warning('AI sınıflandırma modeli çalışmadı. Kendi modelini kullanıyorsan .pt yolunu kontrol et; hazır modeller yalnızca sen seçersen ayrıca indirilir.')
            if LOCAL_ADMIN:
                with st.expander('AI sınıflandırıcı teknik hatası'):
                    st.code(str(e)[-4000:])

secondary_result = None
secondary_error = None
if use_classifier and second_opinion and not model_choice.startswith('ViT'):
    try:
        with st.spinner('İkinci AI dedektörü yükleniyor…'):
            secondary_result = classify_ai(base.rgb, load_classifier_old(), model_id=LEGACY_CLASSIFIER)
    except Exception as e:
        secondary_error = str(e)

localization = None
localization_status = 'Çalıştırılmadı'
if run_localizer:
    if not trufor_available(as_project_path(repo_dir), as_project_path(weights)):
        localization_status = 'Ağırlık/depo eksik'
        st.warning('Bölgesel model **TruFor çalıştırılamıyor** (depo veya eğitilmiş ağırlık eksik). Bu durumda görselde neresi değişti diye renkli tahmin uydurmuyoruz. `docs/TRUFOR.md` dosyasını izle, sonra Gelişmiş model ayarlarında iki yolu belirt.')
    else:
        with st.spinner('TruFor modeli şüpheli alanları inceliyor…'):
            try:
                localization = run_trufor(base.rgb, as_project_path(repo_dir), as_project_path(weights), use_gpu=gpu, original_bytes=content, original_format=base.metadata['format'], trust_official_checkpoint=trust_checkpoint)
                localization_status = 'Çalıştı'
            except Exception as e:
                localization_status = 'Hata'
                st.error('TruFor modeli çalışamadı. Ayrıntıları aşağıda; bu durumda sahte bölge haritası üretmiyoruz.')
                if LOCAL_ADMIN:
                    with st.expander('TruFor teknik hata kaydı', expanded=True):
                        st.code(str(e)[-7000:])
                else:
                    st.caption('Teknik ayrıntılar sunucu yöneticisine özeldir.')

st.divider()
if reference_upload is not None:
    st.subheader('İsteğe bağlı: Orijinal ile gerçek farkların karşılaştırılması')
    try:
        compared = compare_to_original(base.rgb, read_optional_reference(reference_upload.getvalue()))
        a, b = st.columns(2)
        with a:
            st.image(compared['marked'], caption='KIRMIZI = dosyalar arasındaki ölçülen farklar (AI kanıtı değil)', width='stretch')
        with b:
            st.image(compared['heat_overlay'], caption='Orijinal ile sayısal fark haritası', width='stretch')
        st.caption(f"Farklı alan: %{compared['difference_percent']:.2f} · hizalama: {compared['alignment']}")
        st.download_button('⬇️ Orijinal karşılaştırma işaretini indir', png(compared['marked']),
                           file_name='deepshield_referans_farklari.png', mime='image/png')
    except Exception as e:
        st.warning('İsteğe bağlı karşılaştırma yapılamadı.' + ((' ' + str(e)[:400]) if LOCAL_ADMIN else ''))
if uploaded is not None:
    actual_format = base.metadata.get('format', '').upper()
    ext = Path(name).suffix.lower()
    if (ext in ('.jpg', '.jpeg') and actual_format == 'PNG') or (ext == '.png' and actual_format == 'JPEG'):
        st.info(f'Dosya adı {ext} uzantılı fakat dosyanın gerçek biçimi {actual_format}. Bu tek başına manipülasyon kanıtı değildir; ELA kontrolü gerçek biçime göre yapıldı.')
st.divider()
st.subheader('1. Sonuç — iki farklı yüzde')
# The regional MASK is computed once and reused for overlays, area and JSON.
marked = mask = regions = None
if localization is not None:
    marked, mask, regions = marked_regions(base.rgb, localization.mask_scores, threshold)
metrics = summarize_percentages(ai_result.ai_score if ai_result else None,
                                 ai_result.decision if ai_result else None, mask)
card1, card2 = st.columns(2)
with card1:
    st.metric('Ham AI SINIF SKORU (model çıktısı)', text_for_ai_percent(metrics.ai_class_score_percent))
    st.caption('Modelin ham AI sınıf puanı. %93,24 skor = görüntünün %93,24 AI ile oluşturulduğu DEĞİLDİR. Gerçek dünyaya kalibre edilmiş olasılık da değildir.')
    st.write('**Model kararı:** ' + metrics.ai_decision)
with card2:
    st.metric('ŞÜPHELİ ALAN ORANI (TruFor)',
              text_for_ai_percent(metrics.altered_area_percent))
    st.caption('Sadece eğitilmiş bölgesel model çalışırsa hesaplanır. Bu oran AI düzenlemesi yüzdesini kanıtlamaz; Photoshop ve başka müdahaleleri de kapsayan şüpheli piksellerdir.')
if ai_result and ai_result.decision and 'AI üretimi yönünde' in ai_result.decision:
    st.info('**Tamamen AI üretimi ihtimali:** Bu durumda tek bir "değiştirilmiş nesne" bulunması beklenmez. Bölge maskesi, AI ile üretilmiş kısımların yüzdesini ölçmez.')
if localization is None:
    st.warning('**Neresi değiştirilmiş?** Bölgesel modelin kullanılabilir sonucu yok; kurulum veya teknik hata ayrıntılarını yukarıdan kontrol et.')
st.divider()
st.subheader('2. Teknik ayrıntılar')
c1, c2, c3 = st.columns(3)
c1.metric('AI üretimi sınıflandırıcısı', ai_status)
c2.metric('Bölgesel tespit modeli', localization_status)
c3.metric('Klonlama benzerlik grubu', len(base.clone_pairs))
if ai_result is not None:
    st.markdown('**Model değerlendirmesi (tüm görüntü):** ' + (ai_result.decision or categorical_ai(ai_result.ai_score)))
    st.caption('Kullanılan model: ' + ai_result.model)
    if ai_result.raw_real_score is not None:
        st.metric('CLIP modelinin ham p(real) çıktısı', f'{ai_result.raw_real_score:.4f}')
        st.caption('Bu modelin dokümanındaki karar: p(real) < 0.91 → AI yönünde; 0.91–0.93 → belirsiz; ≥ 0.93 → gerçek yönünde. Bu değer kanıt veya gerçek dünyada doğruluk yüzdesi değildir.')
    elif ai_result.ai_score is not None:
        st.metric('AI sınıfının ham model skoru (0–1)', f'{ai_result.ai_score:.4f}')
    st.caption(ai_result.note)
    if model_choice.startswith('Kendi'):
        st.info('AI kararı orijinal/JPEG90/JPEG70 görüntülerindeki tutarlılığa ve karar eşiklerine dayanır. '
                    'Temkinli mod açıksa AI diye etiketlemek zorlaşır; ham % skor DEĞİŞMEZ ve başarı oranı değildir.')
        if conservative_mode:
            if local_guard:
                st.caption(f'✅ Ayrı gerçek fotoğraflarla ayarlanmış koruma: {local_guard["n_calibration_real"]} gerçek fotoğraf; AI kanıt eşiği >{local_guard["ai_min_score_must_exceed"]:.4f}. Bağımsız test hâlâ gerekli.')
            else:
                st.warning(f'Geçici temkinli AI eşiği >{DEFAULT_AI_MIN_SCORE:.2f} uygulanıyor. Gerçek fotoğraflarla kalibre edilmedi; bu eşik doğruluk garantisi değildir. Kalibrasyon için docs/TRAINING.md dosyasına bak.')
    st.dataframe(ai_result.labels, hide_index=True, width='stretch')
    if secondary_result is not None:
        st.markdown('**İkinci görüş (' + secondary_result.model + '):**')
        st.dataframe(secondary_result.labels, hide_index=True, width='stretch')
        def _direction(res):
            if res.decision:
                d=res.decision.lower()
                if 'belirsiz' in d: return 'belirsiz'
                if 'ai üretimi' in d: return 'ai'
                if 'gerçek fotoğraf' in d: return 'real'
            if res.ai_score is None: return 'belirsiz'
            return 'ai' if res.ai_score >= .85 else 'real' if res.ai_score <= .15 else 'belirsiz'
        d1,d2=_direction(ai_result),_direction(secondary_result)
        if d1 != d2:
            st.warning('Modeller arasında uyuşmazlık var. **SONUÇ: BELİRSİZ.** Farklı modellerden gelen ham skorları ortalayıp kesin tahmin üretmiyoruz.')
        elif d1 == 'belirsiz':
            st.warning('İki modelden en az biri açık karar veremiyor: **BELİRSİZ.**')
        else:
            st.info('Her iki model de ' + ('AI üretimi' if d1 == 'ai' else 'gerçek fotoğraf') + ' yönünde sinyal verdi. Bu, kesin adli kanıt değildir.')
    elif secondary_error:
        st.warning('İkinci görüş alınamadı.' + ((' ' + secondary_error[:300]) if LOCAL_ADMIN else ''))
else:
    st.error('**AI sınıflandırması yapılmadı.** Görüntü AI değildir/gerçektir diye etiketlenemez. AI modelinin gerçekten yüklenip yüklenmediğini ve paket kurulumunu kontrol et.')

if truth_label != 'Bilmiyorum':
    from datetime import datetime, timezone
    import hashlib
    feedback = {
        'tested_at_utc': datetime.now(timezone.utc).isoformat(),
        'filename': name,
        'sha256': hashlib.sha256(content).hexdigest(),
        'known_truth': truth_label,
        'model_used': (ai_result.model if ai_result else None),
        'model_decision': (ai_result.decision if ai_result else 'Model çalışmadı'),
        'raw_real_score': (ai_result.raw_real_score if ai_result else None),
        'raw_ai_score': (ai_result.ai_score if ai_result else None),
        'second_opinion': secondary_result.labels if secondary_result else None,
        'note': 'Örnek kullanıcı tarafından etiketlendi, gözden geçirilmeden eğitim/test metriğine dahil edilmemeli.'
    }
    st.download_button('⬇️ Yanlış tahmin örneğini / model test kaydını indir (JSON)',
                       json.dumps(feedback, ensure_ascii=False, indent=2),
                       file_name='deepshield_ornek_geri_bildirim.json', mime='application/json')
if localization is not None:
    st.markdown('**Bölgesel manipülasyon:** TruFor tarafından değerlendirildi; işaretlemeler aşağıdadır.')
    if localization.image_score is not None:
        st.metric('TruFor ham bütün-görüntü skoru', f'{localization.image_score:.3f}')
    st.caption(localization.note)
else:
    st.markdown('**Bölgesel manipülasyon:** Eğitilmiş yerelleştirme modeli çalıştırılmadı. **Neresi değişmiş sorusuna kesin cevap üretilmedi.**')

st.divider()
st.subheader('3. Görsel üzerinde şüpheli bölgeler')
region_details = (region_summaries(regions, mask.size, base.rgb.shape[1], base.rgb.shape[0])
                  if localization is not None else [])
if localization is not None:
    left, right = st.columns(2)
    with left:
        st.image(marked, caption='KIRMIZI: TruFor modelinin şüpheli bulduğu bölge; SARI: işaretleme sınırı', width='stretch')
    with right:
        st.image(heatmap(base.rgb, localization.mask_scores), caption='TruFor tahmin ısı haritası', width='stretch')
    st.write(f'Modelin işaretlediği bölge: {len(regions)} adet · görüntü alanının %{metrics.altered_area_percent:.2f} kadarı (manipülasyon şüphesi, AI kullanım kanıtı değil).')
    if region_details:
        st.markdown('### 📍 Şüpheli yerler — neresi işaretlendi?')
        for reg in region_details:
            st.markdown(f"**ALAN {reg['alan_no']} — {reg['konum'].upper()}** · Maske alanı: %{reg['bolge_alani_yuzde']:.2f} · "
                        f"Koordinatlar: x={reg['x']}, y={reg['y']}, {reg['genislik']}×{reg['yukseklik']} px")
        st.caption('Buradaki konum, kırmızı işaretli bölgenin geometrik yeridir. Nesnenin adı, AI düzenleme yöntemi veya kesin tahrifat kanıtı değildir.')
        st.dataframe(region_details, hide_index=True, width='stretch')
    else:
        st.info('Seçilen eşikte belirgin şüpheli bölge bulunmadı; bu sonuç görselin gerçek olduğunu kanıtlamaz.')
    st.download_button('⬇️ Türkçe bölge raporunu indir',
                       text_report(metrics.ai_decision, metrics.ai_class_score_percent,
                                   metrics.altered_area_percent, region_details),
                       file_name='deepshield_bolge_raporu.txt', mime='text/plain')
    st.download_button('⬇️ Kırmızı işaretli sonucu indir', png(marked), file_name='deepshield_trufor_isaretli.png', mime='image/png')
    st.download_button('⬇️ Bölgesel maske indir', png(mask), file_name='deepshield_trufor_maske.png', mime='image/png')
    if localization.conf_scores is not None:
        st.image(heatmap(base.rgb, localization.conf_scores), caption='TruFor güvenilirlik haritası (gerçek doğruluk garantisi değil)', width=600)
    if regions:
        st.markdown('**Şüpheli alan yakın görünümleri**')
        cols = st.columns(min(3, len(regions)))
        for i, reg in enumerate(regions[:12]):
            x,y,w,h = (reg[k] for k in ('x','y','width','height'))
            with cols[i % len(cols)]:
                st.image(base.rgb[y:y+h, x:x+w], caption=f"ALAN {i+1} — {region_details[i]['konum']} · x={x}, y={y}, {w}×{h}", width='stretch')
else:
    st.warning('Eğitilmiş bölgesel tespit modeli çalışmadı. Bu dosyada değiştirilen alanı güvenilir biçimde işaretleyemiyoruz. Alt bölüm sadece teknik tanısal haritalardır.')

if base.clone_pairs:
    st.markdown('**Olası kopyala–yapıştır / klonlama eşleşmeleri**')
    clone_marked = base.rgb.copy()
    for r in base.clone_pairs:
        for key in ('area_a','area_b'):
            x1,y1,x2,y2 = r[key]
            cv2.rectangle(clone_marked, (x1,y1), (x2,y2), (255,40,40), 3)
    st.image(clone_marked, caption='Kırmızı kutular: birbirine benzeyen uzak alanlar. Tekrarlanan doğal desen de olabilir.', width=800)
    for clone_description in describe_clone_candidates(base.clone_pairs, base.rgb.shape[1], base.rgb.shape[0]):
        st.write('📍 ' + clone_description)
    st.caption('Klonlama adayları sadece görüntü içindeki benzer bölgelerden çıkarılır; AI kullanımı kanıtı veya değişimin kesin yönü değildir.')
    st.dataframe(base.clone_pairs, width='stretch', hide_index=True)

st.divider()
st.subheader('4. Tanısal görüntü haritaları (tahrifat maskesi DEĞİL)')
st.caption('Bunlar eğitilmiş manipülasyon modelleri değil. Yüksek görsel yoğunluk, değiştirilmiş piksel oranı olarak yorumlanamaz.')
labels = {'noise':'Sensör/doku kalıntısı','ela':'ELA · JPEG yeniden sıkıştırma','jpeg_blocks':'JPEG blok tutarsızlığı','copy_move':'Klonlama benzerliği'}
cols = st.columns(2)
for i, (key, label) in enumerate(labels.items()):
    with cols[i%2]:
        st.markdown('**'+label+'**')
        if not base.available[key]:
            st.info('Bu dosya biçimine uygun değil — atlandı.')
        else:
            signal = base.maps[key]
            if key == 'copy_move' and not base.clone_pairs:
                st.image(base.rgb, caption='Güvenilir bir kopyalama-benzerliği kümesi oluşturulamadı.', width='stretch')
            else:
                st.image(heatmap(base.rgb, signal), caption=f'{label} — yalnızca görüntüleme. Kırmızı alan/AI kararı üretmez.', width='stretch')
        st.caption(base.explanations[key])

st.divider()
st.subheader('5. Kriptografik köken ve bütünlük kontrolü')
with st.spinner('SHA-256 ve C2PA kontrolü yapılıyor…'):
    try:
        provenance = inspect_provenance(content, known_hash)
    except ValueError as e:
        st.warning('SHA-256 karşılaştırması geçersiz; referans değeri 64 karakterlik hex olmalıdır. ' + str(e))
        provenance = inspect_provenance(content, None)
st.code('SHA-256: '+provenance.sha256, language=None)
cc1,cc2 = st.columns(2)
cc1.metric('C2PA doğrulaması', provenance.c2pa_status)
cc2.write('**Güvenilir referans karşılaştırması:** ' + provenance.reference_status)
if provenance.claim_generator:
    st.write('**Kayıt üreticisi (manifest iddiası):** ' + provenance.claim_generator)
if provenance.recorded_actions:
    st.write('**Manifestte bildirilen işlemler:** ' + ', '.join(provenance.recorded_actions))
for notice in provenance.notices:
    st.caption('ℹ️ ' + notice)
with st.expander('C2PA teknik raporu'):
    if provenance.c2pa_report is not None:
        st.json(provenance.c2pa_report)
    else:
        st.info('C2PA teknik raporu oluşturulmadı.')
st.divider()
st.subheader('6. Adli rapor ve EXIF')
st.json(base.metadata)
for hypothesis in base.hypotheses:
    st.write('• '+hypothesis)
for warning in base.warnings:
    st.caption('⚠ '+warning)
report = {
    'filename':name,'sha256':base.sha256, 'metadata':base.metadata,
    'provenance':provenance.to_dict(),
    'classification': ({'model':ai_result.model,'raw_labels':ai_result.labels,
                        'raw_ai_score':ai_result.ai_score, 'raw_real_score':ai_result.raw_real_score,
                        'model_decision':ai_result.decision} if ai_result is not None else {'status':ai_status}),
    'percentages': {'raw_ai_class_score_percent': metrics.ai_class_score_percent,
                    'suspicious_area_percent': metrics.altered_area_percent,
                    'warning': 'AI sınıf skoru ve bölge alanı farklıdır. Bölge maskesi AI yöntemi atfetmez.'},
    'localization': ({'model':localization.model, 'raw_image_score':localization.image_score,
                      'threshold':threshold, 'regions': region_details} if localization is not None
                     else {'status':localization_status}),
    'clone_candidates':base.clone_pairs,
    'caveats':base.warnings,
    'assessment':'Adli ön inceleme. Görselin sahte olduğu veya AI kullanıldığı hukuken/teknik olarak kanıtlanmamıştır.'
}
st.download_button('⬇️ JSON adli ön raporunu indir', json.dumps(report, ensure_ascii=False, indent=2),
                   file_name='deepshield_rapor.json', mime='application/json')
st.warning('**Sonuç sınırı:** Tek görüntüde bazı kusursuz müdahaleler iz bırakmayabilir. Model yanılabilir. Tamamen AI üretimi ile yalnızca AI rötuşunu kesin ayırmak her zaman mümkün değildir; yüksek riskli vakalarda bağımsız doğrulama ve uzman incelemesi gerekir.')
