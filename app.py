"""Run: streamlit run app.py"""
import base64
import hashlib
import html
import logging
from pathlib import Path
import streamlit as st
from core.config import DISPLAY_TOPICS, THRESHOLD
from core.preprocessing import InvalidImage, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title='IndoMeme-XAI — Sistem Analisis Multimodal',
    page_icon='assets/logo.png',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Load Obsidian Dark Glass theme
css_path = Path(__file__).parent / 'assets/theme.css'
if css_path.is_file():
    st.html(f'<style>{css_path.read_text(encoding="utf-8")}</style>')

@st.cache_resource(show_spinner=False)
def cached_model():
    from models.loader import load_model
    return load_model()

def clear_result():
    st.session_state.pop('analysis', None)
    st.session_state.pop('analysis_error', None)
    st.session_state.pop('analysis_error_detail', None)

def remove_image():
    st.session_state['upload_version'] = st.session_state.get('upload_version', 0) + 1
    for key in ['ocr', 'caption', 'topics', 'image_digest', 'preset_active']:
        st.session_state.pop(key, None)
    st.session_state['preset_choice'] = '— Unggah Gambar Mandiri —'
    clear_result()

def _arc_gauge_html(pct, percentage_text, card_class):
    clamped = max(0.0, min(100.0, float(pct)))
    fill_deg = clamped * 1.8
    if card_class == 'hate':
        fill_color = '#fb7185'
    elif card_class == 'not-hate':
        fill_color = '#34d399'
    else:
        fill_color = '#475569'
    gradient = (
        f'conic-gradient(from 270deg, {fill_color} 0deg {fill_deg:.1f}deg, '
        f'rgba(148, 163, 184, 0.18) {fill_deg:.1f}deg 180deg, transparent 180deg)'
    )
    return f'''
    <div style="position:relative;width:152px;height:86px;overflow:hidden;flex-shrink:0;">
      <div style="width:152px;height:152px;border-radius:50%;background:{gradient};
                  -webkit-mask:radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 13px));
                  mask:radial-gradient(farthest-side, transparent calc(100% - 14px), #000 calc(100% - 13px));"></div>
      <div style="position:absolute;left:50%;bottom:10px;width:2px;height:76px;
                  transform-origin:bottom center;transform:translateX(-50%) rotate(-9deg);pointer-events:none;">
        <div style="width:3px;height:17px;background:#fbbf24;border-radius:2px;box-shadow:0 0 6px rgba(251,191,36,0.9);"></div>
      </div>
      <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;padding-bottom:6px;" class="probability-top">
        <b style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:800;color:#ffffff;line-height:1.05;">{percentage_text}</b>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:0.06em;">AMBANG 45%</span>
      </div>
    </div>
    '''

PRESETS = {
    '— Unggah Gambar Mandiri —': None,
    'Sampel 1 (Ujaran Kebencian - Politik)': {
        'file': Path(__file__).parent / 'assets/presets/sample1.png',
        'name': 'sample1_hate_political.png',
        'ocr': 'odvlan walaupun Rimsvek Omyo itu kurang memuaskan apalagi omke gas kalo kalian perhatikan baik-baik but nonton lagi sebenarnya tam vkr tam vkr memeruk Assalamualaikum',
        'caption': 'Visual ini memadukan ulasan teks informal yang disensor dengan adegan komedi karakter kartun, menciptakan kontras humoris dan ironi.',
        'topics': ['individual', 'political'],
    },
    'Sampel 2 (Konten Netral - Pop Culture)': {
        'file': Path(__file__).parent / 'assets/presets/sample2.png',
        'name': 'sample2_neutral_popculture.png',
        'ocr': 'kenapa di kamar mu ada baby oil ? padahal di tempat ini tidak ada bayi',
        'caption': 'Gambar menampilkan Spongebob mengajukan pertanyaan menjebak mengenai keberadaan baby oil, mengejutkan karakter di hadapannya.',
        'topics': ['gender'],
    },
    'Sampel 3 (Konten Netral - Satir Pancasila)': {
        'file': Path(__file__).parent / 'assets/presets/sample3.png',
        'name': 'sample3_neutral_pancasila.png',
        'ocr': "RELAX edgelord. It's called HARI KESAKTIAN PANCASILA",
        'caption': 'Visual ini menampilkan Superman mengklarifikasi perubahan tanggal dari 30 September ke 1 Oktober sebagai Hari Kesaktian Pancasila.',
        'topics': ['political'],
    },
    'Sampel 4 (Ujaran Kebencian - Gender)': {
        'file': Path(__file__).parent / 'assets/presets/sample4.png',
        'name': 'sample4_hate_gender.png',
        'ocr': 'Ketika ketemu cowo ganteng di jalan Gewe: Dia Liatin AKU TADI AAA GANTENG Dia Suka GW KECNYA DEH',
        'caption': 'Visual ini secara humoris merepresentasikan berbagai respons emosional wanita saat berpapasan dengan pria menarik, dari kekaguman hingga asumsi pribadi.',
        'topics': ['gender'],
    },
}

# Brand Logo (Option 2: Attention Patch Matrix)
logo_png_path = Path(__file__).parent / 'assets/logo.png'
if logo_png_path.is_file():
    logo_b64 = base64.b64encode(logo_png_path.read_bytes()).decode('ascii')
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="40" height="40" class="brand-logo-img" alt="IndoMeme Logo" />'
else:
    logo_svg_path = Path(__file__).parent / 'assets/logo.svg'
    if logo_svg_path.is_file():
        logo_b64 = base64.b64encode(logo_svg_path.read_bytes()).decode('ascii')
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="40" height="40" class="brand-logo-img" alt="IndoMeme Logo" />'
    else:
        logo_html = '<span style="color:#818cf8;font-size:22px;font-weight:bold;">◈</span>'

# 1. TOPBAR HEADER
st.html(f'''
<header class="topbar-container">
  <div class="brand-wrapper">
    <div class="brand-icon-box">{logo_html}</div>
    <div>
      <div class="brand-title">IndoMeme<span>-XAI</span></div>
      <div class="brand-sub">Obsidian AI Lab · Analisis Ujaran Kebencian Multimodal &amp; Explainable AI</div>
      <h1 class="sr-only">Analisis Meme</h1>
    </div>
  </div>
  <div class="top-specs-group">
    <div class="spec-chip ready"><span class="dot"></span><span>Sistem Siap</span></div>
    <div class="spec-chip">Model: <strong>ICAM (Varian C)</strong></div>
    <div class="spec-chip">Ambang Batas: <strong>0.45</strong></div>
    <div class="spec-chip">Protokol: <strong>SHA-256 Verified</strong></div>
  </div>
</header>
''')

# 2. TOOLBAR DATA SAMPEL
st.html('''
<div class="sample-bar-header">
  <strong>Pilih Data Sampel Benchmark:</strong>
  <span>Gunakan sampel dataset INDOMEME atau unggah gambar meme secara mandiri</span>
</div>
''')

preset_choice = st.selectbox(
    'Pilih Data Sampel:',
    list(PRESETS.keys()),
    key='preset_choice',
    label_visibility='collapsed'
)

preset_info = PRESETS.get(preset_choice)
if preset_info and preset_choice != st.session_state.get('preset_active'):
    st.session_state['preset_active'] = preset_choice
    st.session_state['ocr'] = preset_info['ocr']
    st.session_state['caption'] = preset_info['caption']
    st.session_state['topics'] = preset_info['topics']
    clear_result()
elif not preset_info and st.session_state.get('preset_active'):
    st.session_state.pop('preset_active', None)
    clear_result()

# 3. DASHBOARD DUA KOLOM SIMETRIS
left, right = st.columns([1.0, 1.0], gap='medium')
image = None
active_filename = None
active_size_mb = 0.0

with left, st.container(key='upload_card', border=True):
    st.html('''
    <div class="card-title-bar">
      <h2 class="card-title-text">Masukan Multimodal</h2>
      <span class="card-sub-text">Citra · Teks OCR · Topik</span>
    </div>
    ''')

    version = st.session_state.get('upload_version', 0)
    upload = st.file_uploader(
        'Pilih gambar meme',
        type=['jpg', 'jpeg', 'png', 'webp'],
        key=f'upload_{version}',
        label_visibility='collapsed',
        on_change=clear_result,
        help='Tarik gambar ke sini atau pilih dari perangkat. Maksimal 10 MB.'
    )

    if upload:
        data = upload.getvalue()
        active_filename = upload.name
        active_size_mb = len(data) / 1024 / 1024
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.get('image_digest'):
            clear_result()
            st.session_state['image_digest'] = digest
        try:
            image = validate_image(data, upload.name)
        except InvalidImage as exc:
            clear_result()
            st.error(str(exc))
    elif preset_info and preset_info['file'].is_file():
        data = preset_info['file'].read_bytes()
        active_filename = preset_info['name']
        active_size_mb = len(data) / 1024 / 1024
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.get('image_digest'):
            clear_result()
            st.session_state['image_digest'] = digest
        try:
            image = validate_image(data, preset_info['name'])
        except InvalidImage as exc:
            clear_result()
            st.error(str(exc))

    if image is not None:
        img_col, meta_col = st.columns([0.58, 0.42], gap='small')
        with img_col:
            st.image(image, caption=None, width='stretch')
        with meta_col:
            st.html(f'''
            <div style="background:rgba(15,23,42,0.62);border:1px solid rgba(148,163,184,0.16);border-radius:11px;padding:10px 12px;font-size:11.5px;display:flex;flex-direction:column;gap:6px;">
              <div>
                <div style="color:#94a3b8;font-size:10px;text-transform:uppercase;font-weight:700;">Berkas Aktif</div>
                <div class="filename" style="color:#f8fafc;font-weight:600;font-family:'JetBrains Mono',monospace;word-break:break-all;margin-top:2px;">{html.escape(active_filename)}</div>
              </div>
              <div style="display:flex;justify-content:space-between;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
                <span>Ukuran:</span><strong style="color:#e2e8f0;">{active_size_mb:.2f} MB</strong>
              </div>
              <div style="display:flex;justify-content:space-between;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
                <span>Resolusi:</span><strong style="color:#e2e8f0;">{image.width}×{image.height}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
                <span>Patch ViT:</span><strong style="color:#818cf8;">196 (14×14)</strong>
              </div>
            </div>
            ''')
            st.button('Hapus gambar', on_click=remove_image, key='remove', width='stretch')
    else:
        st.html('''
        <div class="file-meta" style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;margin:4px 0 8px;">
          <span>Belum ada gambar dipilih</span>
          <span>JPG / PNG / WEBP · Max 10 MB</span>
        </div>
        ''')

    with st.form('analysis_form', border=False):
        submitted = st.form_submit_button('Jalankan Analisis Model →', type='primary', width='stretch', disabled=image is None)
        with st.expander('Detail Masukan Multimodal', expanded=True):
            ocr = st.text_area('Teks meme (OCR)', placeholder='Tuliskan teks yang terdapat pada meme.', key='ocr', max_chars=12000, height=68)
            caption = st.text_area('Deskripsi gambar', placeholder='Jelaskan isi gambar secara singkat.', key='caption', max_chars=12000, height=68)
            topics = st.multiselect('Topik', DISPLAY_TOPICS, key='topics', placeholder='Pilih kategori topik')

    if submitted and image is not None:
        clear_result()
        try:
            from core.bootstrap import artifacts_present, ensure_environment_ready
            from core.config import load_protocol
            from core.inference import analyze

            # Validate research protocol up front before downloading/loading
            load_protocol()
            if not artifacts_present():
                with st.status('Menyiapkan artefak model ICAM di cloud (~1,39 GB, hanya unduhan pertama)...', expanded=True) as status:
                    log_area = st.empty()
                    def _on_progress(msg):
                        log_area.caption(msg)
                        status.update(label=msg)
                    ensure_environment_ready(progress_callback=_on_progress)
                    status.update(label='Artefak model & encoder terverifikasi.', state='complete', expanded=False)

            with st.spinner('Memuat model & menjalankan inferensi ICAM serta 4 metode XAI…'):
                bundle = cached_model()
                st.session_state['analysis'] = analyze(bundle, image, ocr, caption, topics)
        except Exception as exc:
            logger.exception('Analysis failed; no prediction returned')
            st.session_state['analysis_error'] = 'Analisis belum dapat dijalankan. Silakan coba kembali setelah layanan analisis tersedia.'
            st.session_state['analysis_error_detail'] = f'{type(exc).__name__}: {exc}'

result = st.session_state.get('analysis')
error = st.session_state.get('analysis_error')
error_detail = st.session_state.get('analysis_error_detail')

if error_detail:
    st.error(f'Analisis tidak dapat diselesaikan ({error_detail}).')

with right, st.container(key='result_card', border=True):
    st.html('''
    <div class="card-title-bar">
      <h2 class="card-title-text">Hasil Analisis Model</h2>
      <span class="card-sub-text">Keputusan &amp; Telemetri ICAM</span>
    </div>
    ''')

    if result:
        label = result['label']
        is_hate = label == 'Hate'
        card_class = 'hate' if is_hate else 'not-hate'
        pred_text = 'HATE SPEECH' if is_hate else 'NOT HATE'
        p_val = float(result['probability'])
        p = p_val * 100.0
        margin_pp = p - THRESHOLD * 100.0
        sign = '+' if margin_pp >= 0 else ''
        pred_desc = f'{"Melampaui" if is_hate else "Di bawah"} ambang batas 0.45 ({sign}{margin_pp:.2f}%)'
        percentage = f'{p:.2f}%'.replace('.', ',')
        fill_class = card_class
        metrics = result.get('metrics') or {}
        gate_ca = metrics.get('gate_ca')
        gate_inc = metrics.get('gate_inc')
        elapsed = metrics.get('elapsed_seconds')
        device_str = metrics.get('device', 'cpu').upper()
    else:
        card_class = 'idle'
        pred_text = '—'
        pred_desc = error or ('Gambar siap dianalisis.' if image else 'Pilih sampel atau unggah gambar untuk mulai.')
        p = 0.0
        p_val = 0.0
        percentage = '—'
        fill_class = 'idle'
        gate_ca = None
        gate_inc = None
        elapsed = None
        device_str = 'SIAP'

    gauge_html = _arc_gauge_html(p, percentage, card_class)

    if gate_ca is not None and gate_inc is not None:
        ca_pct = gate_ca * 100.0
        inc_pct = gate_inc * 100.0
        gate_section = f'''
        <div class="telemetry-block">
          <div class="scale-header" style="margin-bottom:4px;">
            <span>Rasio Gate Internal ICAM (<i>g</i>)</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#c7d2fe;">Cross-Attn {ca_pct:.1f}% vs Incongruity {inc_pct:.1f}%</span>
          </div>
          <div class="gate-bar-outer">
            <div class="gate-fill-ca" style="width:{ca_pct:.1f}%;"></div>
            <div class="gate-fill-inc" style="width:{inc_pct:.1f}%;"></div>
          </div>
          <div class="scale-footer-labels">
            <span>z_CA (Keselarasan Lintas Modal)</span>
            <span>z_INC (Pertentangan Semantik)</span>
          </div>
        </div>
        '''
    else:
        gate_section = '''
        <div class="telemetry-block" style="opacity:0.65;">
          <div class="scale-header" style="margin-bottom:4px;">
            <span>Rasio Gate Internal ICAM (<i>g</i>)</span>
            <span style="font-family:'JetBrains Mono',monospace;">Menunggu inferensi</span>
          </div>
          <div class="gate-bar-outer">
            <div class="gate-fill-ca" style="width:50%;opacity:0.25;"></div>
            <div class="gate-fill-inc" style="width:50%;opacity:0.25;"></div>
          </div>
        </div>
        '''

    runtime_text = f'{elapsed:.2f} detik ({device_str})' if elapsed is not None else 'Menunggu analisis'

    st.html(f'''
    <div class="decision-card {card_class}">
      <div style="min-width:180px;flex:1;">
        <div class="decision-label-sub">Klasifikasi Akhir</div>
        <div class="decision-title prediction {card_class}">{pred_text}</div>
        <div class="decision-desc">{html.escape(pred_desc)}</div>
        <div style="margin-top:8px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#94a3b8;">
          Skor Logit Softmax: <strong style="color:#f8fafc;">{p_val:.4f}</strong>
        </div>
      </div>
      {gauge_html}
    </div>

    <div class="telemetry-block">
      <div class="scale-header">
        <span>Kontinum Keyakinan Model</span>
        <span class="scale-threshold-tag">Ambang Batas: 0.45 (45%)</span>
      </div>
      <div class="scale-bar-outer">
        <div class="scale-bar-marker" title="Ambang Batas: 45%"></div>
        <div class="scale-bar-fill {fill_class}" style="width:{p:.2f}%;"></div>
      </div>
      <div class="scale-footer-labels">
        <span>0% (Not Hate)</span>
        <span style="color:#fbbf24;font-weight:700;">45% (Cutoff)</span>
        <span>100% (Hate)</span>
      </div>
    </div>

    {gate_section}

    <div class="specs-table-box">
      <div class="specs-table-title">Telemetri &amp; Spesifikasi Inferensi:</div>
      <div class="specs-row"><span>Enkoder Tekstual:</span><strong>IndoBERTweet (128 Token Max)</strong></div>
      <div class="specs-row"><span>Enkoder Visual:</span><strong>Vision Transformer (196 Patch)</strong></div>
      <div class="specs-row"><span>Mekanisme Fusi:</span><strong>Cross-Modal Incongruity-Aware</strong></div>
      <div class="specs-row"><span>Waktu Eksekusi:</span><strong>{html.escape(runtime_text)}</strong></div>
      <div class="specs-row"><span>Integritas Protokol:</span><strong style="color:#34d399;">Frozen SHA-256 Verified</strong></div>
    </div>
    ''')

# 4. DASBOR EXPLAINABLE AI
with st.container(key='xai_card', border=True):
    st.html('''
    <div class="card-title-bar">
      <h2 class="card-title-text">Interpretasi Model (Explainable AI)</h2>
      <span class="card-sub-text">Atribusi fitur tekstual &amp; fokus 196 patch visual</span>
    </div>
    ''')

    col_ref, col_tabs = st.columns([0.34, 0.66], gap='medium')

    with col_ref:
        st.html('''
        <div style="font-size:12px;font-weight:600;color:#cbd5e1;margin-bottom:6px;">
          Citra Asli Acuan:
        </div>
        ''')
        if image is not None:
            st.image(image, caption=None, width='stretch')
        else:
            st.html('''
            <div style="background:rgba(15,23,42,0.8);border:1px dashed rgba(148,163,184,0.22);border-radius:11px;height:210px;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:12px;text-align:center;padding:16px;">
              Citra acuan tampil setelah gambar dimuat
            </div>
            ''')
        st.html('''
        <div style="font-size:11px;color:#94a3b8;margin-top:6px;">
          Acuan pembanding posisi koordinat patch 14×14 visual.
        </div>
        ''')

    with col_tabs:
        tabs = st.tabs(['Integrated Gradients', 'Incongruity Heatmap', 'Cross-Attention', 'Attention Rollout'])
        method_keys = ['ig', 'incongruity', 'cross_attention', 'rollout']
        method_notes = {
            'ig': 'Token ungu mendorong prediksi ke arah Hate Speech; token hijau menahan ke arah Netral (50 langkah Riemann terhadap baseline [PAD]).',
            'incongruity': 'Modul Incongruity ICAM: area bercahaya rose-violet menandai patch visual dengan pertentangan semantik tertinggi terhadap teks meme.',
            'cross_attention': 'Atensi lintas modalitas (Text → Image): area gambar yang paling diperhatikan representasi teks [CLS].',
            'rollout': 'Propagasi atensi internal multi-layer murni dari Vision Transformer (12 blok encoder dengan matriks identitas residual).'
        }
        patch_peaks = (result.get('metrics') or {}).get('patch_peaks', {}) if result else {}

        for tab, method in zip(tabs, method_keys):
            with tab:
                attribution = result['xai'].get(method) if result else None
                if attribution is None:
                    message = 'Penjelasan metode ini tidak tersedia.' if result else 'Penjelasan muncul setelah analisis model dijalankan.'
                    st.html(f'''
                    <div style="border:1px dashed rgba(148,163,184,0.25);background:rgba(15,23,42,0.55);border-radius:12px;min-height:175px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:12px;text-align:center;padding:16px;">
                      {message}
                    </div>
                    ''')
                elif method == 'ig':
                    from xai.visualization import token_html, top_tokens_html
                    st.html(token_html(attribution) + top_tokens_html(attribution, top_k=5))
                else:
                    st.image(attribution, width='stretch')
                    if method in patch_peaks:
                        st.html(f'''
                        <div style="margin-top:8px;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:rgba(15,23,42,0.72);border:1px solid rgba(148,163,184,0.16);border-radius:9px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#cbd5e1;">
                          <span>Konsentrasi Massa Atensi Puncak:</span>
                          <strong style="color:#fda4af;">{patch_peaks[method] * 100:.2f}% pada 1 patch (rata-rata uniform: 0.51%)</strong>
                        </div>
                        ''')

                st.html(f'''
                <div class="xai-subnote-box">
                  <strong>Keterangan Atribusi:</strong> {method_notes.get(method, "")}
                </div>
                ''')
