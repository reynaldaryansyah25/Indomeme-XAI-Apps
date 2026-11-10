"""Run: streamlit run app.py"""
import base64
import hashlib
import html
import logging
from pathlib import Path
import streamlit as st
from core.config import DISPLAY_TOPICS
from core.preprocessing import InvalidImage, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title='IndoMeme-XAI — Sistem Analisis Multimodal',
    page_icon='assets/logo.png',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Load custom styling
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

def remove_image():
    st.session_state['upload_version'] = st.session_state.get('upload_version', 0) + 1
    for key in ['ocr', 'caption', 'topics', 'image_digest', 'preset_active']:
        st.session_state.pop(key, None)
    st.session_state['preset_choice'] = '— Unggah Gambar Mandiri —'
    clear_result()

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

# Logo (Option 2: Attention Patch Matrix)
logo_png_path = Path(__file__).parent / 'assets/logo.png'
if logo_png_path.is_file():
    logo_b64 = base64.b64encode(logo_png_path.read_bytes()).decode('ascii')
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="38" height="38" class="brand-logo-img" alt="IndoMeme Logo" />'
else:
    logo_svg_path = Path(__file__).parent / 'assets/logo.svg'
    if logo_svg_path.is_file():
        logo_b64 = base64.b64encode(logo_svg_path.read_bytes()).decode('ascii')
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" width="38" height="38" class="brand-logo-img" alt="IndoMeme Logo" />'
    else:
        logo_html = '<span style="color:#a855f7;font-size:22px;font-weight:bold;">◈</span>'

# 1. TOPBAR HEADER (v3.0 dihapus sesuai permintaan)
st.html(f'''
<header class="topbar-container">
  <div class="brand-wrapper">
    <div class="brand-icon-box">{logo_html}</div>
    <div>
      <div class="brand-title">IndoMeme<span>-XAI</span></div>
      <div class="brand-sub">Sistem Analisis Ujaran Kebencian Multimodal &amp; Interpretasi XAI</div>
      <h1 class="sr-only">Analisis Meme</h1>
    </div>
  </div>
  <div class="top-specs-group">
    <div class="spec-chip ready"><span class="dot"></span><span>Sistem Siap</span></div>
    <div class="spec-chip">Model: <strong>ICAM (Varian C)</strong></div>
    <div class="spec-chip">Ambang Batas: <strong>0.45</strong></div>
  </div>
</header>
''')

# 2. TOOLBAR DATA SAMPEL
st.html('''
<div class="sample-bar-header">
  <strong>Pilih Data Sampel:</strong>
  <span>Pilih sampel uji cepat atau gunakan mode unggah mandiri</span>
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

# 3. DASHBOARD DUA KOLOM
left, right = st.columns([1.02, 0.98], gap='medium')
image = None

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
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.get('image_digest'):
            clear_result()
            st.session_state['image_digest'] = digest
        try:
            image = validate_image(data, upload.name)
            st.image(image, caption=None, width='stretch')
            st.html(f'''
            <div class="file-meta" style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;margin-top:8px;">
              <span class="filename" style="font-weight:600;color:#0f172a;">{html.escape(upload.name)}</span>
              <span>{len(data)/1024/1024:.2f} MB</span>
            </div>
            ''')
            st.button('Hapus gambar', on_click=remove_image, key='remove')
        except InvalidImage as exc:
            clear_result()
            st.error(str(exc))
    elif preset_info and preset_info['file'].is_file():
        data = preset_info['file'].read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != st.session_state.get('image_digest'):
            clear_result()
            st.session_state['image_digest'] = digest
        try:
            image = validate_image(data, preset_info['name'])
            st.image(image, caption=None, width='stretch')
            st.html(f'''
            <div class="file-meta" style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;margin-top:8px;">
              <span class="filename" style="font-weight:600;color:#0f172a;">{html.escape(preset_info["name"])}</span>
              <span>{len(data)/1024/1024:.2f} MB</span>
            </div>
            ''')
            st.button('Hapus gambar', on_click=remove_image, key='remove')
        except InvalidImage as exc:
            clear_result()
            st.error(str(exc))
    else:
        st.html('''
        <div class="file-meta" style="display:flex;justify-content:space-between;font-size:11.5px;color:#94a3b8;margin-top:8px;">
          <span>Belum ada gambar</span>
          <span>Max. 10 MB</span>
        </div>
        ''')

    with st.form('analysis_form', border=False):
        submitted = st.form_submit_button('Jalankan Analisis Model →', type='primary', width='stretch', disabled=image is None)
        with st.expander('Detail Masukan Multimodal', expanded=True):
            ocr = st.text_area('Teks meme (OCR)', placeholder='Tuliskan teks yang terdapat pada meme.', key='ocr', max_chars=12000)
            caption = st.text_area('Deskripsi gambar', placeholder='Jelaskan isi gambar secara singkat.', key='caption', max_chars=12000)
            topics = st.multiselect('Topik', DISPLAY_TOPICS, key='topics', placeholder='Pilih kategori topik')

    if submitted and image is not None:
        clear_result()
        try:
            with st.spinner('Menjalankan analisis model ICAM…'):
                from core.inference import analyze
                bundle = cached_model()
                st.session_state['analysis'] = analyze(bundle, image, ocr, caption, topics)
        except Exception:
            logger.exception('Analysis failed; no prediction returned')
            st.session_state['analysis_error'] = 'Analisis belum dapat dijalankan. Silakan coba kembali setelah layanan analisis tersedia.'

result = st.session_state.get('analysis')
error = st.session_state.get('analysis_error')

with right, st.container(key='result_card', border=True):
    st.html('''
    <div class="card-title-bar">
      <h2 class="card-title-text">Hasil Analisis Model</h2>
      <span class="card-sub-text">Keputusan Klasifikasi</span>
    </div>
    ''')

    if result:
        label = result['label']
        is_hate = label == 'Hate'
        card_class = 'hate' if is_hate else 'not-hate'
        pred_text = 'HATE SPEECH' if is_hate else 'NOT HATE'
        pred_desc = 'Probabilitas melampaui ambang batas 0.45' if is_hate else 'Probabilitas berada di bawah ambang batas 0.45'
        p = result['probability'] * 100
        p_val = result['probability']
        percentage = f'{p:.2f}%'.replace('.', ',')
        fill_class = 'hate' if is_hate else 'not-hate'
    else:
        card_class = 'idle'
        pred_text = '—'
        pred_desc = error or ('Gambar siap dianalisis.' if image else 'Pilih sampel atau unggah gambar untuk mulai.')
        p = 0
        p_val = 0.0
        percentage = '—'
        fill_class = 'idle'

    st.html(f'''
    <div class="decision-card {card_class}">
      <div>
        <div class="decision-label-sub">Klasifikasi</div>
        <div class="decision-title prediction {card_class}">{pred_text}</div>
        <div class="decision-desc">{pred_desc}</div>
      </div>
      <div class="decision-metric-box probability-top">
        <div class="decision-metric-title">Probabilitas</div>
        <div class="decision-metric-val"><b>{percentage}</b></div>
        <div class="decision-metric-sub">Nilai: {p_val:.4f}</div>
      </div>
    </div>

    <div class="scale-wrapper">
      <div class="scale-header">
        <span>Tingkat Keyakinan (Probabilitas Hate)</span>
        <span class="scale-threshold-tag">Ambang Batas: 0.45 (45%)</span>
      </div>
      <div class="scale-bar-outer">
        <div class="scale-bar-marker" title="Ambang Batas: 45%"></div>
        <div class="scale-bar-fill {fill_class}" style="width: {p}%;"></div>
      </div>
      <div class="scale-footer-labels">
        <span>0% (Not Hate)</span>
        <span style="color:#e11d48;font-weight:700;">45% (Ambang)</span>
        <span>100% (Hate)</span>
      </div>
    </div>

    <div class="specs-table-box">
      <div class="specs-table-title">Spesifikasi Inferensi:</div>
      <div class="specs-row"><span>Enkoder Tekstual:</span><strong>IndoBERTweet (128 Token Max)</strong></div>
      <div class="specs-row"><span>Enkoder Visual:</span><strong>Vision Transformer (196 Patch)</strong></div>
      <div class="specs-row"><span>Mekanisme Fusi:</span><strong>Cross-Modal Incongruity-Aware</strong></div>
      <div class="specs-row"><span>Status Verifikasi:</span><strong>Protokol Frozen SHA-256 OK</strong></div>
    </div>
    ''')

# 4. DASBOR EXPLAINABLE AI
with st.container(key='xai_card', border=True):
    st.html('''
    <div class="card-title-bar">
      <h2 class="card-title-text">Interpretasi Model (Explainable AI)</h2>
      <span class="card-sub-text">Atribusi fitur tekstual dan fokus patch citra</span>
    </div>
    ''')

    col_ref, col_tabs = st.columns([0.34, 0.66], gap='medium')

    with col_ref:
        st.html('''
        <div style="font-size:12px;font-weight:600;color:#334155;margin-bottom:6px;">
          Citra Asli Acuan:
        </div>
        ''')
        if image is not None:
            st.image(image, caption=None, width='stretch')
        else:
            st.html('''
            <div style="background:#0f172a;border-radius:10px;height:220px;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:12px;text-align:center;padding:16px;">
              Citra acuan tampil setelah gambar dimuat
            </div>
            ''')
        st.html('''
        <div style="font-size:11px;color:#94a3b8;margin-top:6px;">
          Digunakan untuk pembanding posisi koordinat patch visual.
        </div>
        ''')

    with col_tabs:
        tabs = st.tabs(['Integrated Gradients', 'Incongruity Heatmap', 'Cross-Attention', 'Attention Rollout'])
        method_keys = ['ig', 'incongruity', 'cross_attention', 'rollout']
        method_notes = {
            'ig': 'Token dengan warna ungu pekat memiliki kontribusi nilai gradien positif tertinggi terhadap keputusan klasifikasi.',
            'incongruity': 'Modul Incongruity ICAM: patch visual dengan warna pekat mengindikasikan pertentangan semantik kuat terhadap teks meme.',
            'cross_attention': 'Atensi lintas modalitas: area citra yang paling diperhatikan oleh representasi teks meme.',
            'rollout': 'Distribusi atensi visual internal: propagasi perhatian antar-layer murni dari Vision Transformer.'
        }

        for tab, method in zip(tabs, method_keys):
            with tab:
                attribution = result['xai'].get(method) if result else None
                if attribution is None:
                    message = 'Penjelasan metode ini tidak tersedia.' if result else 'Penjelasan muncul setelah analisis model dijalankan.'
                    st.html(f'''
                    <div style="border:1px dashed #cbd5e1;background:#f8fafc;border-radius:10px;min-height:160px;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:12px;text-align:center;padding:16px;">
                      {message}
                    </div>
                    ''')
                elif method == 'ig':
                    from xai.visualization import token_html
                    st.html(token_html(attribution))
                else:
                    st.image(attribution, width='stretch')

                st.html(f'''
                <div class="xai-subnote-box">
                  <strong>Keterangan Atribusi:</strong> {method_notes.get(method, "")}
                </div>
                ''')
