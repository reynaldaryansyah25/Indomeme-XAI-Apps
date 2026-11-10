<div align="center">

<img src="assets/logo.png" alt="IndoMeme-XAI Logo" width="88" height="88" style="border-radius: 18px; margin-bottom: 12px;" />

# IndoMeme-XAI
### Sistem Analisis Ujaran Kebencian Multimodal & Interpretasi Explainable AI

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-ICAM%20Varian%20C-6366F1)](models/icam.py)
[![Verification](https://img.shields.io/badge/Protocol-Frozen%20SHA--256-059669)](research/frozen_artifacts/protocol_v3.yaml)
[![Status](https://img.shields.io/badge/System-Production--Ready-059669)](#)

<p align="center">
  <b>IndoMeme-XAI</b> adalah platform analitik cerdas multimodal mutakhir yang dirancang untuk mendeteksi ujaran kebencian (*hate speech*) pada meme berbahasa Indonesia secara objektif, presisi, dan dapat diinterpretasikan secara ilmiah menggunakan teknik <i>Explainable AI</i> (XAI).
</p>

</div>

---

## 📌 Ringkasan Sistem

Deteksi kebencian pada meme merupakan permasalahan klasifikasi multimodal yang kompleks karena konteks semantik sering kali lahir dari pertentangan (*incongruity*) atau interaksi antara pesan teks dan visual. IndoMeme-XAI mengimplementasikan arsitektur **ICAM (Incongruity-Aware Cross-Modal Attention) Varian C** yang menggabungkan representasi tekstual kontekstual bahasa Indonesia dengan fitur visual berbasis patch Vision Transformer, dilengkapi verifikasi protokol beku (*frozen SHA-256*) dan empat metode interpretasi XAI komprehensif.

```
                  ┌──────────────────────┐
                  │      MEME INPUT      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │ Citra Meme (RGB)│               │  Teks & Topik   │
   └────────┬────────┘               └────────┬────────┘
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │ Vision Transf.  │               │  IndoBERTweet   │
   │  (ViT-B/16)     │               │  (128 Tokens)   │
   └────────┬────────┘               └────────┬────────┘
            │                                 │
            └────────────────┬────────────────┘
                             ▼
              ┌─────────────────────────────┐
              │  ICAM Fusion (Varian C)     │
              │  - Incongruity-Aware Matrix │
              │  - Cross-Modal Attention    │
              └──────────────┬──────────────┘
                             ▼
              ┌─────────────────────────────┐
              │   Klasifikasi & Ambang      │
              │   (Threshold: 0.45)         │
              └──────────────┬──────────────┘
                             ▼
       ┌───────────────────────────────────────────┐
       │             SUITE EXPLAINABLE AI          │
       ├─────────────────────┬─────────────────────┤
       │ Integrated Gradient │ Incongruity Heatmap │
       │ Cross-Attention Map │ Attention Rollout   │
       └─────────────────────┴─────────────────────┘
```

---

## ✨ Fitur Utama

- **Dual-Stream Specialized Encoders**:
  - **Tekstual**: `indolem/indobertweet-base-uncased` (diproses hingga 128 subword token dengan normalisasi *slang* NFKC).
  - **Visual**: `google/vit-base-patch16-224-in21k` (resolusi 224×224 px menghasilkan 196 patch visual).
- **Mekanisme Fusi ICAM (Varian C)**:
  - Fusi lintas modalitas berbasis atensi silang yang secara eksplisit memetakan pertentangan semantik teks dan gambar.
- **Suite Explainable AI (XAI) Terpadu**:
  1. **Integrated Gradients (50 Langkah Riemann)**: Mengatribusikan kontribusi nilai gradien kata/token terhadap keputusan model terhadap baseline token `[PAD]`.
  2. **Incongruity Heatmap**: Visualisasi aktivasi spasial 14×14 patch visual yang memiliki pertentangan semantik paling tajam terhadap teks.
  3. **Cross-Attention Map**: Distribusi bobot atensi timbal-balik antara representasi token `[CLS]` teks terhadap patch citra.
  4. **Attention Rollout**: Propagasi atensi visual murni multi-layer dari Vision Transformer.
- **Deterministic & Frozen Research Protocol**:
  - Pemuatan checkpoint bobot terverifikasi SHA-256 (`strict=True`, format `trainable_state_dict_v1`).
  - Tidak ada data leakage, hallucination, ataupun mock data.
- **Pustaka Sampel Benchmark INDOMEME**:
  - Dilengkapi sampel preset resmi dari dataset acuan (mencakup kasus ujaran kebencian, konten netral pop culture, dan satir politik).
- **Antarmuka Pengguna Modern & Profesional**:
  - UI responsif dua kolom berbasis komponen kustom non-AI slop, indikator metrik presisi, kartu keputusan dengan penanda ambang batas 45%, dan tata letak anti-*offside*.

---

## 🚀 Panduan Memulai

### Prasyarat Sistem
- **Sistem Operasi**: Windows 10/11, Linux, atau macOS.
- **Python**: Versi `3.12.x` (disarankan).
- **GPU (Opsional)**: NVIDIA GPU dengan dukungan CUDA (otomatis mendeteksi CPU jika CUDA tidak tersedia).

### 1. Instalasi Otomatis

Buka terminal PowerShell pada direktori repositori:

```powershell
# Untuk perangkat dengan NVIDIA CUDA:
.\setup.ps1

# Untuk perangkat tanpa GPU (CPU Only):
.\setup.ps1 -Cpu
```

Skrip `setup.ps1` akan otomatis:
1. Membentuk virtual environment `.venv`.
2. Memasang pustaka dependensi resmi (`requirements.txt`).
3. Mengunduh snapshot lokal encoder penelitian via `scripts/download_encoders.py`.
4. Memvalidasi integritas hash checkpoint dan protokol eksperimental.
5. Menjalankan verifikasi unit testing dasar.

### 2. Menjalankan Aplikasi

Jalankan server aplikasi Streamlit:

```powershell
# Menjalankan launcher bawaan:
.\run.ps1

# Atau paksa eksekusi CPU:
.\run.ps1 -Device cpu
```

Buka peramban di alamat: **`http://localhost:8501`**

*Alternatif eksekusi langsung:*
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 🧪 Pengujian & Verifikasi Kualitas

Repositori ini menerapkan kontrak pengujian ketat guna menjamin reproduksibilitas ilmiah:

```powershell
# 1. Menjalankan seluruh pengujian unit (Encoder, Upload, & Contract Integrity):
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Menjalankan verifikasi alur inferensi & toleransi numerik XAI:
.\.venv\Scripts\python.exe tests/research_acceptance.py

# 3. Uji integrasi antarmuka peramban (Opsional via Playwright & MS Edge):
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tests/browser_smoke.py
```

---

## 📂 Struktur Repositori

```text
Indomeme-XAI-Apps/
├── app.py                      # Aplikasi utama dashboard Streamlit
├── assets/                     # Aset visual sistem
│   ├── logo.png                # Brand vector logo (Attention Patch Matrix)
│   ├── theme.css               # Desain antarmuka profesional & anti-overflow
│   └── presets/                # Sampel citra nyata dari dataset benchmark
├── core/                       # Inti pemrosesan multimodal
│   ├── config.py               # Konfigurasi runtime & skema topik
│   ├── encoders.py             # Loader tokenizer & transformer
│   ├── inference.py            # Orkes inferensi pipeline end-to-end
│   ├── precision.py            # Manajemen FP16/FP32 autocast
│   ├── preprocessing.py        # Normalisasi teks slang & validasi citra
│   └── text.py                 # Pembentukan controlled text prompt
├── models/                     # Arsitektur jaringan neural
│   ├── icam.py                 # Arsitektur ICAM (Varian C) PyTorch
│   ├── loader.py               # Loader aman weights_only dengan cache lock
│   └── checkpoints/            # Bobot model terlatih (best_model.pt)
├── research/                   # Kontrak artefak penelitian
│   ├── frozen_artifacts/       # Protokol beku protocol_v3.yaml & hash
│   └── reference_notebooks/    # Notebook acuan pelatihan & evaluasi
├── scripts/
│   └── download_encoders.py    # Skrip sinkronisasi snapshot model HuggingFace
├── tests/                      # Suite pengujian otomatis
│   ├── test_pipeline.py        # Pengujian upload, kontrak, dan UI Streamlit
│   ├── test_encoder_integrity.py # Verifikasi SHA-256 encoder
│   ├── research_acceptance.py  # Uji paritas numerik representatif
│   └── browser_smoke.py        # Uji end-to-end headless browser
├── requirements.txt            # Dependensi produksi
├── requirements-dev.txt        # Dependensi pengujian & browser test
├── setup.ps1                   # Skrip otomatisasi setup environment
└── run.ps1                     # Launcher aplikasi
```

---

## ⚙️ Spesifikasi Teknis & Konfigurasi

| Parameter | Spesifikasi | Keterangan |
| :--- | :--- | :--- |
| **Model Arsitektur** | ICAM Varian C | *Incongruity-Aware Cross-Modal Attention* |
| **Enkoder Tekstual** | IndoBERTweet | Maksimum 128 token, lowercasing, NFKC, slang mapping |
| **Enkoder Visual** | ViT-B/16 | 224×224 px input, resolusi 196 patch visual |
| **Ambang Batas (*Threshold*)** | **0.45** | $\ge 0.45 \to \text{Hate Speech}$, $< 0.45 \to \text{Not Hate}$ |
| **Langkah IG** | 50 Langkah Riemann | Baseline `[PAD]` token, target argmax logit |
| **Batas Unggah** | 10 MB / 20 MP | Validasi format `RGB` (JPEG, PNG, WebP) |
| **Variabel Lingkungan** | `INDOMEME_DEVICE` | Pilihan: `auto`, `cuda`, `cpu` |
| **Verifikasi Protokol** | SHA-256 YAML | Menolak eksekusi jika spesifikasi model termodifikasi |

---

## 🔒 Privasi & Keamanan Data

- **Pemrosesan Lokal Murni**: Seluruh komputasi berjalan di *local host* tanpa mengirimkan gambar atau teks ke layanan API cloud pihak ketiga.
- **In-Memory Buffer**: Citra yang diunggah oleh pengguna tidak disimpan secara permanen di media disk dan segera dibersihkan saat sesi berakhir.
- **Strict Weights Loading**: Checkpoint dimuat secara ketat (`weights_only=True`) untuk mencegah eksekusi *arbitrary code* berbahaya.

---

## 📄 Lisensi

Repositori dan perangkat lunak ini dirilis di bawah lisensi [MIT License](LICENSE).
Semua hak kekayaan intelektual atas arsitektur model dan dataset acuan mengikuti lisensi publik masing-masing penyedia sumber.
