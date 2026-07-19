# DeteksiBerita 🔍

> **Implementasi Natural Language Processing Menggunakan Model BART untuk Deteksi Clickbait dan Hoaks pada Berita Berbahasa Indonesia**

Aplikasi web berbasis **Flask** yang mengintegrasikan model **BART (Bidirectional and Auto-Regressive Transformers)** untuk mendeteksi **clickbait** pada judul berita dan **hoaks** pada isi berita secara otomatis.

---

## 📁 Struktur Proyek

```
Berita/
│
├── app.py                    ← File utama Flask (routing & logika)
│
├── utils/
│   ├── __init__.py           ← Penanda package Python
│   ├── scraping.py           ← Web scraping (requests + BeautifulSoup)
│   ├── clickbait_predict.py  ← Prediksi clickbait (model BART)
│   └── hoax_predict.py       ← Prediksi hoaks (model BART)
│
├── models/
│   ├── model_clickbait/      ← Model BART clickbait (hasil trainer.save_model())
│   └── model_hoax/           ← Model BART hoaks (hasil trainer.save_model())
│
├── templates/
│   ├── index.html            ← Halaman utama (form input URL)
│   ├── hasil.html            ← Halaman hasil analisis
│   └── about.html            ← Halaman tentang sistem
│
├── static/
│   └── css/
│       └── style.css         ← CSS kustom
│
├── requirements.txt          ← Daftar library yang dibutuhkan
└── README.md                 ← Dokumentasi ini
```

---

## ⚙️ Cara Instalasi

### 1. Buat Virtual Environment (disarankan)
```bash
python -m venv env
env\Scripts\activate        # Windows
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Pastikan Model Tersedia
Letakkan model hasil training di:
- `models/model_clickbait/` → berisi `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`
- `models/model_hoax/` → berisi file yang sama

### 4. Jalankan Aplikasi
```bash
python app.py
```

Buka browser dan akses: **http://localhost:5000**

---

## 🔄 Alur Sistem

```
User memasukkan URL Berita
        ↓
 requests.get(url)          ← scraping.py
        ↓
 BeautifulSoup parsing      ← scraping.py
        ↓
 Judul Berita               Isi Berita
        ↓                        ↓
Model Clickbait BART    Model Hoaks BART
        ↓                        ↓
   Clickbait?               Hoaks?
        ↓                        ↓
         Kesimpulan Akhir (4 kemungkinan)
                ↓
        Tampilan hasil.html
```

---

## 📊 Output Analisis

| Informasi         | Keterangan                              |
|-------------------|-----------------------------------------|
| URL Berita        | URL yang dimasukkan pengguna            |
| Judul Berita      | Diambil dari tag `<h1>` atau meta OG    |
| Isi Berita        | Diambil dari `<article>` atau `<p>`     |
| Status Clickbait  | Clickbait / Non-Clickbait               |
| % Clickbait       | Confidence probabilitas model           |
| Status Hoaks      | Hoaks / Non-Hoaks                       |
| % Hoaks           | Confidence probabilitas model           |
| Kesimpulan Akhir  | Salah satu dari 4 kemungkinan di bawah  |

### 4 Kemungkinan Kesimpulan:
1. 🚨 **Clickbait dan Hoaks** – Sangat berbahaya
2. ⚠️ **Clickbait tetapi Non-Hoaks** – Judul menyesatkan, isi oke
3. ⚡ **Non-Clickbait tetapi Hoaks** – Judul oke, isi menyesatkan
4. ✅ **Non-Clickbait dan Non-Hoaks** – Berita valid

---

## 🛠️ Teknologi yang Digunakan

| Komponen      | Library/Tool                        |
|---------------|-------------------------------------|
| Backend       | Python, Flask                       |
| Model NLP     | Transformers (Hugging Face), PyTorch|
| Scraping      | Requests, BeautifulSoup4            |
| Frontend      | Bootstrap 5, Font Awesome, HTML/CSS |
| Model         | BART (BartForSequenceClassification)|

---

## 📝 Catatan Pengembangan

- Model dimuat **sekali** saat server mulai (`muat_semua_model()` di `app.py`)
- Scraping menggunakan multiple strategi untuk kompatibilitas berbagai portal berita Indonesia (Kompas, Detik, Tempo, CNN Indonesia, dll.)
- Konten berita dibatasi `MAX_LENGTH=512` token untuk model hoaks agar efisien
- Judul berita menggunakan `MAX_LENGTH=64` token untuk model clickbait

---

*Dibuat untuk keperluan Skripsi S1 – Fakultas Teknik Informatika*
