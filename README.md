# 📰 DeteksiBerita

Implementasi Natural Language Processing Menggunakan Model BART untuk Deteksi Clickbait dan Hoaks pada Berita Berbahasa Indonesia.

## Fitur

- Deteksi Clickbait
- Deteksi Hoaks
- Web Scraping Berita
- Highlight bagian Clickbait
- Highlight kalimat Hoaks
- Flask Web Application
- Model otomatis diunduh dari Hugging Face

## Teknologi

- Python
- Flask
- Transformers
- PyTorch
- BeautifulSoup
- Hugging Face

## Model

Clickbait

nuel-saja07/model-clickbait

Hoaks

nuel-saja07/model-hoaks

## Menjalankan Lokal

```bash
pip install -r requirements.txt

python app.py
```

Website:

```
http://localhost:5000
```

---

## Docker

Docker akan otomatis menjalankan

```
python app.py
```

dan menggunakan

```
PORT=7860
```

sesuai standar Hugging Face Spaces.
