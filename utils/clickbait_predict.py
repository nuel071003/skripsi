"""
=============================================================
  clickbait_predict.py  –  Utils Deteksi Clickbait
  Tugas : Memuat model BART Clickbait dan memprediksi apakah
          sebuah judul berita termasuk clickbait atau bukan.
=============================================================

Cara kerja model:
  - Arsitektur : BartForSequenceClassification (2 kelas)
  - LABEL_0    : Non-Clickbait
  - LABEL_1    : Clickbait
  - Input      : Judul berita (string)
  - Output     : label (str) dan confidence (float 0-100)
"""

import os                          # Untuk membaca path folder model
import torch                       # Framework deep learning (PyTorch)
import torch.nn.functional as F    # Untuk fungsi softmax (mengubah logit → probabilitas)

from transformers import (
    AutoTokenizer,                       # Memuat tokenizer secara otomatis
    AutoModelForSequenceClassification   # Memuat model klasifikasi secara otomatis
)


# ---------------------------------------------------------------
# Konfigurasi Path Model
# ---------------------------------------------------------------
# __file__ adalah path dari file ini sendiri (clickbait_predict.py)
# os.path.dirname naik ke folder utils/
# os.path.join mengarah ke models/model_clickbait/
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "model_clickbait")

# ---------------------------------------------------------------
# Panjang maksimum token yang dikirim ke model
# BART memiliki batas 1024 token, 64 cukup untuk judul berita
# ---------------------------------------------------------------
MAX_LENGTH = 64


# ---------------------------------------------------------------
# Variabel Global untuk menyimpan model dan tokenizer
# Agar tidak perlu dimuat ulang setiap kali prediksi dipanggil
# ---------------------------------------------------------------
_tokenizer = None   # Tokenizer untuk mengubah teks → angka
_model     = None   # Model BART yang sudah dilatih


# ---------------------------------------------------------------
# Fungsi: muat_model_clickbait
# ---------------------------------------------------------------
def muat_model_clickbait():
    """
    Memuat tokenizer dan model clickbait dari folder lokal.
    Fungsi ini hanya perlu dipanggil SATU KALI saat aplikasi mulai.

    Model disimpan di: models/model_clickbait/
    """
    global _tokenizer, _model

    # Jika sudah dimuat sebelumnya, tidak perlu dimuat lagi
    if _tokenizer is not None and _model is not None:
        return

    # Pastikan folder model ada
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Folder model clickbait tidak ditemukan: {MODEL_PATH}\n"
            f"Pastikan folder 'models/model_clickbait/' ada dan berisi file model."
        )

    print(f"[INFO] Memuat model clickbait dari: {MODEL_PATH}")

    # Muat tokenizer: mengubah teks menjadi token angka yang dimengerti model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Muat model klasifikasi BART
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Set model ke mode evaluasi (bukan training)
    # Ini penting agar dropout dinonaktifkan dan hasil konsisten
    _model.eval()

    print("[INFO] Model clickbait berhasil dimuat.")


# ---------------------------------------------------------------
# Fungsi Utama: prediksi_clickbait
# ---------------------------------------------------------------
def prediksi_clickbait(judul: str) -> dict:
    """
    Memprediksi apakah judul berita termasuk clickbait atau tidak.

    Parameter:
        judul (str) : Judul berita yang akan dianalisis.

    Mengembalikan (dict) dengan kunci:
        - label      (str)   : "Clickbait" atau "Non-Clickbait"
        - confidence (float) : Persentase keyakinan model (0-100)
        - label_class(str)   : Warna Bootstrap untuk tampilan (danger/success)
    """

    # Pastikan model sudah dimuat
    if _tokenizer is None or _model is None:
        raise RuntimeError(
            "[ERROR] Model clickbait belum dimuat. "
            "Panggil muat_model_clickbait() terlebih dahulu."
        )

    # Validasi input
    judul = judul.strip()
    if not judul:
        return {
            "label"      : "Non-Clickbait",
            "confidence" : 0.0,
            "label_class": "secondary"
        }

    # ---------------------------------------------------------------
    # Langkah 1: Tokenisasi – ubah teks menjadi tensor angka
    # ---------------------------------------------------------------
    # return_tensors="pt" → format PyTorch tensor
    # truncation=True     → potong jika lebih panjang dari MAX_LENGTH
    # padding=True        → tambah padding agar panjang seragam
    inputs = _tokenizer(
        judul,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # ---------------------------------------------------------------
    # Langkah 2: Inferensi – jalankan model tanpa menghitung gradien
    # torch.no_grad() menghemat memori karena kita tidak melatih model
    # ---------------------------------------------------------------
    with torch.no_grad():
        outputs = _model(**inputs)

    # outputs.logits adalah "skor mentah" dari model (bisa negatif/positif)
    # Contoh: tensor([[-1.2, 2.5]]) → kelas 0 skornya -1.2, kelas 1 skornya 2.5
    logits = outputs.logits

    # ---------------------------------------------------------------
    # Langkah 3: Softmax – ubah logit menjadi probabilitas (0 hingga 1)
    # Setelah softmax: tensor([[0.05, 0.95]]) → 5% Non-Clickbait, 95% Clickbait
    # ---------------------------------------------------------------
    probs = F.softmax(logits, dim=-1)

    # Ambil indeks kelas dengan probabilitas tertinggi
    # Contoh: jika probs = [0.05, 0.95] → predicted_class = 1
    predicted_class = torch.argmax(probs, dim=-1).item()

    # Ambil nilai confidence (probabilitas) untuk kelas yang diprediksi
    # Kalikan 100 untuk dijadikan persentase
    confidence = probs[0][predicted_class].item() * 100

    # ---------------------------------------------------------------
    # Langkah 4: Terjemahkan hasil ke label yang mudah dipahami
    # ---------------------------------------------------------------
    # LABEL_0 = Non-Clickbait, LABEL_1 = Clickbait
    # (sesuai config.json model yang telah dilatih)
    if predicted_class == 1:
        label       = "Clickbait"
        label_class = "danger"    # Warna merah di Bootstrap
    else:
        label       = "Non-Clickbait"
        label_class = "success"   # Warna hijau di Bootstrap

    return {
        "label"      : label,
        "confidence" : round(confidence, 2),   # Bulatkan 2 desimal
        "label_class": label_class,
        "clickbait_prob": probs[0][1].item() * 100
    }

import re

def cari_bagian_clickbait(judul: str) -> str:
    """
    Mencari bagian/potongan judul yang paling berindikasi clickbait.
    Membelah judul berdasarkan tanda baca tertentu, lalu menyoroti (highlight)
    potongan yang paling memicu terdeteksinya clickbait.
    """
    # Pisahkan berdasarkan tanda baca pemisah umum pada judul
    bagian_list = re.split(r'([,\|:;—-])', judul)
    
    max_prob = -1.0
    idx_terindikasi = -1
    
    for i in range(0, len(bagian_list), 2):
        teks = bagian_list[i].strip()
        if len(teks) < 5:  # Abaikan jika potongannya terlalu pendek
            continue
            
        hasil = prediksi_clickbait(teks)
        prob = hasil.get("clickbait_prob", 0.0)
        
        if prob > max_prob:
            max_prob = prob
            idx_terindikasi = i
            
    if idx_terindikasi == -1:
        return judul
        
    hasil_konteks = []
    for i, teks in enumerate(bagian_list):
        if i == idx_terindikasi:
            # Highlight bagian ini dengan style gelap & teks kuning (warning) agar kontras di atas background merah
            hasil_konteks.append(f'<span class="bg-dark text-warning fw-bolder px-2 py-1 rounded shadow-sm border border-warning border-opacity-50">{teks}</span>')
        else:
            hasil_konteks.append(teks)
            
    return "".join(hasil_konteks)

