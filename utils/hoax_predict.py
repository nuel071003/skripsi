import os                          # Untuk membaca path folder model
import torch                       # Framework deep learning (PyTorch)
import torch.nn.functional as F    # Untuk fungsi softmax

from transformers import (
    AutoTokenizer,                       # Memuat tokenizer secara otomatis
    AutoModelForSequenceClassification   # Memuat model klasifikasi secara otomatis
)


# ---------------------------------------------------------------
# Konfigurasi Path Model
# ---------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_hoax")

HF_MODEL = "nuel-saja07/model-hoaks"

# ---------------------------------------------------------------
# Panjang maksimum token untuk isi berita
# BART memiliki batas 1024; kita gunakan 512 agar efisien
# ---------------------------------------------------------------
MAX_LENGTH = 512


# ---------------------------------------------------------------
# Variabel Global
# ---------------------------------------------------------------
_tokenizer = None
_model     = None


# ---------------------------------------------------------------
# Fungsi: muat_model_hoax
# ---------------------------------------------------------------
def muat_model_hoax():
    """
    Memuat model Hoaks.

    Prioritas:
    1. Folder lokal (development)
    2. Hugging Face (Docker/Render)
    """

    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    try:

        # ==========================
        # Gunakan model lokal
        # ==========================
        if os.path.exists(MODEL_PATH):

            print(f"[INFO] Menggunakan model lokal: {MODEL_PATH}")

            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH
            )

            _model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH
            )

        # ==========================
        # Download dari Hugging Face
        # ==========================
        else:

            print("[INFO] Model lokal tidak ditemukan.")
            print(f"[INFO] Mengunduh model dari Hugging Face: {HF_MODEL}")

            _tokenizer = AutoTokenizer.from_pretrained(
                HF_MODEL
            )

            _model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL
            )

        _model.eval()

        print("[INFO] Model Hoaks siap digunakan.")

    except Exception as e:

        raise RuntimeError(
            f"Gagal memuat model Hoaks:\n{e}"
        )


# ---------------------------------------------------------------
# Fungsi Utama: prediksi_hoax
# ---------------------------------------------------------------
def prediksi_hoax(konten: str) -> dict:
    """
    Memprediksi apakah isi berita mengandung hoaks atau tidak.

    Parameter:
        konten (str) : Isi artikel berita yang akan dianalisis.

    Mengembalikan (dict) dengan kunci:
        - label      (str)   : "Hoaks" atau "Non-Hoaks"
        - confidence (float) : Persentase keyakinan model (0-100)
        - label_class(str)   : Warna Bootstrap untuk tampilan (danger/success)
    """

    # Pastikan model sudah dimuat
    if _tokenizer is None or _model is None:
        raise RuntimeError(
            "[ERROR] Model hoaks belum dimuat. "
            "Panggil muat_model_hoax() terlebih dahulu."
        )

    # Validasi input
    konten = konten.strip()
    if not konten:
        return {
            "label"      : "Non-Hoaks",
            "confidence" : 0.0,
            "label_class": "secondary"
        }

    # ---------------------------------------------------------------
    # Langkah 1: Tokenisasi
    # truncation=True → potong teks yang melebihi MAX_LENGTH
    # Ini penting karena isi berita bisa sangat panjang
    # ---------------------------------------------------------------
    inputs = _tokenizer(
        konten,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # ---------------------------------------------------------------
    # Langkah 2: Inferensi (jalankan model)
    # ---------------------------------------------------------------
    with torch.no_grad():
        outputs = _model(**inputs)

    logits = outputs.logits

    # ---------------------------------------------------------------
    # Langkah 3: Ubah logit → probabilitas dengan Softmax
    # ---------------------------------------------------------------
    probs = F.softmax(logits, dim=-1)

    # Kelas dengan probabilitas tertinggi
    predicted_class = torch.argmax(probs, dim=-1).item()

    # Confidence dalam persentase
    confidence = probs[0][predicted_class].item() * 100

    # ---------------------------------------------------------------
    # Langkah 4: Terjemahkan hasil
    # ---------------------------------------------------------------
    # LABEL_0 = Non-Hoaks, LABEL_1 = Hoaks
    if predicted_class == 1:
        label       = "Hoaks"
        label_class = "danger"   # Warna merah di Bootstrap
    else:
        label       = "Non-Hoaks"
        label_class = "success"  # Warna hijau di Bootstrap

    return {
        "label"      : label,
        "confidence" : round(confidence, 2),
        "label_class": label_class,
        "hoax_prob"  : probs[0][1].item() * 100  # Tambahkan probabilitas kelas Hoaks khusus
    }

import re

def cari_kalimat_hoax(konten: str) -> str:
    """
    Mencari kalimat dalam konten yang memiliki indikasi hoaks paling tinggi.
    Memecah konten menjadi kalimat-kalimat, memprediksi masing-masing kalimat,
    lalu mengembalikan kalimat tersebut beserta konteksnya (1 kalimat sebelum dan 1 sesudah),
    dengan kalimat utama di-highlight menggunakan tag HTML.
    """
    # Pisahkan berdasarkan tanda titik, tanda seru, atau tanda tanya yang diikuti spasi
    kalimat_list = re.split(r'(?<=[.!?]) +', konten)
    
    max_prob = -1.0
    idx_terindikasi = -1
    
    for i, kalimat in enumerate(kalimat_list):
        kalimat = kalimat.strip()
        if len(kalimat) < 15:  # Abaikan kalimat yang terlalu pendek
            continue
            
        hasil = prediksi_hoax(kalimat)
        prob_hoax = hasil.get("hoax_prob", 0.0)
        
        if prob_hoax > max_prob:
            max_prob = prob_hoax
            idx_terindikasi = i
            
    # Jika tidak ada kalimat yang valid
    if idx_terindikasi == -1:
        return konten[:200] + "..." if len(konten) > 200 else konten
        
    # Ambil konteks: 1 kalimat sebelum, kalimat itu sendiri, 1 kalimat sesudah
    start_idx = max(0, idx_terindikasi - 1)
    end_idx = min(len(kalimat_list), idx_terindikasi + 2)
    
    hasil_konteks = []
    for i in range(start_idx, end_idx):
        kal_bersih = kalimat_list[i].strip()
        if not kal_bersih:
            continue
            
        # Highlight kalimat yang terindikasi hoaks dengan style gelap & kuning agar kontras
        if i == idx_terindikasi:
            hasil_konteks.append(f'<span class="bg-dark text-warning fw-bolder px-2 py-1 rounded shadow-sm border border-warning border-opacity-50">{kal_bersih}</span>')
        else:
            hasil_konteks.append(kal_bersih)
            
    return " ".join(hasil_konteks)


