import os
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ---------------------------------------------------------------
# Konfigurasi Model
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_hoax")

# Repository Hugging Face
HF_MODEL = "nuel-saja07/model-hoax"

MAX_LENGTH = 512

# ---------------------------------------------------------------
# Variabel Global
# ---------------------------------------------------------------
_tokenizer = None
_model = None


# ---------------------------------------------------------------
# Fungsi: muat_model_hoax
# ---------------------------------------------------------------
def muat_model_hoax():
    """
    Memuat model Hoaks.

    Prioritas:
    1. Folder lokal (untuk development)
    2. Hugging Face (untuk Render)
    """

    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    # ==========================================================
    # Gunakan model lokal jika tersedia
    # ==========================================================
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Menggunakan model lokal: {MODEL_PATH}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

    # ==========================================================
    # Jika tidak ada, download dari Hugging Face
    # ==========================================================
    else:
        print("[INFO] Folder model lokal tidak ditemukan.")
        print(f"[INFO] Mengunduh model dari Hugging Face: {HF_MODEL}")

        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

        _model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL
        )

    _model.eval()

    print("[INFO] Model Hoaks berhasil dimuat.")
