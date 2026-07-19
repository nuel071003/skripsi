import os
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ==========================================================
# Konfigurasi Model
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_hoax")

# Repository Hugging Face
HF_MODEL = "nuel-saja07/model-hoaks"   # Sesuaikan dengan nama repository

MAX_LENGTH = 512

# ==========================================================
# Variabel Global
# ==========================================================
_tokenizer = None
_model = None


# ==========================================================
# Memuat Model
# ==========================================================
def muat_model_hoax():
    """
    Memuat model hoaks.

    Prioritas:
    1. Folder lokal (development)
    2. Hugging Face (Render)
    """

    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    try:

        # ===========================
        # Gunakan model lokal
        # ===========================
        if os.path.exists(MODEL_PATH):

            print("[INFO] Menggunakan model Hoaks lokal...")

            _tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH
            )

            _model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH
            )

        # ===========================
        # Gunakan Hugging Face
        # ===========================
        else:

            print("[INFO] Model lokal tidak ditemukan.")
            print(f"[INFO] Mengunduh model dari Hugging Face ({HF_MODEL})...")

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
            f"Gagal memuat model Hoaks: {e}"
        )
