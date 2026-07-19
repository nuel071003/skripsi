import os
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_clickbait")

HF_MODEL = "nuel-saja07/model-clickbait"

MAX_LENGTH = 64

_tokenizer = None
_model = None


def muat_model_clickbait():
    """
    Memuat model clickbait.

    Prioritas:
    1. Folder lokal (untuk development)
    2. Hugging Face (untuk Render)
    """

    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    try:
        if os.path.exists(MODEL_PATH):
            print("[INFO] Menggunakan model lokal...")

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH
            )
        else:
            print("[INFO] Model lokal tidak ditemukan.")
            print("[INFO] Mengunduh model dari Hugging Face...")

            _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
            _model = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL
            )

        _model.eval()
        print("[INFO] Model Clickbait siap digunakan.")

    except Exception as e:
        raise RuntimeError(
            f"Gagal memuat model Clickbait: {e}"
        )
