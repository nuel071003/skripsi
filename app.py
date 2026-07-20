"""
=============================================================
  app.py  –  File Utama Aplikasi Flask
  Judul Skripsi:
    "Implementasi Natural Language Processing Menggunakan
     Model BART untuk Deteksi Clickbait dan Hoaks pada
     Berita Berbahasa Indonesia"
=============================================================

Tanggung jawab file ini:
  1. Inisialisasi aplikasi Flask
  2. Memuat model BART saat server mulai (sekali saja)
  3. Mendefinisikan routing (halaman-halaman website)
  4. Menerima URL dari pengguna
  5. Memanggil scraping.py untuk mengambil konten berita
  6. Memanggil clickbait_predict.py untuk analisis judul
  7. Memanggil hoax_predict.py untuk analisis isi berita
  8. Mengirimkan hasil ke template hasil.html

Alur:
User Input URL → Scraping → Analisis Clickbait → Analisis Hoaks → Tampil Hasil
"""

import os
import sys

# ---------------------------------------------------------------
# Paksa stdout ke UTF-8 agar karakter Indonesia tidak error
# ---------------------------------------------------------------
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------
# Import fungsi dari folder utils/
# Setiap file di utils/ punya satu tanggung jawab yang jelas
# ---------------------------------------------------------------
from utils.scraping         import ambil_berita
from utils.clickbait_predict import muat_model_clickbait, prediksi_clickbait, cari_bagian_clickbait
from utils.hoax_predict      import muat_model_hoax,      prediksi_hoax, cari_kalimat_hoax


# ---------------------------------------------------------------
# Inisialisasi Aplikasi Flask
# ---------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "deteksi-berita-secret-2026")


# ---------------------------------------------------------------
# Konfigurasi Path Model
# ---------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_CLICKBAIT = os.path.join(BASE_DIR, "models", "model_clickbait")
MODEL_HOAX      = os.path.join(BASE_DIR, "models", "model_hoax")

# Flag: apakah model berhasil dimuat?
model_loaded = False


# ---------------------------------------------------------------
# Fungsi: muat_semua_model
# Dipanggil SATU KALI saat server mulai agar tidak lambat
# setiap kali ada request dari pengguna.
# ---------------------------------------------------------------
def muat_semua_model():
    """
    Memuat model clickbait dan hoaks ke dalam memori.
    Proses ini membutuhkan waktu beberapa menit pertama kali,
    tapi setelah itu setiap prediksi akan berjalan cepat.
    """
    global model_loaded

    # Cek apakah folder model ada
    cb_ada = os.path.exists(MODEL_CLICKBAIT)
    hx_ada = os.path.exists(MODEL_HOAX)

    if not cb_ada or not hx_ada:
        print("\n[WARN] ====================================")
        if not cb_ada:
            print(f"  Model clickbait tidak ditemukan: {MODEL_CLICKBAIT}")
        if not hx_ada:
            print(f"  Model hoaks tidak ditemukan: {MODEL_HOAX}")
        print("  Sistem tidak dapat menjalankan prediksi.")
        print("[WARN] ====================================\n")
        return

    try:
        # Muat kedua model ke memori
        muat_model_clickbait()
        muat_model_hoax()
        model_loaded = True
        print("\n[INFO] ====================================")
        print("  Semua model BART berhasil dimuat!")
        print("  Sistem siap menerima request.")
        print("[INFO] ====================================\n")
    except Exception as e:
        print(f"\n[ERROR] Gagal memuat model: {e}\n")


# ---------------------------------------------------------------
# Fungsi: tentukan_kesimpulan
# Menentukan kesimpulan akhir berdasarkan hasil kedua model
# ---------------------------------------------------------------
def tentukan_kesimpulan(is_clickbait: bool, is_hoax: bool) -> dict:
    """
    Menentukan kesimpulan akhir berdasarkan 4 kemungkinan kombinasi hasil:
      1. Clickbait DAN Hoaks     → paling berbahaya
      2. Clickbait, Non-Hoaks    → judul menyesatkan, isi masih oke
      3. Non-Clickbait, Hoaks    → judul oke, tapi isi menyesatkan
      4. Non-Clickbait, Non-Hoaks→ berita valid

    Parameter:
      is_clickbait (bool) : True jika terdeteksi clickbait
      is_hoax      (bool) : True jika terdeteksi hoaks

    Mengembalikan dict dengan:
      - teks        : teks kesimpulan singkat
      - detail      : penjelasan lengkap
      - kelas       : warna Bootstrap (danger/warning/info/success)
      - ikon        : emoji/ikon untuk tampilan
    """
    if is_clickbait and is_hoax:
        return {
            "teks"  : "Clickbait dan Hoaks",
            "detail": (
                "Berita ini terdeteksi memiliki judul clickbait "
                "sekaligus mengandung informasi yang tidak akurat (hoaks). "
                "Sangat disarankan untuk tidak mempercayai dan menyebarkan berita ini."
            ),
            "kelas" : "danger",
            "ikon"  : "🚨"
        }
    elif is_clickbait and not is_hoax:
        return {
            "teks"  : "Clickbait tetapi Non-Hoaks",
            "detail": (
                "Judul berita terdeteksi menggunakan gaya clickbait yang berlebihan, "
                "namun isi berita tidak mengandung informasi hoaks. "
                "Konten berita relatif dapat dipercaya, meskipun judulnya mungkin berlebihan."
            ),
            "kelas" : "warning",
            "ikon"  : "⚠️"
        }
    elif not is_clickbait and is_hoax:
        return {
            "teks"  : "Non-Clickbait tetapi Hoaks",
            "detail": (
                "Judul berita tidak terdeteksi sebagai clickbait, "
                "namun isi berita mengandung informasi yang tidak akurat (hoaks). "
                "Harap berhati-hati dan verifikasi informasi ke sumber terpercaya."
            ),
            "kelas" : "info",
            "ikon"  : "⚡"
        }
    else:
        return {
            "teks"  : "Non-Clickbait dan Non-Hoaks",
            "detail": (
                "Berita ini terdeteksi sebagai berita yang VALID. "
                "Judul tidak mengandung clickbait dan isi berita tidak mengandung hoaks. "
                "Berita ini relatif aman untuk dibaca dan dibagikan."
            ),
            "kelas" : "success",
            "ikon"  : "✅"
        }


# ===============================================================
# ROUTING – Definisi halaman-halaman website
# ===============================================================

@app.route("/")
def index():
    """Halaman utama – form input URL berita."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Halaman Tentang Sistem – penjelasan skripsi dan model."""
    return render_template("about.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    Halaman utama proses deteksi.

    Alur:
    1. Terima URL dari form HTML (method POST)
    2. Validasi URL tidak kosong
    3. Cek apakah model sudah dimuat
    4. Panggil scraping.py → ambil judul & isi berita
    5. Panggil clickbait_predict.py → analisis judul
    6. Panggil hoax_predict.py → analisis isi berita
    7. Tentukan kesimpulan akhir
    8. Kirim semua data ke template hasil.html
    """

    # -----------------------------------------------------------
    # Langkah 1: Ambil URL dari form (input pengguna)
    # -----------------------------------------------------------
    url = request.form.get("url", "").strip()

    # Validasi: URL tidak boleh kosong
    if not url:
        return render_template(
            "index.html",
            error="Silakan masukkan URL berita terlebih dahulu."
        )

    # -----------------------------------------------------------
    # Langkah 2: Cek apakah model sudah siap digunakan
    # -----------------------------------------------------------
    if not model_loaded:
        return render_template(
            "index.html",
            error=(
                "Model belum berhasil dimuat. "
                "Pastikan folder models/model_clickbait dan models/model_hoax tersedia, "
                "lalu restart server Flask."
            )
        )

    # -----------------------------------------------------------
    # Langkah 3: Web Scraping – ambil judul dan isi berita
    # -----------------------------------------------------------
    try:
        hasil_scraping = ambil_berita(url)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Terjadi kesalahan saat melakukan scraping: {str(e)}"
        )

    # Jika scraping gagal, tampilkan pesan error ke pengguna
    if not hasil_scraping["sukses"]:
        return render_template(
            "index.html",
            error=hasil_scraping["error"]
        )

    judul  = hasil_scraping["judul"]
    konten = hasil_scraping["konten"]

    # -----------------------------------------------------------
    # Langkah 4: Prediksi Clickbait – analisis judul berita
    # -----------------------------------------------------------
    try:
        hasil_clickbait = prediksi_clickbait(judul)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Terjadi kesalahan saat menjalankan model clickbait: {str(e)}"
        )

    # -----------------------------------------------------------
    # Langkah 5: Prediksi Hoaks – analisis isi berita
    # -----------------------------------------------------------
    try:
        hasil_hoax = prediksi_hoax(konten)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Terjadi kesalahan saat menjalankan model hoaks: {str(e)}"
        )

    # -----------------------------------------------------------
    # Langkah 6: Tentukan kesimpulan akhir (4 kemungkinan)
    # -----------------------------------------------------------
    is_clickbait = (hasil_clickbait["label"] == "Clickbait")
    is_hoax      = (hasil_hoax["label"]      == "Hoaks")

    kesimpulan = tentukan_kesimpulan(is_clickbait, is_hoax)

    # -----------------------------------------------------------
    # Langkah 7: Susun data hasil untuk dikirim ke template HTML
    # -----------------------------------------------------------
    hasil = {
        # Data artikel hasil scraping
        "url"          : url,
        "judul"        : judul,
        "isi_teks"     : konten,           # Teks lengkap untuk ditampilkan di collapsible
        "isi_preview"  : konten[:300] + "..." if len(konten) > 300 else konten,

        # Hasil analisis clickbait
        "clickbait": {
            "status"        : hasil_clickbait["label"],
            "confidence"    : hasil_clickbait["confidence"],
            "label_class"   : hasil_clickbait["label_class"],
            "judul_analyzed": cari_bagian_clickbait(judul) if is_clickbait else judul,
        },

        # Hasil analisis hoaks
        "hoax": {
            "status"       : hasil_hoax["label"],
            "confidence"   : hasil_hoax["confidence"],
            "label_class"  : hasil_hoax["label_class"],
            "isi_analyzed" : cari_kalimat_hoax(konten) if is_hoax else (konten[:200] + "..." if len(konten) > 200 else konten),
        },

        # Kesimpulan akhir
        "kesimpulan"       : kesimpulan["teks"],
        "kesimpulan_detail": kesimpulan["detail"],
        "kesimpulan_class" : kesimpulan["kelas"],
        "kesimpulan_ikon"  : kesimpulan["ikon"],

        # Flag: ini bukan placeholder, ini data nyata
        "is_placeholder"   : False,
    }

    return render_template("hasil.html", hasil=hasil)


# ---------------------------------------------------------------
# API Endpoint – untuk keperluan debugging / pengecekan status
# ---------------------------------------------------------------
@app.route("/api/status")
def api_status():
    """Mengembalikan status aplikasi dalam format JSON."""
    return jsonify({
        "aplikasi"    : "DeteksiBerita",
        "versi"       : "1.0.0",
        "model_loaded": model_loaded,
        "models": {
            "clickbait": os.path.exists(MODEL_CLICKBAIT),
            "hoax"     : os.path.exists(MODEL_HOAX),
        },
    })


@app.route("/health")
def health():
    """Health-check endpoint untuk memastikan server berjalan."""
    return jsonify({"status": "healthy"}), 200


# ===============================================================
# Entry Point – Titik masuk program
# ===============================================================
if __name__ == "__main__":
    # Muat model sebelum server mulai menerima request
    muat_semua_model()

    print("=" * 55)
    print("  DeteksiBerita - Server Berjalan")
    print("  Buka browser dan akses: http://localhost:5000")
    print("=" * 55)

    # debug=True → otomatis reload saat kode berubah (mode pengembangan)
    app.run(debug=True, host="0.0.0.0", port=5000)
