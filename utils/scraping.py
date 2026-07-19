"""
=============================================================
  scraping.py  –  Utils Web Scraping
  Tugas : Mengambil judul dan isi berita dari sebuah URL
  Cara  : Menggunakan library requests (HTTP) dan
          BeautifulSoup (parsing HTML)
=============================================================
"""

import re                          # Untuk membersihkan teks dengan regex
import requests                    # Untuk mengambil halaman web (HTTP GET)
from bs4 import BeautifulSoup      # Untuk parsing / membaca HTML


# ---------------------------------------------------------------
# Daftar User-Agent agar request tidak diblokir seperti bot
# ---------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Batas waktu tunggu saat mengambil halaman (detik)
TIMEOUT = 15


# ---------------------------------------------------------------
# Fungsi Utama: ambil_berita
# ---------------------------------------------------------------
def ambil_berita(url: str) -> dict:
    """
    Mengambil judul dan isi berita dari URL yang diberikan.

    Parameter:
        url (str) : URL artikel berita yang ingin dianalisis.

    Mengembalikan (dict) dengan kunci:
        - sukses   (bool)  : True jika berhasil, False jika gagal.
        - judul    (str)   : Judul berita.
        - konten   (str)   : Isi berita (teks bersih).
        - error    (str)   : Pesan error jika gagal (kosong jika sukses).
    """

    # Validasi awal: pastikan URL dimulai dengan http atau https
    if not url.startswith(("http://", "https://")):
        return {
            "sukses": False,
            "judul": "",
            "konten": "",
            "error": "URL tidak valid. Pastikan URL diawali dengan http:// atau https://"
        }

    # ---------------------------------------------------------------
    # Langkah 1: Ambil HTML dari URL menggunakan requests
    # ---------------------------------------------------------------
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        # Jika status bukan 200 OK, berarti ada masalah
        if response.status_code != 200:
            return {
                "sukses": False,
                "judul": "",
                "konten": "",
                "error": f"Website merespons dengan kode error: {response.status_code}. "
                         f"Halaman mungkin tidak tersedia."
            }

        # Paksa encoding UTF-8 agar karakter Indonesia terbaca benar
        response.encoding = response.apparent_encoding

    except requests.exceptions.ConnectionError:
        return {
            "sukses": False,
            "judul": "",
            "konten": "",
            "error": "Tidak dapat terhubung ke website. Periksa koneksi internet atau URL Anda."
        }
    except requests.exceptions.Timeout:
        return {
            "sukses": False,
            "judul": "",
            "konten": "",
            "error": f"Website tidak merespons dalam {TIMEOUT} detik. Coba lagi nanti."
        }
    except requests.exceptions.RequestException as e:
        return {
            "sukses": False,
            "judul": "",
            "konten": "",
            "error": f"Terjadi kesalahan saat mengakses URL: {str(e)}"
        }

    # ---------------------------------------------------------------
    # Langkah 2: Parsing HTML menggunakan BeautifulSoup
    # ---------------------------------------------------------------
    soup = BeautifulSoup(response.text, "html.parser")

    # ---------------------------------------------------------------
    # Langkah 3: Ambil JUDUL berita
    # ---------------------------------------------------------------
    judul = _ambil_judul(soup)

    if not judul:
        return {
            "sukses": False,
            "judul": "",
            "konten": "",
            "error": "Judul berita tidak ditemukan. "
                     "Mungkin halaman ini bukan artikel berita atau strukturnya tidak umum."
        }

    # ---------------------------------------------------------------
    # Langkah 4: Ambil ISI berita
    # ---------------------------------------------------------------
    konten = _ambil_konten(soup)

    if not konten:
        return {
            "sukses": False,
            "judul": judul,
            "konten": "",
            "error": "Isi berita tidak ditemukan. Scraping tidak berhasil mengekstrak konten artikel."
        }

    # ---------------------------------------------------------------
    # Langkah 5: Kembalikan hasil jika berhasil
    # ---------------------------------------------------------------
    return {
        "sukses": True,
        "judul": judul,
        "konten": konten,
        "error": ""
    }


# ---------------------------------------------------------------
# Fungsi Pembantu: _ambil_judul
# ---------------------------------------------------------------
def _ambil_judul(soup: BeautifulSoup) -> str:
    """
    Mencoba mengambil judul dari berbagai sumber HTML:
    1. Tag <h1> (paling umum untuk judul artikel)
    2. Meta tag Open Graph (og:title)
    3. Tag <title> pada head halaman

    Mengembalikan string judul yang sudah dibersihkan.
    """

    # Coba ambil dari tag <h1> – paling banyak digunakan portal berita Indonesia
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return bersihkan_teks(h1.get_text())

    # Coba ambil dari meta og:title (format Open Graph)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content", "").strip():
        return bersihkan_teks(og_title["content"])

    # Coba ambil dari tag <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return bersihkan_teks(title_tag.get_text())

    return ""


# ---------------------------------------------------------------
# Fungsi Pembantu: _ambil_konten
# ---------------------------------------------------------------
def _ambil_konten(soup: BeautifulSoup) -> str:
    """
    Mencoba mengambil isi artikel dari berbagai elemen HTML yang umum
    digunakan oleh portal berita Indonesia.

    Strategi pencarian (berurutan dari yang paling spesifik):
    1. Tag <article>
    2. Div dengan class/id yang mengandung kata 'content', 'article', 'body', 'detail'
    3. Semua paragraf <p> pada halaman

    Mengembalikan string konten yang sudah dibersihkan.
    """

    teks_paragraf = []

    # ------ Strategi 1: Cari tag <article> ------
    article_tag = soup.find("article")
    if article_tag:
        paragraf = article_tag.find_all("p")
        teks_paragraf = [p.get_text() for p in paragraf if len(p.get_text(strip=True)) > 30]

    # ------ Strategi 2: Cari div berdasarkan class/id yang umum ------
    if not teks_paragraf:
        # Kata kunci yang sering dipakai portal berita Indonesia
        kandidat_class = [
            "article-content", "article-body", "article-text",
            "content-article", "post-content", "entry-content",
            "detail-content", "detail-desc", "read__content",    # Kompas, Detik
            "itp_bodycontent", "article__body",                   # Tempo
            "content", "body-content", "article-detail"
        ]
        for cls in kandidat_class:
            div = soup.find(["div", "section"], class_=cls)
            if div:
                paragraf = div.find_all("p")
                teks_paragraf = [p.get_text() for p in paragraf if len(p.get_text(strip=True)) > 30]
                if teks_paragraf:
                    break

    # ------ Strategi 3: Ambil semua <p> di halaman (fallback) ------
    if not teks_paragraf:
        semua_p = soup.find_all("p")
        teks_paragraf = [p.get_text() for p in semua_p if len(p.get_text(strip=True)) > 50]

    if not teks_paragraf:
        return ""

    # Gabungkan semua paragraf menjadi satu teks
    konten_gabung = " ".join(teks_paragraf)

    return bersihkan_teks(konten_gabung)


# ---------------------------------------------------------------
# Fungsi Pembantu: bersihkan_teks
# ---------------------------------------------------------------
def bersihkan_teks(teks: str) -> str:
    """
    Membersihkan teks dari karakter yang tidak diperlukan:
    - Menghapus spasi berlebih (tab, newline ganda, dll)
    - Menghapus karakter aneh / non-printable

    Parameter : teks (str) – teks mentah hasil scraping.
    Kembalian : teks (str) – teks yang sudah bersih dan rapi.
    """

    if not teks:
        return ""

    # Ganti newline, tab, dan spasi berulang menjadi satu spasi
    teks = re.sub(r"\s+", " ", teks)

    # Hapus spasi di awal dan akhir
    teks = teks.strip()

    return teks
