# 🤖 Motion Control AI — Telegram Bot

Bot Telegram untuk **analisis pose manusia** dari foto menggunakan:
- **MediaPipe** → deteksi 33 landmark tubuh & hitung sudut sendi
- **Claude AI** → interpretasi cerdas & rekomendasi

---

## 🗂️ Struktur File

```
motion_bot/
├── bot.py            ← Kode utama bot
├── requirements.txt  ← Dependensi Python
├── .env.example      ← Template variabel lingkungan
└── README.md         ← Panduan ini
```

---

## ⚡ Cara Setup

### 1. Clone / Salin File
Letakkan semua file dalam satu folder, misalnya `motion_bot/`.

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Dapatkan Token & API Key

**Telegram Bot Token:**
1. Buka Telegram, cari `@BotFather`
2. Kirim `/newbot`
3. Ikuti instruksi → salin token yang diberikan

**Anthropic API Key:**
1. Daftar/login di [console.anthropic.com](https://console.anthropic.com)
2. Buat API Key baru
3. Salin key tersebut

### 5. Set Variabel Lingkungan

**Cara A — Export langsung (terminal):**
```bash
export TELEGRAM_BOT_TOKEN="12345:ABCDefgh..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Cara B — File .env (pakai python-dotenv):**
```bash
cp .env.example .env
# Edit .env, isi TOKEN dan API_KEY
```
Lalu tambahkan di awal `bot.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```
Dan install: `pip install python-dotenv`

### 6. Jalankan Bot
```bash
python bot.py
```

---

## 🎮 Cara Pakai Bot

| Perintah | Fungsi |
|----------|--------|
| `/start` | Sambutan & info bot |
| `/help`  | Panduan penggunaan |
| `/demo`  | Info mode analisis |
| Kirim foto | Mulai analisis pose |

**Alur penggunaan:**
1. Kirim foto yang menampilkan seluruh tubuh
2. Pilih mode analisis:
   - 🔍 **General** — gambaran pose umum
   - 🏋️ **Olahraga** — teknik & risiko cedera
   - 💺 **Ergonomi** — postur kerja
   - 🏥 **Kesehatan** — kelainan postur
3. Bot mengirim balik:
   - Gambar dengan skeleton pose tergambar
   - Laporan analisis lengkap dari Claude AI

---

## 📐 Data yang Dianalisis

MediaPipe mengekstrak **33 landmark** tubuh dan bot menghitung:

| Metrik | Keterangan |
|--------|------------|
| Sudut siku kiri/kanan | Fleksi/ekstensi lengan |
| Sudut bahu kiri/kanan | Elevasi bahu |
| Sudut lutut kiri/kanan | Fleksi lutut |
| Sudut pinggul kiri/kanan | Fleksi pinggul |
| Postur vertikal (%) | Keselarasan vertikal tubuh |
| Simetri bahu (%) | Keseimbangan kiri-kanan |

---

## 💡 Tips Foto Terbaik

- ✅ Seluruh tubuh terlihat dari kepala hingga kaki
- ✅ Pencahayaan cukup, tidak terlalu gelap
- ✅ Latar belakang kontras dengan pakaian
- ✅ Pakaian pas (bukan terlalu longgar)
- ❌ Hindari foto terlalu jauh atau terpotong

---

## 🔧 Troubleshooting

**"Pose tidak terdeteksi"**
→ Coba foto lebih dekat, pencahayaan lebih terang, atau sudut berbeda.

**Error `TELEGRAM_BOT_TOKEN`**
→ Pastikan token sudah di-set dengan benar di environment.

**Error MediaPipe**
→ Pastikan OpenCV dan MediaPipe terinstall: `pip install mediapipe opencv-python`

---

## 🚀 Deployment (Opsional)

Untuk menjalankan 24/7, bisa deploy ke:
- **Railway** / **Render** (gratis tier tersedia)
- **VPS** dengan `screen` atau `tmux`
- **Docker** (buat Dockerfile dari `python:3.11-slim`)

---

## 📄 Lisensi
MIT — bebas digunakan dan dimodifikasi.
