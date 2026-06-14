"""
Motion Control AI - Telegram Bot
Analisis pose manusia dari foto menggunakan MediaPipe + Claude AI
"""

import os
import logging
import tempfile
import base64
from pathlib import Path

import anthropic
import cv2
import mediapipe as mp
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Konfigurasi ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY_HERE")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Helpers ───────────────────────────────────────────────────────────────────

def analyze_pose_landmarks(landmarks) -> dict:
    """Ekstrak data landmark pose dan hitung sudut sendi utama."""
    if not landmarks:
        return {}

    lm = landmarks.landmark

    def get_coords(idx):
        p = lm[idx]
        return np.array([p.x, p.y, p.z])

    def angle_between(a, b, c):
        """Sudut di titik b (derajat)."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return round(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))), 1)

    # Landmark indices (MediaPipe Pose)
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW,    R_ELBOW    = 13, 14
    L_WRIST,    R_WRIST    = 15, 16
    L_HIP,      R_HIP      = 23, 24
    L_KNEE,     R_KNEE     = 25, 26
    L_ANKLE,    R_ANKLE    = 27, 28
    NOSE                   = 0

    try:
        data = {
            "sudut_siku_kiri":   angle_between(get_coords(L_SHOULDER),  get_coords(L_ELBOW),  get_coords(L_WRIST)),
            "sudut_siku_kanan":  angle_between(get_coords(R_SHOULDER),  get_coords(R_ELBOW),  get_coords(R_WRIST)),
            "sudut_bahu_kiri":   angle_between(get_coords(L_HIP),       get_coords(L_SHOULDER), get_coords(L_ELBOW)),
            "sudut_bahu_kanan":  angle_between(get_coords(R_HIP),       get_coords(R_SHOULDER), get_coords(R_ELBOW)),
            "sudut_lutut_kiri":  angle_between(get_coords(L_HIP),       get_coords(L_KNEE),   get_coords(L_ANKLE)),
            "sudut_lutut_kanan": angle_between(get_coords(R_HIP),       get_coords(R_KNEE),   get_coords(R_ANKLE)),
            "sudut_pinggul_kiri":  angle_between(get_coords(L_SHOULDER), get_coords(L_HIP),   get_coords(L_KNEE)),
            "sudut_pinggul_kanan": angle_between(get_coords(R_SHOULDER), get_coords(R_HIP),   get_coords(R_KNEE)),
        }

        # Estimasi postur vertikal kepala-pinggul
        nose_y    = lm[NOSE].y
        hip_mid_y = (lm[L_HIP].y + lm[R_HIP].y) / 2
        data["postur_vertikal_persen"] = round((1 - abs(nose_y - hip_mid_y)) * 100, 1)

        # Simetri bahu kiri-kanan
        data["simetri_bahu_persen"] = round(
            (1 - abs(lm[L_SHOULDER].y - lm[R_SHOULDER].y)) * 100, 1
        )

        return data
    except Exception as e:
        logger.warning(f"Error saat ekstrak landmark: {e}")
        return {}


def draw_pose_on_image(image_path: str, output_path: str) -> bool:
    """Gambar skeleton pose pada gambar dan simpan hasilnya."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return False

        annotated = img.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        cv2.imwrite(output_path, annotated)
        return True


def process_image_for_pose(image_path: str) -> tuple[dict, str | None]:
    """
    Proses gambar: deteksi pose & gambar skeleton.
    Return (landmark_data, output_image_path).
    """
    output_path = image_path.replace(".jpg", "_pose.jpg").replace(".png", "_pose.png")
    if output_path == image_path:
        output_path = image_path + "_pose.jpg"

    img = cv2.imread(image_path)
    if img is None:
        return {}, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(img_rgb)

        if not results.pose_landmarks:
            return {}, None

        # Gambar skeleton
        annotated = img.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        cv2.imwrite(output_path, annotated)

        landmark_data = analyze_pose_landmarks(results.pose_landmarks)
        return landmark_data, output_path


def ask_claude_analysis(image_path: str, landmark_data: dict, mode: str = "general") -> str:
    """Kirim gambar + data landmark ke Claude untuk analisis mendalam."""

    # Encode gambar ke base64
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    # Susun data landmark sebagai teks
    landmark_text = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}°" if "sudut" in k
        else f"- {k.replace('_', ' ').title()}: {v}%"
        for k, v in landmark_data.items()
    ) if landmark_data else "Data landmark tidak tersedia."

    # Prompt berdasarkan mode
    mode_prompts = {
        "general": "Analisis pose secara umum: postur tubuh, keseimbangan, dan kesan keseluruhan.",
        "olahraga": "Analisis dari sudut pandang atletik: teknik, efisiensi gerakan, dan potensi cedera.",
        "ergonomi": "Analisis ergonomi kerja: risiko MSDs, rekomendasi perbaikan postur.",
        "kesehatan": "Analisis kesehatan postur: kelainan postur, kelengkungan tulang belakang, saran perbaikan.",
    }
    focus = mode_prompts.get(mode, mode_prompts["general"])

    prompt = f"""Kamu adalah ahli analisis biomekanik dan pose manusia. 
Analisis gambar pose manusia berikut beserta data landmark yang telah diekstrak secara otomatis.

DATA SUDUT SENDI (dari MediaPipe):
{landmark_text}

FOKUS ANALISIS: {focus}

Berikan analisis dalam format berikut (gunakan emoji, buat mudah dibaca):

🏃 **RINGKASAN POSE**
[Deskripsi singkat pose yang terdeteksi]

📐 **ANALISIS SENDI & SUDUT**
[Interpretasi sudut-sudut sendi yang signifikan]

⚠️ **POTENSI MASALAH**
[Risiko atau ketidakseimbangan yang terdeteksi]

✅ **REKOMENDASI**
[Saran perbaikan atau latihan yang relevan]

📊 **SKOR POSTUR**
[Berikan skor 0-100 dan penjelasannya]

Gunakan Bahasa Indonesia yang jelas dan mudah dipahami."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return response.content[0].text


# ── Telegram Handlers ─────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /start"""
    text = (
        "🤖 *Motion Control AI Bot*\n\n"
        "Selamat datang! Bot ini menganalisis *pose manusia* dari foto menggunakan "
        "MediaPipe + Claude AI.\n\n"
        "📸 *Cara pakai:*\n"
        "1\\. Kirim foto ke bot ini\n"
        "2\\. Pilih jenis analisis\n"
        "3\\. Dapatkan laporan detail pose\n\n"
        "⚡ *Perintah tersedia:*\n"
        "/start \\- Tampilkan pesan ini\n"
        "/help \\- Panduan lengkap\n"
        "/demo \\- Info mode analisis\n"
    )
    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /help"""
    text = (
        "📖 *Panduan Motion Control AI Bot*\n\n"
        "*Mode Analisis:*\n"
        "🔍 General \\- Analisis pose umum\n"
        "🏋️ Olahraga \\- Teknik & performa atletik\n"
        "💺 Ergonomi \\- Postur kerja & MSDs\n"
        "🏥 Kesehatan \\- Kelainan postur & saran\n\n"
        "*Tips foto terbaik:*\n"
        "• Seluruh tubuh terlihat jelas\n"
        "• Pencahayaan cukup\n"
        "• Hindari pakaian longgar berlebihan\n"
        "• Latar belakang kontras dengan tubuh\n\n"
        "Cukup kirim foto dan pilih mode\\!"
    )
    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def demo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /demo"""
    text = (
        "🎯 *Mode Analisis yang Tersedia*\n\n"
        "Setelah kirim foto, pilih salah satu:\n\n"
        "🔍 *General* — Gambaran keseluruhan pose\n"
        "🏋️ *Olahraga* — Analisis teknik & risiko cedera\n"
        "💺 *Ergonomi* — Optimal untuk pekerja kantoran\n"
        "🏥 *Kesehatan* — Deteksi kelainan postur\n\n"
        "Kirim foto sekarang untuk mencoba\\!"
    )
    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler saat user kirim foto."""
    await update.message.reply_text("📸 Foto diterima! Pilih mode analisis:")

    keyboard = [
        [
            InlineKeyboardButton("🔍 General",   callback_data="mode_general"),
            InlineKeyboardButton("🏋️ Olahraga",  callback_data="mode_olahraga"),
        ],
        [
            InlineKeyboardButton("💺 Ergonomi",  callback_data="mode_ergonomi"),
            InlineKeyboardButton("🏥 Kesehatan", callback_data="mode_kesehatan"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Simpan file_id foto ke context user
    photo = update.message.photo[-1]  # ambil resolusi tertinggi
    context.user_data["pending_photo_id"] = photo.file_id

    await update.message.reply_text(
        "Pilih jenis analisis yang kamu inginkan:",
        reply_markup=reply_markup,
    )


async def mode_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler saat user memilih mode analisis."""
    query = update.callback_query
    await query.answer()

    mode = query.data.replace("mode_", "")
    mode_labels = {
        "general":   "🔍 General",
        "olahraga":  "🏋️ Olahraga",
        "ergonomi":  "💺 Ergonomi",
        "kesehatan": "🏥 Kesehatan",
    }
    label = mode_labels.get(mode, mode)

    await query.edit_message_text(f"⏳ Menganalisis pose dengan mode *{label}*...\nMohon tunggu sebentar.", parse_mode="Markdown")

    file_id = context.user_data.get("pending_photo_id")
    if not file_id:
        await query.edit_message_text("❌ Foto tidak ditemukan. Silakan kirim ulang foto.")
        return

    try:
        # Download foto
        file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        # Proses pose detection
        await query.edit_message_text("🦴 Mendeteksi landmark pose...")
        landmark_data, pose_image_path = process_image_for_pose(tmp_path)

        if not landmark_data:
            await query.edit_message_text(
                "⚠️ *Pose tidak terdeteksi.*\n\n"
                "Tips:\n"
                "• Pastikan seluruh tubuh terlihat\n"
                "• Gunakan foto dengan pencahayaan baik\n"
                "• Hindari pose terlalu jauh dari kamera",
                parse_mode="Markdown",
            )
            return

        # Analisis Claude AI
        await query.edit_message_text("🤖 Claude AI sedang menganalisis...")
        analysis = ask_claude_analysis(tmp_path, landmark_data, mode)

        # Kirim gambar pose skeleton
        if pose_image_path and Path(pose_image_path).exists():
            with open(pose_image_path, "rb") as img_file:
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=img_file,
                    caption=f"🦴 Skeleton pose terdeteksi (mode: {label})",
                )

        # Kirim analisis
        # Telegram max 4096 chars per message
        if len(analysis) > 4000:
            for i in range(0, len(analysis), 4000):
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=analysis[i:i+4000],
                )
        else:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=analysis,
            )

        # Tombol analisis ulang
        keyboard = [[InlineKeyboardButton("🔄 Analisis Ulang dengan Mode Lain", callback_data="reanalyze")]]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ Analisis selesai! Kirim foto lain atau pilih mode berbeda.",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

        await query.edit_message_text(f"✅ Analisis mode *{label}* selesai!", parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error saat analisis: {e}", exc_info=True)
        await query.edit_message_text(
            f"❌ Terjadi kesalahan: `{str(e)[:200]}`\n\nCoba kirim foto ulang.",
            parse_mode="Markdown",
        )
    finally:
        # Cleanup temp files
        for path in [tmp_path, pose_image_path if 'pose_image_path' in locals() else None]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except Exception:
                    pass


async def reanalyze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler tombol analisis ulang."""
    query = update.callback_query
    await query.answer()

    if not context.user_data.get("pending_photo_id"):
        await query.edit_message_text("⚠️ Silakan kirim foto baru terlebih dahulu.")
        return

    keyboard = [
        [
            InlineKeyboardButton("🔍 General",   callback_data="mode_general"),
            InlineKeyboardButton("🏋️ Olahraga",  callback_data="mode_olahraga"),
        ],
        [
            InlineKeyboardButton("💺 Ergonomi",  callback_data="mode_ergonomi"),
            InlineKeyboardButton("🏥 Kesehatan", callback_data="mode_kesehatan"),
        ],
    ]
    await query.edit_message_text(
        "Pilih mode analisis:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ Perintah tidak dikenal. Gunakan /help untuk panduan.\n"
        "Atau langsung kirim *foto* untuk dianalisis! 📸",
        parse_mode="Markdown",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("⚠️  Set TELEGRAM_BOT_TOKEN di environment variable!")
        print("   export TELEGRAM_BOT_TOKEN='token_kamu'")
        print("   export ANTHROPIC_API_KEY='key_kamu'")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help",  help_handler))
    app.add_handler(CommandHandler("demo",  demo_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(CallbackQueryHandler(reanalyze_callback,   pattern="^reanalyze$"))
    app.add_handler(CallbackQueryHandler(mode_callback_handler, pattern="^mode_"))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_handler))

    print("🤖 Motion Control AI Bot berjalan...")
    print("   Tekan Ctrl+C untuk berhenti.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
