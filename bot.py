"""
Motion Control AI - Telegram Bot
Analisis pose manusia dari foto menggunakan MediaPipe + Claude AI
Versi: tanpa OpenCV (pakai Pillow)
"""

import os
import logging
import tempfile
import base64
from pathlib import Path

import anthropic
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
anthropic_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Warna skeleton (RGB)
LANDMARK_COLOR   = (0, 255, 100)
CONNECTION_COLOR = (0, 180, 255)
DOT_RADIUS = 4
LINE_WIDTH = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def angle_between(a, b, c) -> float:
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return round(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))), 1)


def analyze_pose_landmarks(landmarks, w: int, h: int) -> dict:
    lm = landmarks.landmark

    def pt(idx):
        return np.array([lm[idx].x, lm[idx].y])

    try:
        data = {
            "sudut_siku_kiri":      angle_between(pt(11), pt(13), pt(15)),
            "sudut_siku_kanan":     angle_between(pt(12), pt(14), pt(16)),
            "sudut_bahu_kiri":      angle_between(pt(23), pt(11), pt(13)),
            "sudut_bahu_kanan":     angle_between(pt(24), pt(12), pt(14)),
            "sudut_lutut_kiri":     angle_between(pt(23), pt(25), pt(27)),
            "sudut_lutut_kanan":    angle_between(pt(24), pt(26), pt(28)),
            "sudut_pinggul_kiri":   angle_between(pt(11), pt(23), pt(25)),
            "sudut_pinggul_kanan":  angle_between(pt(12), pt(24), pt(26)),
        }
        nose_y    = lm[0].y
        hip_mid_y = (lm[23].y + lm[24].y) / 2
        data["postur_vertikal_persen"] = round((1 - abs(nose_y - hip_mid_y)) * 100, 1)
        data["simetri_bahu_persen"]    = round((1 - abs(lm[11].y - lm[12].y)) * 100, 1)
        return data
    except Exception as e:
        logger.warning(f"Landmark error: {e}")
        return {}


def draw_skeleton_pillow(img: Image.Image, landmarks) -> Image.Image:
    """Gambar skeleton pose di atas gambar PIL tanpa OpenCV."""
    w, h   = img.size
    draw   = ImageDraw.Draw(img)
    lm     = landmarks.landmark

    def px(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Gambar koneksi
    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
        try:
            x1, y1 = px(start_idx)
            x2, y2 = px(end_idx)
            draw.line([(x1, y1), (x2, y2)], fill=CONNECTION_COLOR, width=LINE_WIDTH)
        except Exception:
            pass

    # Gambar titik landmark
    for i in range(33):
        try:
            x, y = px(i)
            draw.ellipse(
                [(x - DOT_RADIUS, y - DOT_RADIUS), (x + DOT_RADIUS, y + DOT_RADIUS)],
                fill=LANDMARK_COLOR,
            )
        except Exception:
            pass

    return img


def process_image_for_pose(image_path: str) -> tuple[dict, str | None]:
    """Deteksi pose & gambar skeleton. Return (landmark_data, output_path)."""
    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Gagal buka gambar: {e}")
        return {}, None

    img_array = pil_to_rgb_array(pil_img)
    w, h      = pil_img.size

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        results = pose.process(img_array)

        if not results.pose_landmarks:
            return {}, None

        landmark_data = analyze_pose_landmarks(results.pose_landmarks, w, h)

        # Gambar skeleton
        skeleton_img = pil_img.copy()
        skeleton_img = draw_skeleton_pillow(skeleton_img, results.pose_landmarks)

        output_path = image_path + "_pose.jpg"
        skeleton_img.save(output_path, "JPEG", quality=90)

        return landmark_data, output_path


def ask_claude_analysis(image_path: str, landmark_data: dict, mode: str = "general") -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    landmark_text = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}{'°' if 'sudut' in k else '%'}"
        for k, v in landmark_data.items()
    ) if landmark_data else "Data landmark tidak tersedia."

    mode_prompts = {
        "general":   "Analisis pose secara umum: postur tubuh, keseimbangan, dan kesan keseluruhan.",
        "olahraga":  "Analisis dari sudut pandang atletik: teknik, efisiensi gerakan, dan potensi cedera.",
        "ergonomi":  "Analisis ergonomi kerja: risiko MSDs, rekomendasi perbaikan postur.",
        "kesehatan": "Analisis kesehatan postur: kelainan postur, kelengkungan tulang belakang, saran perbaikan.",
    }

    prompt = f"""Kamu adalah ahli analisis biomekanik dan pose manusia.
Analisis gambar pose manusia berikut beserta data landmark yang telah diekstrak.

DATA SUDUT SENDI (MediaPipe):
{landmark_text}

FOKUS ANALISIS: {mode_prompts.get(mode, mode_prompts['general'])}

Format respons (gunakan emoji):

🏃 **RINGKASAN POSE**
[Deskripsi singkat pose]

📐 **ANALISIS SENDI & SUDUT**
[Interpretasi sudut-sudut signifikan]

⚠️ **POTENSI MASALAH**
[Risiko atau ketidakseimbangan]

✅ **REKOMENDASI**
[Saran perbaikan atau latihan]

📊 **SKOR POSTUR: X/100**
[Penjelasan skor]

Gunakan Bahasa Indonesia yang jelas."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text


# ── Telegram Handlers ─────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Motion Control AI Bot*\n\n"
        "Bot analisis *pose manusia* dari foto menggunakan MediaPipe \\+ Claude AI\\.\n\n"
        "📸 *Cara pakai:*\n"
        "1\\. Kirim foto\n"
        "2\\. Pilih mode analisis\n"
        "3\\. Terima laporan lengkap\\!\n\n"
        "/help \\- Panduan lengkap",
        parse_mode="MarkdownV2",
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Panduan Bot*\n\n"
        "*Mode Analisis:*\n"
        "🔍 General \\- Pose umum\n"
        "🏋️ Olahraga \\- Teknik & performa\n"
        "💺 Ergonomi \\- Postur kerja\n"
        "🏥 Kesehatan \\- Kelainan postur\n\n"
        "*Tips foto:*\n"
        "• Seluruh tubuh terlihat\n"
        "• Pencahayaan cukup\n"
        "• Latar kontras dengan tubuh",
        parse_mode="MarkdownV2",
    )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    context.user_data["pending_photo_id"] = photo.file_id

    keyboard = [
        [InlineKeyboardButton("🔍 General",  callback_data="mode_general"),
         InlineKeyboardButton("🏋️ Olahraga", callback_data="mode_olahraga")],
        [InlineKeyboardButton("💺 Ergonomi", callback_data="mode_ergonomi"),
         InlineKeyboardButton("🏥 Kesehatan",callback_data="mode_kesehatan")],
    ]
    await update.message.reply_text(
        "📸 Foto diterima! Pilih mode analisis:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def mode_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    mode = query.data.replace("mode_", "")
    labels = {"general": "🔍 General", "olahraga": "🏋️ Olahraga",
              "ergonomi": "💺 Ergonomi", "kesehatan": "🏥 Kesehatan"}
    label = labels.get(mode, mode)

    await query.edit_message_text(f"⏳ Menganalisis dengan mode *{label}*...", parse_mode="Markdown")

    file_id = context.user_data.get("pending_photo_id")
    if not file_id:
        await query.edit_message_text("❌ Foto tidak ditemukan. Kirim ulang foto.")
        return

    tmp_path = pose_path = None
    try:
        file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        await query.edit_message_text("🦴 Mendeteksi landmark pose...")
        landmark_data, pose_path = process_image_for_pose(tmp_path)

        if not landmark_data:
            await query.edit_message_text(
                "⚠️ *Pose tidak terdeteksi.*\n\n"
                "Tips:\n• Pastikan seluruh tubuh terlihat\n"
                "• Gunakan foto dengan pencahayaan baik",
                parse_mode="Markdown",
            )
            return

        await query.edit_message_text("🤖 Claude AI sedang menganalisis...")
        analysis = ask_claude_analysis(tmp_path, landmark_data, mode)

        # Kirim skeleton
        if pose_path and Path(pose_path).exists():
            with open(pose_path, "rb") as f:
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=f,
                    caption=f"🦴 Skeleton pose (mode: {label})",
                )

        # Kirim analisis (max 4096 char per pesan Telegram)
        for i in range(0, len(analysis), 4000):
            await context.bot.send_message(chat_id=query.message.chat_id, text=analysis[i:i+4000])

        # Tombol analisis ulang
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ Selesai! Kirim foto lain atau pilih mode berbeda.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔄 Analisis Ulang", callback_data="reanalyze")
            ]]),
        )
        await query.edit_message_text(f"✅ Analisis mode *{label}* selesai!", parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error analisis: {e}", exc_info=True)
        await query.edit_message_text(f"❌ Error: `{str(e)[:200]}`\n\nCoba kirim foto ulang.", parse_mode="Markdown")
    finally:
        for p in [tmp_path, pose_path]:
            if p and Path(p).exists():
                try: Path(p).unlink()
                except: pass


async def reanalyze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not context.user_data.get("pending_photo_id"):
        await query.edit_message_text("⚠️ Silakan kirim foto baru.")
        return
    keyboard = [
        [InlineKeyboardButton("🔍 General",  callback_data="mode_general"),
         InlineKeyboardButton("🏋️ Olahraga", callback_data="mode_olahraga")],
        [InlineKeyboardButton("💺 Ergonomi", callback_data="mode_ergonomi"),
         InlineKeyboardButton("🏥 Kesehatan",callback_data="mode_kesehatan")],
    ]
    await query.edit_message_text("Pilih mode analisis:", reply_markup=InlineKeyboardMarkup(keyboard))


async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❓ Kirim /help atau langsung kirim *foto* untuk dianalisis! 📸", parse_mode="Markdown")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        print("⚠️  Set TELEGRAM_BOT_TOKEN di Railway → Variables!")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help",  help_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(CallbackQueryHandler(reanalyze_callback,    pattern="^reanalyze$"))
    app.add_handler(CallbackQueryHandler(mode_callback_handler, pattern="^mode_"))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_handler))

    print("🤖 Motion Control AI Bot berjalan...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
