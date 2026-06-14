"""
Motion Control AI - Telegram Bot
Analisis pose manusia dari foto menggunakan MediaPipe Tasks API + Claude AI
Kompatibel dengan mediapipe >= 0.10.x (Python 3.13)
"""

import os
import logging
import tempfile
import base64
import urllib.request
from pathlib import Path

import anthropic
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from PIL import Image, ImageDraw
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Konfigurasi ───────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
anthropic_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL_PATH = "/app/pose_landmarker_full.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

# Koneksi skeleton MediaPipe Pose (33 landmark)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

LANDMARK_COLOR   = (0, 255, 100)
CONNECTION_COLOR = (0, 180, 255)


def download_model():
    """Download model MediaPipe jika belum ada."""
    if not Path(MODEL_PATH).exists():
        logger.info("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model downloaded.")


def angle_between(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return round(float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))), 1)


def analyze_landmarks(landmarks, w: int, h: int) -> dict:
    """Hitung sudut sendi dari landmark pose."""
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x, lm.y)

    try:
        return {
            "sudut_siku_kiri":     angle_between(pt(11), pt(13), pt(15)),
            "sudut_siku_kanan":    angle_between(pt(12), pt(14), pt(16)),
            "sudut_bahu_kiri":     angle_between(pt(23), pt(11), pt(13)),
            "sudut_bahu_kanan":    angle_between(pt(24), pt(12), pt(14)),
            "sudut_lutut_kiri":    angle_between(pt(23), pt(25), pt(27)),
            "sudut_lutut_kanan":   angle_between(pt(24), pt(26), pt(28)),
            "sudut_pinggul_kiri":  angle_between(pt(11), pt(23), pt(25)),
            "sudut_pinggul_kanan": angle_between(pt(12), pt(24), pt(26)),
            "postur_vertikal_persen": round(
                (1 - abs(landmarks[0].y - (landmarks[23].y + landmarks[24].y) / 2)) * 100, 1),
            "simetri_bahu_persen": round(
                (1 - abs(landmarks[11].y - landmarks[12].y)) * 100, 1),
        }
    except Exception as e:
        logger.warning(f"Landmark error: {e}")
        return {}


def draw_skeleton(img: Image.Image, landmarks) -> Image.Image:
    """Gambar skeleton di atas gambar PIL."""
    w, h = img.size
    draw = ImageDraw.Draw(img)

    def px(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    for s, e in POSE_CONNECTIONS:
        try:
            draw.line([px(s), px(e)], fill=CONNECTION_COLOR, width=2)
        except Exception:
            pass

    for i in range(33):
        try:
            x, y = px(i)
            draw.ellipse([(x-4, y-4), (x+4, y+4)], fill=LANDMARK_COLOR)
        except Exception:
            pass

    return img


def process_image(image_path: str) -> tuple[dict, str | None]:
    """Deteksi pose & gambar skeleton. Return (data, output_path)."""
    download_model()

    pil_img = Image.open(image_path).convert("RGB")
    w, h    = pil_img.size

    # Buat MediaPipe image dari numpy array
    img_array = np.array(pil_img)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    # Setup PoseLandmarker (Tasks API)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return {}, None

    landmarks     = result.pose_landmarks[0]
    landmark_data = analyze_landmarks(landmarks, w, h)

    skeleton_img  = draw_skeleton(pil_img.copy(), landmarks)
    output_path   = image_path + "_pose.jpg"
    skeleton_img.save(output_path, "JPEG", quality=90)

    return landmark_data, output_path


def ask_claude(image_path: str, landmark_data: dict, mode: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode()

    landmark_text = "\n".join(
        f"- {k.replace('_',' ').title()}: {v}{'°' if 'sudut' in k else '%'}"
        for k, v in landmark_data.items()
    ) if landmark_data else "Tidak tersedia."

    mode_focus = {
        "general":   "Analisis pose secara umum: postur, keseimbangan, kesan keseluruhan.",
        "olahraga":  "Analisis atletik: teknik, efisiensi gerakan, potensi cedera.",
        "ergonomi":  "Analisis ergonomi kerja: risiko MSDs, rekomendasi postur.",
        "kesehatan": "Analisis kesehatan: kelainan postur, tulang belakang, saran perbaikan.",
    }.get(mode, "Analisis umum.")

    prompt = f"""Kamu adalah ahli biomekanik. Analisis pose manusia ini.

DATA SUDUT SENDI:
{landmark_text}

FOKUS: {mode_focus}

Jawab dalam format (gunakan emoji):

🏃 **RINGKASAN POSE**
📐 **ANALISIS SENDI**
⚠️ **POTENSI MASALAH**
✅ **REKOMENDASI**
📊 **SKOR POSTUR: X/100**

Bahasa Indonesia, jelas dan mudah dipahami."""

    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return resp.content[0].text


# ── Telegram Handlers ─────────────────────────────────────────────────────────

MODE_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("🔍 General",  callback_data="mode_general"),
     InlineKeyboardButton("🏋️ Olahraga", callback_data="mode_olahraga")],
    [InlineKeyboardButton("💺 Ergonomi", callback_data="mode_ergonomi"),
     InlineKeyboardButton("🏥 Kesehatan",callback_data="mode_kesehatan")],
])

LABELS = {"general":"🔍 General","olahraga":"🏋️ Olahraga","ergonomi":"💺 Ergonomi","kesehatan":"🏥 Kesehatan"}


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Motion Control AI Bot*\n\n"
        "Analisis pose manusia dari foto menggunakan AI\\.\n\n"
        "Kirim foto untuk mulai\\! /help untuk panduan\\.",
        parse_mode="MarkdownV2",
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Panduan*\n\n"
        "Kirim foto → pilih mode analisis:\n"
        "🔍 General • 🏋️ Olahraga • 💺 Ergonomi • 🏥 Kesehatan\n\n"
        "*Tips foto terbaik:*\n"
        "• Seluruh tubuh terlihat\n• Pencahayaan cukup\n• Latar kontras",
        parse_mode="Markdown",
    )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["pending_photo_id"] = update.message.photo[-1].file_id
    await update.message.reply_text("📸 Foto diterima! Pilih mode analisis:", reply_markup=MODE_KEYBOARD)


async def mode_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode  = query.data.replace("mode_", "")
    label = LABELS.get(mode, mode)

    file_id = context.user_data.get("pending_photo_id")
    if not file_id:
        await query.edit_message_text("❌ Foto tidak ditemukan. Kirim ulang foto.")
        return

    tmp_path = pose_path = None
    try:
        await query.edit_message_text(f"⏳ Mode *{label}* — mendownload foto...", parse_mode="Markdown")
        file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

        await query.edit_message_text("🦴 Mendeteksi pose...")
        landmark_data, pose_path = process_image(tmp_path)

        if not landmark_data:
            await query.edit_message_text(
                "⚠️ *Pose tidak terdeteksi.*\n\nTips:\n• Pastikan seluruh tubuh terlihat\n• Gunakan foto lebih terang",
                parse_mode="Markdown",
            )
            return

        await query.edit_message_text("🤖 Claude AI sedang menganalisis...")
        analysis = ask_claude(tmp_path, landmark_data, mode)

        if pose_path and Path(pose_path).exists():
            with open(pose_path, "rb") as f:
                await context.bot.send_photo(
                    chat_id=query.message.chat_id, photo=f,
                    caption=f"🦴 Skeleton pose — mode {label}",
                )

        for i in range(0, len(analysis), 4000):
            await context.bot.send_message(chat_id=query.message.chat_id, text=analysis[i:i+4000])

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ Selesai! Kirim foto lain atau analisis ulang.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Analisis Ulang", callback_data="reanalyze")]]),
        )
        await query.edit_message_text(f"✅ Analisis *{label}* selesai!", parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await query.edit_message_text(f"❌ Error: `{str(e)[:300]}`", parse_mode="Markdown")
    finally:
        for p in [tmp_path, pose_path]:
            if p and Path(p).exists():
                try: Path(p).unlink()
                except: pass


async def reanalyze_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not context.user_data.get("pending_photo_id"):
        await query.edit_message_text("⚠️ Kirim foto baru terlebih dahulu.")
        return
    await query.edit_message_text("Pilih mode analisis:", reply_markup=MODE_KEYBOARD)


async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Kirim /help atau langsung kirim 📸 *foto* untuk dianalisis!", parse_mode="Markdown")


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
