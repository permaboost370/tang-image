import os
import io
import logging
from typing import Optional
from collections import deque

from fastapi import FastAPI, Request, HTTPException

from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatType, ChatAction

from ref_loader import load_reference_png_bytes
from image_provider import generate_image_from_reference, ImageGenError

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# =========================
# Environment & Settings
# =========================
BOT_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
PUBLIC_URL     = os.getenv("PUBLIC_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change_me")
WEBHOOK_PATH   = os.getenv("WEBHOOK_PATH", "hook")

if not BOT_TOKEN or not PUBLIC_URL:
    raise RuntimeError("Missing TELEGRAM_TOKEN or PUBLIC_URL")

# --- Optional user access control ---
_allow_ids = set()
_allow_env = (os.getenv("ALLOW_USER_IDS") or "").strip()
if _allow_env:
    for s in _allow_env.split(","):
        s = s.strip()
        if s.isdigit():
            _allow_ids.add(int(s))

ACCESS_CODE = (os.getenv("ACCESS_CODE") or "").strip()
_redeemed_ids = set()  # in-memory

# =========================
# App Skeletons
# =========================
app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

HELP = (
    "I generate images from your *text prompt* using this project's fixed reference image.\n\n"
    "In groups: use `/pic <prompt>`.\n"
    "In DM: send any prompt text (no command needed) or use `/pic <prompt>`.\n\n"
    "Commands:\n"
    "/start – welcome\n"
    "/help – how to use\n"
    "/info – what I do\n"
    "/status – service status\n"
    "/redeem <CODE> – unlock access with a code (if required)\n"
)

# Will hold the fixed reference PNG bytes after startup
REF_BYTES: Optional[bytes] = None

# =========================
# De-duplication guard
# =========================
_SEEN_MSGS = set()
_SEEN_ORDER = deque()
_MAX_SEEN = 5000

def _already_processed(update: Update) -> bool:
    msg = update.message
    if not msg:
        return False
    key = (msg.chat.id, msg.message_id)
    if key in _SEEN_MSGS:
        return True
    _SEEN_MSGS.add(key)
    _SEEN_ORDER.append(key)
    if len(_SEEN_ORDER) > _MAX_SEEN:
        old = _SEEN_ORDER.popleft()
        _SEEN_MSGS.discard(old)
    return False

# =========================
# Helpers
# =========================
def _is_allowed_user(user_id: int) -> bool:
    if _allow_ids:
        if user_id in _allow_ids:
            return True
        if ACCESS_CODE and user_id in _redeemed_ids:
            return True
        return False

    if ACCESS_CODE:
        return user_id in _redeemed_ids
    return True

def extract_pic_prompt(update: Update) -> str:
    text = (update.message.text or "").strip()
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    if not parts:
        return ""
    if parts[0].startswith("/pic"):
        if len(parts) == 1:
            return ""
        return parts[1].strip()
    return ""

# =========================
# Handlers
# =========================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return
    await update.message.reply_text("Hey! 👋\n" + HELP, disable_web_page_preview=True)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return
    await update.message.reply_text(HELP, disable_web_page_preview=True)

async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return
    await update.message.reply_text(
        "This bot generates images from *your text prompt* using a fixed project reference image. 🎯",
        disable_web_page_preview=True,
    )

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return
    ok_ref = "✅" if REF_BYTES else "❌"
    ok_wh  = "✅ (webhook mode)"
    await update.message.reply_text(f"Ref image: {ok_ref}\nWebhook: {ok_wh}")

async def redeem_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return
    if not ACCESS_CODE:
        await update.message.reply_text("Redeem not required.")
        return
    parts = (update.message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /redeem <CODE>")
        return
    code = parts[1].strip()
    if code == ACCESS_CODE:
        _redeemed_ids.add(update.effective_user.id)
        await update.message.reply_text("✅ Access unlocked!")
    else:
        await update.message.reply_text("❌ Invalid code.")

async def _generate_and_reply(msg, user_prompt: str):
    await msg.chat.send_action(ChatAction.UPLOAD_PHOTO)
    try:
        out_png = await generate_image_from_reference(user_prompt, REF_BYTES)
    except ImageGenError as e:
        err_text = str(e).lower()
        if "moderation" in err_text or "safety" in err_text or "rejected" in err_text:
            await msg.reply_text("🚫 Sorry, this image can’t be generated due to copyright or content restrictions.")
        else:
            await msg.reply_text("⚠️ Something went wrong while generating the image. Please try again later.")
        return

    await msg.reply_photo(
        photo=InputFile(io.BytesIO(out_png), filename="result.png"),
        caption=f"✅ Done!\nPrompt: {user_prompt}",
    )

async def pic_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return

    user_id = update.effective_user.id
    if not _is_allowed_user(user_id):
        if ACCESS_CODE:
            await update.message.reply_text("🔒 Access restricted. If you have a code, DM me and send `/redeem <CODE>`.")
        else:
            await update.message.reply_text("🔒 Access restricted.")
        return

    user_prompt = extract_pic_prompt(update)
    if not user_prompt:
        await update.message.reply_text("Usage: `/pic <your prompt>`")
        return

    log.info(f"/pic from user={user_id} chat={update.effective_chat.id} msg={update.message.message_id}")
    await _generate_and_reply(update.message, user_prompt)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if _already_processed(update):
        return

    msg = update.message
    if not msg or not msg.text:
        return

    if msg.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        return

    user_id = update.effective_user.id
    if not _is_allowed_user(user_id):
        if ACCESS_CODE:
            await msg.reply_text("🔒 Access restricted. If you have a code, send `/redeem <CODE>`.")
        else:
            await msg.reply_text("🔒 Access restricted.")
        return

    user_prompt = msg.text.strip()
    if not user_prompt or user_prompt.startswith("/"):
        return

    log.info(f"text prompt from user={user_id} chat={msg.chat.id} msg={msg.message_id}")
    await _generate_and_reply(msg, user_prompt)

# Register handlers
tg_app.add_handler(CommandHandler("start", start_cmd))
tg_app.add_handler(CommandHandler("help", help_cmd))
tg_app.add_handler(CommandHandler("info", info_cmd))
tg_app.add_handler(CommandHandler("status", status_cmd))
tg_app.add_handler(CommandHandler("redeem", redeem_cmd))
tg_app.add_handler(CommandHandler("pic", pic_cmd))
tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# =========================
# FastAPI Webhook
# =========================
@app.get("/")
async def root():
    return {"ok": True, "status": "up"}

@app.post(f"/{WEBHOOK_PATH}")
async def telegram_webhook(request: Request):
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret header")

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

# =========================
# Lifecycle
# =========================
@app.on_event("startup")
async def on_startup():
    global REF_BYTES
    await tg_app.initialize()
    REF_BYTES = await load_reference_png_bytes()
    await tg_app.bot.set_webhook(
        url=f"{PUBLIC_URL}/{WEBHOOK_PATH}",
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True,
    )
    await tg_app.start()
    log.info("Startup complete.")

@app.on_event("shutdown")
async def on_shutdown():
    await tg_app.stop()
    await tg_app.shutdown()
    log.info("Shutdown complete.")

# =========================
# Local dev
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=True)
