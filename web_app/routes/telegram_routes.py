"""
Telegram Bot Webhook Routes.

Provides webhook endpoints for Telegram Bot API integration.
"""
import os
import logging
import asyncio
import hmac
import requests
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required

from web_app.security.security_decorators import require_role

logger = logging.getLogger(__name__)

telegram_bp = Blueprint("telegram", __name__, url_prefix="/api/telegram")

# Global event loop for webhook processing
_event_loop: asyncio.AbstractEventLoop = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop


def _get_bot_token() -> str:
    """Get Telegram bot token from config or environment."""
    # Try environment first
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        return token
    
    # Try app config
    try:
        token = current_app.config.get("TELEGRAM_BOT_TOKEN")
        if token:
            return token
    except Exception:
        pass
    
    return ""


def _get_webhook_secret() -> str:
    """Get webhook secret for verification."""
    secret = (os.environ.get("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if secret:
        return secret

    try:
        secret = str(current_app.config.get("TELEGRAM_WEBHOOK_SECRET", "")).strip()
        if secret:
            return secret
    except Exception:
        pass

    try:
        from web_app.services import load_configuration
        cfg = load_configuration()
        secret = str((cfg.get("app", {}) or {}).get("webhook_secret", "")).strip()
        if secret:
            return secret
    except Exception:
        pass

    return ""


def _is_production() -> bool:
    env = (os.environ.get("AUTOOCR_ENV") or os.environ.get("FLASK_ENV") or "").strip().lower()
    if env:
        return env == "production"
    try:
        return str(current_app.config.get("ENV", "")).strip().lower() == "production"
    except Exception:
        return False


async def _process_telegram_update(bot, update_data: Dict[str, Any], bot_token: str) -> None:
    """Process a single Telegram update through python-telegram-bot."""
    from telegram import Update
    from telegram.ext import Application

    app = getattr(bot, "application", None)
    if app is None:
        app = Application.builder().token(bot_token).build()
        bot.setup_handlers(app)
        await app.initialize()
        bot.application = app

    update = Update.de_json(update_data, app.bot)
    if update is not None:
        await app.process_update(update)


@telegram_bp.route("/webhook", methods=["POST"])
def telegram_webhook():
    """
    Handle incoming Telegram updates via webhook.
    
    This endpoint receives all messages and commands sent to the bot.
    """
    bot_token = _get_bot_token()
    if not bot_token:
        logger.error("Telegram bot token not configured")
        return jsonify({"error": "Bot not configured"}), 500

    expected_secret = _get_webhook_secret()
    provided_secret = (request.headers.get("X-Telegram-Bot-Api-Secret-Token") or "").strip()
    if expected_secret:
        if not hmac.compare_digest(provided_secret, expected_secret):
            logger.warning("Rejected Telegram webhook request with invalid secret token")
            return jsonify({"error": "Forbidden"}), 403
    elif _is_production():
        logger.error("TELEGRAM_WEBHOOK_SECRET is required in production")
        return jsonify({"error": "Webhook secret not configured"}), 500
    elif not provided_secret:
        # No secret configured and not in production - warn about insecure mode
        logger.warning("Telegram webhook running WITHOUT secret verification - DO NOT USE IN PRODUCTION")
    
    # Get update from request
    try:
        update_data = request.get_json(force=True, silent=True)
        if not update_data:
            return jsonify({"error": "No data provided"}), 400
    except Exception as e:
        logger.error(f"Failed to parse Telegram update: {e}")
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Log update for debugging
    logger.info(f"Telegram update received: {update_data.get('update_id', 'unknown')}")
    
    # Process update asynchronously
    # In production, use a task queue (Celery)
    try:
        from modules.telegram_bot import get_telegram_bot
        
        bot = get_telegram_bot(bot_token)
        if not bot:
            return jsonify({"error": "Bot not available"}), 500
        
        loop = _get_event_loop()
        coro = _process_telegram_update(bot, update_data, bot_token)
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result(timeout=20)
        else:
            loop.run_until_complete(coro)

        logger.info(f"Telegram message processed: {update_data.get('message', {}).get('text', '')}")
             
    except Exception as e:
        logger.error(f"Error processing Telegram update: {e}")
        # Still return 200 to acknowledge receipt
        # Telegram will retry on error
    
    return jsonify({"ok": True})


@telegram_bp.route("/webhook", methods=["GET"])
def telegram_webhook_verify():
    """
    Verify webhook setup with Telegram.
    """
    bot_token = _get_bot_token()
    if not bot_token:
        return jsonify({
            "status": "error",
            "message": "Bot token not configured"
        }), 500
    
    return jsonify({
        "status": "ok",
        "message": "Telegram webhook endpoint is active",
        "endpoints": {
            "webhook": "/api/telegram/webhook",
            "commands": [
                "/start", "/ayuda", "/buscar", "/subir",
                "/misdocs", "/stats", "/login", "/logout"
            ]
        }
    })


@telegram_bp.route("/set-webhook", methods=["POST"])
@login_required
@require_role("ADMIN")
def set_telegram_webhook():
    """
    Configure the Telegram webhook URL.
    
    Request JSON:
        - url (str): The webhook URL to set
        - secret_token (str, optional): Secret token for verification
    """
    data = request.get_json(force=True, silent=True) or {}
    webhook_url = data.get("url")
    
    if not webhook_url:
        return jsonify({"error": "URL is required"}), 400
    if _is_production() and not str(webhook_url).startswith("https://"):
        return jsonify({"error": "Webhook URL must use HTTPS in production"}), 400
    
    bot_token = _get_bot_token()
    if not bot_token:
        return jsonify({"error": "Bot token not configured"}), 500
    
    # Call Telegram API to set webhook
    api_url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
    
    try:
        requested_secret = data.get("secret_token")
        if requested_secret is None:
            secret_token = _get_webhook_secret()
        else:
            secret_token = str(requested_secret).strip()

        payload = {"url": webhook_url}
        if secret_token:
            payload["secret_token"] = secret_token
        elif _is_production():
            return jsonify({"error": "secret_token is required in production"}), 400
        
        response = requests.post(
            api_url,
            json=payload,
            timeout=10
        )
        
        result = response.json()
        
        if result.get("ok"):
            logger.info(f"Webhook set to: {webhook_url}")
            return jsonify({
                "ok": True,
                "message": f"Webhook set to {webhook_url}"
            })
        else:
            logger.error(f"Failed to set webhook: {result}")
            return jsonify({
                "ok": False,
                "error": result.get("description", "Unknown error")
            }), 400
            
    except requests.RequestException as e:
        logger.error(f"Error setting webhook: {e}")
        return jsonify({"error": str(e)}), 500


@telegram_bp.route("/delete-webhook", methods=["POST"])
@login_required
@require_role("ADMIN")
def delete_telegram_webhook():
    """Remove the Telegram webhook."""
    bot_token = _get_bot_token()
    if not bot_token:
        return jsonify({"error": "Bot token not configured"}), 500
    
    api_url = f"https://api.telegram.org/bot{bot_token}/deleteWebhook"
    
    try:
        response = requests.post(api_url, timeout=10)
        result = response.json()
        
        if result.get("ok"):
            return jsonify({"ok": True, "message": "Webhook deleted"})
        else:
            return jsonify({"error": result.get("description")}), 400
            
    except requests.RequestException as e:
        logger.error(f"Error deleting webhook: {e}")
        return jsonify({"error": str(e)}), 500


@telegram_bp.route("/bot-info", methods=["GET"])
@login_required
@require_role("ADMIN")
def get_bot_info():
    """Get information about the Telegram bot."""
    bot_token = _get_bot_token()
    if not bot_token:
        return jsonify({"error": "Bot token not configured"}), 500
    
    api_url = f"https://api.telegram.org/bot{bot_token}/getMe"
    
    try:
        response = requests.get(api_url, timeout=10)
        result = response.json()
        
        if result.get("ok"):
            bot_info = result.get("result", {})
            return jsonify({
                "ok": True,
                "bot": {
                    "id": bot_info.get("id"),
                    "is_bot": bot_info.get("is_bot"),
                    "first_name": bot_info.get("first_name"),
                    "username": bot_info.get("username"),
                    "can_join_groups": bot_info.get("can_join_groups"),
                    "can_read_all_group_messages": bot_info.get("can_read_all_group_messages"),
                    "supports_inline_queries": bot_info.get("supports_inline_queries")
                }
            })
        else:
            return jsonify({"error": result.get("description")}), 400
            
    except requests.RequestException as e:
        logger.error(f"Error getting bot info: {e}")
        return jsonify({"error": str(e)}), 500


@telegram_bp.route("/commands", methods=["GET", "POST"])
@login_required
@require_role("ADMIN")
def manage_commands():
    """
    Get or set bot commands.
    GET: List current commands
    POST: Set new commands
    """
    bot_token = _get_bot_token()
    if not bot_token:
        return jsonify({"error": "Bot token not configured"}), 500
    
    if request.method == "GET":
        # Get current commands
        api_url = f"https://api.telegram.org/bot{bot_token}/getMyCommands"
        
        try:
            response = requests.get(api_url, timeout=10)
            result = response.json()
            
            if result.get("ok"):
                return jsonify({
                    "ok": True,
                    "commands": result.get("result", [])
                })
            else:
                return jsonify({"error": result.get("description")}), 400
                
        except requests.RequestException as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # Set commands
        data = request.get_json(force=True, silent=True) or {}
        commands = data.get("commands", [])
        
        if not commands:
            return jsonify({"error": "Commands list is required"}), 400
        
        api_url = f"https://api.telegram.org/bot{bot_token}/setMyCommands"
        
        try:
            response = requests.post(api_url, json={"commands": commands}, timeout=10)
            result = response.json()
            
            if result.get("ok"):
                return jsonify({"ok": True, "message": "Commands updated"})
            else:
                return jsonify({"error": result.get("description")}), 400
                
        except requests.RequestException as e:
            return jsonify({"error": str(e)}), 500


@telegram_bp.route("/status", methods=["GET"])
@login_required
@require_role("ADMIN")
def telegram_status():
    """Get Telegram bot integration status."""
    bot_token = _get_bot_token()
    
    return jsonify({
        "ok": True,
        "configured": bool(bot_token),
        "module_available": True,
        "webhook_url": "/api/telegram/webhook",
        "commands": [
            {"command": "start", "description": "Iniciar el bot"},
            {"command": "buscar", "description": "Buscar documentos"},
            {"command": "subir", "description": "Subir documento para OCR"},
            {"command": "misdocs", "description": "Ver mis documentos"},
            {"command": "stats", "description": "Estadísticas"},
            {"command": "login", "description": "Vincular cuenta AutoOCR"},
            {"command": "logout", "description": "Desvincular cuenta"},
            {"command": "ayuda", "description": "Mostrar ayuda"},
        ]
    })
