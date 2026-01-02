#!/usr/bin/env python3
"""
Main entry point for AutOCR_V2 web interface.
Initialises logging, runs OCR health diagnostics, and starts the Flask app.
"""

import os
import sys

# Ensure root path is in sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger_setup import logger
from modules.startup_test import run_startup_test
from modules.email_importer import EmailImporter

# ✅ GPU Support for RTX 4070
import paddle

def setup_paddle_device():
    """Configure PaddlePaddle device based on GPU availability"""
    logger.info("🔧 Preparing PaddlePaddle runtime...")
    try:
        logger.info(
            "Env snapshot CUDA_VISIBLE_DEVICES={} PADDLEOCR_DISABLE_VLM={}",
            os.getenv("CUDA_VISIBLE_DEVICES", "<unset>"),
            os.getenv("PADDLEOCR_DISABLE_VLM", "<unset>"),
        )
        logger.info("PaddlePaddle version: {}", paddle.__version__)
        logger.info("Paddle compiled with CUDA: {}", paddle.is_compiled_with_cuda())
        if paddle.is_compiled_with_cuda():
            logger.info("Detected CUDA devices: {}", paddle.device.cuda.device_count())
            try:
                logger.info("Active CUDA device: {}", paddle.get_device())
            except Exception as device_exc:  # pragma: no cover - diagnostic
                logger.warning("Unable to query current Paddle device: {}", device_exc)

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            # Use GPU 0 by default, can be configured for multi-GPU
            paddle.set_device('gpu:0')
            logger.info("✅ PaddlePaddle GPU mode (RTX 4070 detected)")
        else:
            paddle.set_device('cpu')
            logger.info("✅ PaddlePaddle CPU mode (GPU not available)")
    except Exception as e:
        logger.warning(f"⚠️ PaddlePaddle device setup failed: {e}, falling back to CPU")
        paddle.set_device('cpu')

# ✅ NEW: Import OCR healthcheck
try:
    from modules.ocr_healthcheck import run_healthcheck
except ImportError:
    run_healthcheck = None


def main() -> None:
    logger.info("🚀 Starting AutOCR_V2...")

    # --- Step 0: Setup PaddlePaddle device ---
    setup_paddle_device()

    # --- Step 1: Run OCR Environment Diagnostic ---
    # Disabled to prevent double-loading models
    # if run_healthcheck:
    #     logger.info("🧠 Running OCR environment health check before app startup...")
    #     passed = run_healthcheck()
    #     if not passed:
    #         logger.warning("⚠️ OCR environment check reported issues. The app will continue but OCR may not function properly.")
    # else:
    #     logger.warning("⚠️ OCR healthcheck module not found. Skipping pre-launch diagnostic.")

    # --- Step 2: Import Flask app ---
    try:
        from web_app.app import app, init_app
    except ImportError as exc:
        logger.exception("❌ Failed to import web application: {}", exc)
        logger.info("👉 Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)

    # --- Step 3: Initialise App ---
        # Prevent caching of static files and templates during dev
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        
        init_app()
        logger.success("✅ Application initialised successfully.")

        # --- Start Email Importer (if enabled) ---
        from web_app.app import post_conf # Import config dict from initialized app context (or load config again)
        # Better: Load config here or access global config if available.
        # Since init_app loads config but doesn't expose it easily globally here, let's load it.
        from postbatch_processor import load_config
        pipeline_conf = load_config("config.yaml")
        email_conf = pipeline_conf.get("email_importer", {})
        
        if email_conf.get("enabled", False):
            input_root = pipeline_conf["postbatch"]["input_folder"]
            # Ensure input folder exists
            if not os.path.exists(input_root):
                os.makedirs(input_root, exist_ok=True)
                
            email_importer = EmailImporter(email_conf, input_root)
            email_importer.start()
            logger.info("📧 Email Importer background service started.")
    except Exception as exc:
        logger.exception("❌ Failed to initialise application: {}", exc)
        logger.info("🧩 Troubleshooting:")
        logger.info("1. Verify config.yaml exists and is properly configured.")
        logger.info("2. Ensure the database file exists or can be created.")
        logger.info("3. Check all OCR dependencies are installed and functional.")
        logger.info("4. Confirm file permissions for data directories.")
        sys.exit(1)

    # --- Step 4: Run startup diagnostic test (internal) ---
    try:
        run_startup_test()
    except Exception as e:
        logger.warning(f"⚠️ Startup self-test reported an issue: {e}")

    # --- Step 5: Launch web server ---
    debug_mode = os.getenv("FLASK_ENV", "").lower() == "development"
    port = int(os.getenv("PORT", "8000"))
    url = f"http://0.0.0.0:{port}"

    logger.info("🌐 Web interface available at {}", url)
    logger.info("📦 Flask debug mode: {}", debug_mode)
    app.run(debug=debug_mode, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

