#!/usr/bin/env python3
"""Entrypoint to initialize the app and run it with waitress (production WSGI).
"""
import os
import sys

# ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger_setup import logger

def main() -> None:
    # Import and initialise the Flask app
    try:
        from web_app.app import app, init_app
    except Exception as exc:  # pragma: no cover - startup
        logger.exception("❌ Failed to import web application: %s", exc)
        raise

    try:
        init_app()
    except Exception as exc:
        logger.exception("❌ Failed to initialise application: %s", exc)
        raise

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Serving app with waitress on %s:%s", host, port)

    # Run with waitress (production WSGI). Import inside main to keep startup errors visible.
    try:
        from waitress import serve
        serve(app, host=host, port=port)
    except Exception as exc:
        logger.exception("❌ Waitress failed to start: %s", exc)
        raise


if __name__ == "__main__":
    main()
