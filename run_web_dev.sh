#!/bin/bash
watchmedo auto-restart \
    --directory=/app \
    --pattern="*.py" \
    --recursive \
    python run_web.py