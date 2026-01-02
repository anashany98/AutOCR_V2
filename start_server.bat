@echo off
TITLE AutoOCR V2.5 Production Server
CLS

ECHO ===============================================================================
ECHO   AUTOOCR V2.5 - PRODUCTION SERVER LAUNCHER
ECHO ===============================================================================
ECHO.
ECHO [INFO] Starting Waitress WSGI Server...
ECHO [INFO] Log file: web_app/logs/web_app.log
ECHO.

:: Check if python is available
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] Python not found in PATH!
    PAUSE
    EXIT /B 1
)

:: Run the production server
python serve.py

:: Keep window open if server crashes
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Server crashed or stopped unexpectedly!
    PAUSE
)
ECHO.
ECHO Server stopped.
PAUSE
