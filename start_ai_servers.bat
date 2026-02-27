@echo off
title AutoOCR AI Launcher
color 0A

echo ==================================================
echo      AutoOCR High-Performance AI Launcher
echo ==================================================
echo.
echo [Target Hardware Config]
echo  - Text AI:   NVIDIA RTX 4070 #1 (Port 1234)
echo  - Vision AI: NVIDIA RTX 4070 #2 (Port 1235)
echo.
echo NOTE: This script requires LM Studio CLI ('lms') to be installed and in PATH.
echo.

echo [1/2] Starting Text Server (DeepSeek-R1-Distill-Llama-8B)...
REM Force GPU 0
set CUDA_VISIBLE_DEVICES=0
start "AutoOCR Text AI (Port 1234)" cmd /k "lms server start --port 1234 --model deepseek-r1-distill-llama-8b"

echo Waiting 5 seconds for initialization...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Vision Server (Qwen2-VL-7B-Instruct)...
REM Force GPU 1
set CUDA_VISIBLE_DEVICES=1
start "AutoOCR Vision AI (Port 1235)" cmd /k "lms server start --port 1235 --model qwen2-vl-7b-instruct"

echo.
echo ==================================================
echo        Servers Launching in Background
echo ==================================================
echo.
echo If windows close immediately, check if 'lms' command works in terminal.
echo.
pause
