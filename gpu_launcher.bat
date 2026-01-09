@echo off
setlocal
echo 🚀 Start AutoOCR (GPU Mode)...

:: Configurar rutas de CUDA y cuDNN
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

:: Verificar si el entorno virtual existe
if not exist "venv311\Scripts\python.exe" (
    echo ❌ Error: No se encuentra el entorno virtual venv311.
    pause
    exit /b 1
)

echo ✅ Environment configured. Launching server...
echo ⏳ Downloading models. Please wait...
.\venv311\Scripts\python.exe -u serve.py
pause
