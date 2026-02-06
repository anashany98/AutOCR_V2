@echo off
echo 🐳 Initializing AutoOCR in Docker (GPU Mode)...

:: Stop any local running instances
taskkill /F /IM python.exe /T >nul 2>&1

echo 🛑 Local server stopped.
echo 🏗️  Building Docker container... (This may take a while first time)

:: Check for docker availability
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Docker is NOT installed or not in PATH.
    echo Please install Docker Desktop for Windows.
    pause
    exit /b 1
)

:: Run composition
docker compose up --build --force-recreate -d

if %errorlevel% neq 0 (
    echo ⚠️ 'docker compose' command failed. Trying legacy 'docker-compose'...
    docker-compose up --build -d
)

echo.
echo ✅ Container started in background!
echo 📊 View logs with: docker logs -f autoocr_gpu
echo.
echo 🌍 Web Interface: http://localhost:8081
echo.
pause
