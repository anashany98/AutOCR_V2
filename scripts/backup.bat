@echo off
setlocal
TITLE AutoOCR Backup System

:: Configuration
set "PROJECT_ROOT=%~dp0.."
set "BACKUP_ROOT=%PROJECT_ROOT%\backups"
set "TIMESTAMP=%date:~6,4%-%date:~3,2%-%date:~0,2%_%time:~0,2%-%time:~3,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "TARGET_DIR=%BACKUP_ROOT%\%TIMESTAMP%"

ECHO ===============================================================================
ECHO   AUTOOCR BACKUP STARTED
ECHO   Target: %TARGET_DIR%
ECHO ===============================================================================

:: Create backup directory
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

:: Backup SQLite Database (if exists)
if exist "%PROJECT_ROOT%\data\digitalizerai.db" (
    ECHO [INFO] Backing up SQLite database...
    copy "%PROJECT_ROOT%\data\digitalizerai.db" "%TARGET_DIR%\" >nul
)

:: Backup Configuration
ECHO [INFO] Backing up configuration...
copy "%PROJECT_ROOT%\config.yaml" "%TARGET_DIR%\" >nul

:: Backup Input Folder (Optional - uncomment to enable)
:: ECHO [INFO] Backing up input documents...
:: xcopy "%PROJECT_ROOT%\input" "%TARGET_DIR%\input\" /E /I /Y >nul

:: Placeholder for PostgreSQL Backup (pg_dump)
:: set PGPASSWORD=yourpassword
:: "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" -h localhost -U postgres autocr > "%TARGET_DIR%\autocr_dump.sql"

ECHO.
ECHO [SUCCESS] Backup completed successfully!
ECHO Location: %TARGET_DIR%
timeout /t 5
