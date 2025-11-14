@echo off
REM ============================================================
REM Vérification rapide de l'installation
REM ============================================================

echo ========================================
echo Verification de l'installation
echo ========================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python non installe!
    echo.
    echo Telechargez Python depuis: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Lancer le script de vérification
python check_installation.py

pause
