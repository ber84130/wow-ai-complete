@echo off
REM ============================================================
REM WoW AI Complete - Lanceur rapide
REM ============================================================

echo ========================================
echo WoW AI Complete - Leveling ^& Endgame Bot
echo ========================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python non trouve!
    echo Lancez d'abord: install_dependencies.bat
    pause
    exit /b 1
)

echo [OK] Python detecte
echo.

REM Vérifier le fichier principal
if not exist "wow_ai_complete_fixed.py" (
    echo [ERREUR] Fichier wow_ai_complete_fixed.py introuvable!
    echo Verifiez que vous etes dans le bon dossier.
    pause
    exit /b 1
)

echo Lancement de WoW AI Complete...
echo.
echo RAPPEL:
echo 1. Lancez World of Warcraft
echo 2. Tapez /combatlog dans le jeu
echo 3. Utilisez l'interface pour configurer
echo.

REM Lancer l'application
python wow_ai_complete_fixed.py

REM En cas d'erreur
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] L'application a rencontre une erreur
    echo.
    pause
)
