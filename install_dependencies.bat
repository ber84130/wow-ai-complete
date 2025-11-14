@echo off
REM ============================================================
REM WoW AI Complete - Installation automatique des dépendances
REM ============================================================

echo ========================================
echo WoW AI Complete - Installation
echo ========================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe!
    echo.
    echo Telechargez Python 3.10 ou superieur depuis:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Cochez "Add Python to PATH" pendant l'installation!
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte
python --version
echo.

REM Mise à jour pip
echo [ETAPE 1/7] Mise a jour de pip...
python -m pip install --upgrade pip
echo.

REM Installation des dépendances de base
echo [ETAPE 2/7] Installation pynput (controles clavier/souris)...
python -m pip install pynput==1.7.6
echo.

echo [ETAPE 3/7] Installation pillow (screenshots)...
python -m pip install pillow==10.0.0
echo.

echo [ETAPE 4/7] Installation opencv-python (vision)...
python -m pip install opencv-python==4.8.0.76
echo.

echo [ETAPE 5/7] Installation numpy (calculs)...
REM Version compatible avec Python 3.12
python -m pip install numpy==1.26.4
echo.

echo [ETAPE 6/7] Installation psutil et GPUtil (specs PC)...
python -m pip install psutil==5.9.5 GPUtil==1.4.0
echo.

REM Détection GPU NVIDIA pour PyTorch
echo [ETAPE 7/7] Installation PyTorch...
echo.
echo Detection GPU...

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] GPU NVIDIA detecte - Installation PyTorch avec CUDA
    REM Utilisation d'une version disponible sur l'index cu118
    python -m pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] Pas de GPU NVIDIA - Installation PyTorch CPU uniquement
    python -m pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
)
echo.

REM Vérification finale
echo ========================================
echo Verification de l'installation...
echo ========================================
echo.

python -c "import pynput; print('[OK] pynput')" 2>nul || echo [ERREUR] pynput
python -c "import PIL; print('[OK] pillow')" 2>nul || echo [ERREUR] pillow
python -c "import cv2; print('[OK] opencv')" 2>nul || echo [ERREUR] opencv
python -c "import numpy; print('[OK] numpy')" 2>nul || echo [ERREUR] numpy
python -c "import torch; print('[OK] torch')" 2>nul || echo [ERREUR] torch
python -c "import psutil; print('[OK] psutil')" 2>nul || echo [ERREUR] psutil
python -c "import GPUtil; print('[OK] GPUtil')" 2>nul || echo [ERREUR] GPUtil

echo.
echo ========================================
echo Installation terminee!
echo ========================================
echo.
echo Vous pouvez maintenant lancer:
echo   python wow_ai_complete_fixed.py
echo.
pause
