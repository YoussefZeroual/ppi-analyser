@echo off
setlocal
title PPI Analyser - Installation

echo ============================================
echo   PPI Analyser - Demarrage de l'installation
echo ============================================
echo.

:: --- Verifier droits admin ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Ce script necessite les droits administrateur.
    echo Clic droit sur ce fichier puis "Executer en tant qu'administrateur"
    pause
    exit /b 1
)

:: --- Verifier Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe.
    echo Installe Python depuis https://www.python.org/downloads/
    echo puis relance ce script.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo [OK] Python %PYVER% detecte.

:: --- Lancer le setup ---
echo.
echo [INFO] Lancement de l'interface d'installation...
python "%~dp0setup_ppi.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] Le script Python a rencontre une erreur.
    pause
)
endlocal
