@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set "ROOT=%~dp0"
set "PIP_CACHE_DIR=%ROOT%pip-cache"
set "HF_HOME=%ROOT%models"
set "APPDATA=%ROOT%appdata"
"%ROOT%venv\Scripts\python.exe" "%ROOT%run_whisper_key_local.py"
