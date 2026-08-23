@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set "ROOT=%~dp0"
set "HF_HOME=%ROOT%models"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
"%ROOT%venv\Scripts\python.exe" "%ROOT%stream_dictation.py"
