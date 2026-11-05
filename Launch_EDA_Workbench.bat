@echo off
set SCRIPT_DIR=%~dp0
set OSS_CAD=C:\Users\surya\.oss-cad-suite\oss-cad-suite
call "%OSS_CAD%\environment.bat"
cd /d "%SCRIPT_DIR%"
start "" python gui_app.py
exit /b 0
