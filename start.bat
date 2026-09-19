@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup is required.
  call setup.bat
)

if not exist ".venv\Scripts\python.exe" (
  echo Setup did not complete.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo Vid-go-clip closed because of an error.
  pause
)
