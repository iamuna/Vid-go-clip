@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo        VID-GO-CLIP SETUP
echo ========================================
echo.
echo Local-first setup. No paid AI API is required.
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.11+ was not found.
  where winget >nul 2>nul
  if not errorlevel 1 (
    set /p INSTALL_PYTHON="Install Python 3.11 with winget? [Y/N]: "
    if /I "%INSTALL_PYTHON%"=="Y" (
      winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
      echo.
      echo Python was installed. Close this window, then run setup.bat again.
      pause
      exit /b 0
    )
  )
  echo Install Python 3.11 or newer and enable Add Python to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating private Python environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)

echo Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing Vid-go-clip dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Checking FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo Installing free FFmpeg...
    winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
  ) else (
    echo WARNING: FFmpeg was not found. Install it before analyzing/exporting video.
  )
) else (
  echo FFmpeg detected.
)

echo.
echo Checking Ollama local AI...
set "OLLAMA_CMD="
where ollama >nul 2>nul
if not errorlevel 1 set "OLLAMA_CMD=ollama"

if not defined OLLAMA_CMD (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
  )
)

if not defined OLLAMA_CMD (
  where winget >nul 2>nul
  if not errorlevel 1 (
    echo Installing free local Ollama...
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
      set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    )
  )
)

if defined OLLAMA_CMD (
  echo.
  echo Checking Qwen3-VL 4B visual model...
  "%OLLAMA_CMD%" list | findstr /I "qwen3-vl:4b" >nul 2>nul
  if errorlevel 1 (
    echo Downloading qwen3-vl:4b. This is several GB and happens once.
    "%OLLAMA_CMD%" pull qwen3-vl:4b
    if errorlevel 1 (
      echo.
      echo Model download failed. If Ollama is old, run:
      echo   winget upgrade --id Ollama.Ollama
      echo Then rerun setup.bat.
    )
  ) else (
    echo qwen3-vl:4b detected.
  )
) else (
  echo WARNING: Ollama is not available yet.
  echo Install/start Ollama and run: ollama pull qwen3-vl:4b
)

echo.
echo ========================================
echo Setup finished.
echo Double-click start.bat to launch Vid-go-clip.
echo ========================================
pause
exit /b 0

:error
echo.
echo Setup failed. Copy the error above if you need help.
pause
exit /b 1
