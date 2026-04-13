@echo off
set "HERE=%~dp0"
cd /d "%HERE%"

if exist "%HERE%char_model.pth" if not exist "%HERE%model.pth" (
  copy /Y "%HERE%char_model.pth" "%HERE%model.pth" >nul
  echo Created model.pth from char_model.pth
  echo.
)

echo ========================================
echo  Handwriting app
echo ========================================
echo  Script: %HERE%app.py
echo  Folder: %HERE%
echo.
echo  After this starts, look for:
echo    Local:   http://127.0.0.1:PORT
echo    Public:  https://....gradio.live  (share with friends; link expires when you close this)
echo  Local-only: remove the line below that sets GRADIO_SHARE=1, or set GRADIO_SHARE=0.
echo ========================================
echo.

REM Public Gradio tunnel (needs internet). Comment out next line for local-only.
set "GRADIO_SHARE=1"

REM Try interpreters in order (adjust if your Python is elsewhere)
if exist "C:\Python314\python.exe" (
  echo Using C:\Python314\python.exe ...
  C:\Python314\python.exe "%HERE%app.py"
  goto :after
)
python "%HERE%app.py"
if errorlevel 1 (
  echo python failed, trying py ...
  py "%HERE%app.py"
)

:after
echo.
echo ----------------------------------------
if errorlevel 1 (
  echo Something failed ^(exit code %errorlevel%^). If nothing started, install Python or fix PATH.
) else (
  echo Server stopped normally.
)
echo ----------------------------------------
pause
