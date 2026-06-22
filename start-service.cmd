@echo off
setlocal
set PANEL_INLINE_LOGS=1
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\model_juggler.py" --gateway %*
exit /b %ERRORLEVEL%
