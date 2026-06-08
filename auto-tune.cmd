@echo off
setlocal
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\auto_tune.py" %*
exit /b %ERRORLEVEL%
