@echo off
setlocal
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\model_juggler.py" %*
exit /b %ERRORLEVEL%
