@echo off
setlocal
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\benchmark.py" %*
exit /b %ERRORLEVEL%
