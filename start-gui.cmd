@echo off
setlocal
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\panel_gui.py" %*
exit /b %ERRORLEVEL%
