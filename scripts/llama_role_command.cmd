@echo off
setlocal
call "%~dp0run_with_python.cmd" "%~dp0llama_role_command.py" %*
exit /b %ERRORLEVEL%
