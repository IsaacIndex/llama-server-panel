@echo off
setlocal
call "%~dp0scripts\run_with_python.cmd" "%~dp0scripts\llama_role_command.py" exec embed --auto-tune %*
exit /b %ERRORLEVEL%
