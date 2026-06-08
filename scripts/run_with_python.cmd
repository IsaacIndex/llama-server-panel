@echo off
setlocal

set "TARGET=%~1"
shift

if defined PYTHON_BIN goto use_custom

where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%TARGET%" %*
  exit /b %ERRORLEVEL%
)

python "%TARGET%" %*
exit /b %ERRORLEVEL%

:use_custom
"%PYTHON_BIN%" "%TARGET%" %*
exit /b %ERRORLEVEL%
