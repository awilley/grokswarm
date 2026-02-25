@echo off
set "ORIG_DIR=%cd%"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem Run natively — pass the caller's working directory as --project-dir
if /i "%ORIG_DIR%"=="%SCRIPT_DIR%" (
    python "%SCRIPT_DIR%\main.py" %*
) else (
    python "%SCRIPT_DIR%\main.py" --project-dir "%ORIG_DIR%" %*
)
