@echo off
set "ORIG_DIR=%cd%"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

if "%1"=="restart" (
    echo Restarting Grok Swarm...
    docker compose restart
    echo Grok Swarm restarted successfully.
    goto :end
)

rem If launched from the grokswarm dir itself, no extra mount needed
if /i "%ORIG_DIR%"=="%SCRIPT_DIR%" (
    docker compose exec grokswarm python main.py %*
) else (
    rem Mount the caller's working directory as /project inside the container
    docker compose run --rm --no-deps -e GROKSWARM_HOST_DIR="%ORIG_DIR%" -v "%ORIG_DIR%:/project" grokswarm python main.py --project-dir /project %*
)
:end
