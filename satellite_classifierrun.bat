@echo off

echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting server...
start "" cmd /c "timeout /t 6 >nul && start http://localhost:8000"

uvicorn app.main:app --reload --port 8000

pause