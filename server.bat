@echo off

REM Проверяем, существует ли виртуальная среда
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Активируем виртуальную среду
call venv\Scripts\activate.bat

REM Устанавливаем зависимости из requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

REM Запускаем скрипт run_server.py
echo Starting server...
python run_server.py

REM Деактивируем виртуальную среду
deactivate