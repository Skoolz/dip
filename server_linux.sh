#!/bin/bash

# Проверяем, существует ли виртуальная среда
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Активируем виртуальную среду
source venv/bin/activate

# Устанавливаем зависимости из requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Запускаем скрипт run_server.py
echo "Starting server..."
python run_server.py

# Деактивируем виртуальную среду
deactivate
