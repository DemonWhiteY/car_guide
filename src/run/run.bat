@echo off

mkdir voice_files

echo 正在启动 main.py...
start "" python main.py
timeout /t 5 >nul

echo 正在启动 face.py...
start "" python face.py
timeout /t 5 >nul

echo 正在启动 gesture_recognization.py...
start "" python gesture_recognization.py
timeout /t 5 >nul

echo 正在启动 voice.py...
start "" python voice.py

echo 所有脚本已启动。
pause