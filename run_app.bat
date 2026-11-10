@echo off
title DermaScan AI Clinical Studio
echo Starting DermaScan AI PyTorch Server...
start "" "http://localhost:8000"
.venv\Scripts\python.exe server.py
pause
