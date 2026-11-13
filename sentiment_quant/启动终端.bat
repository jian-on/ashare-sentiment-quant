@echo off
setlocal
title A-Quant Terminal Launcher
cd /d "%~dp0"

netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto open

echo [1/3] 正在启动行情分析终端...
start "AQuant-Service" /min "C:\Users\27553\.workbuddy\binaries\python\envs\default\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

echo [2/3] 等待服务就绪（最多20秒）...
for /l %%i in (1,1,20) do (
    timeout /t 1 /nobreak >nul
    netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
    if not errorlevel 1 goto open
)
echo.
echo [错误] 启动超时。请查看最小化的 "AQuant-Service" 窗口里的报错信息。
pause
exit /b 1

:open
echo [3/3] 打开浏览器...
start "" http://127.0.0.1:8501
echo.
echo ============================================================
echo   终端已就绪:  http://127.0.0.1:8501
echo   关闭最小化的 AQuant-Service 窗口即可停止服务。
echo ============================================================
timeout /t 3 /nobreak >nul
exit /b 0
