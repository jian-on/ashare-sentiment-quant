@echo off
setlocal
title A-Quant Terminal - Stop
echo 正在停止 8501 端口上的终端服务...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    echo   结束进程 PID %%p
    taskkill /PID %%p /F >nul 2>&1
)
echo 完成。按任意键关闭。
pause >nul
