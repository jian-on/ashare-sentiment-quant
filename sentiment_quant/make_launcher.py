# -*- coding: utf-8 -*-
"""
make_launcher.py — 生成「双击即可运行」的终端启动按钮

背景：WorkBuddy 沙箱会在每轮对话结束时回收它启动的进程，导致终端反复"Connection error"。
      沙箱内无法让进程常驻（独立进程/WMI/计划任务/COM快捷方式均被拦截），
      因此改为生成 .bat 启动器：双击后由你自己的资源管理器/终端托管进程，
      不依赖本会话，只要窗口不关就一直在线。

生成物：
  1. sentiment_quant/启动终端.bat      项目内启动器（相对路径，可搬移整个文件夹）
  2. sentiment_quant/停止终端.bat      一键停止服务
  3. 桌面/A股量化终端.bat               桌面按钮（硬编码绝对路径）

编码说明：.bat 以 GBK 写入。中文 Windows 的 cmd 默认代码页 936，用 UTF-8 写会导致
          中文路径乱码（这是之前快捷方式失败的同类坑）。本脚本显式指定 encoding="gbk"。
"""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
PY_STREAMLIT = r"C:\Users\27553\.workbuddy\binaries\python\envs\default\Scripts\streamlit.exe"
PORT = 8501
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

PORT_CHECK = 'netstat -ano | findstr ":%d " | findstr "LISTENING" >nul 2>&1' % PORT

# ---------- 项目内启动器：用 %~dp0 相对定位，整个文件夹可搬移 ----------
START_LOCAL = r'''@echo off
setlocal
title A-Quant Terminal Launcher
cd /d "%~dp0"

%PORT_CHECK%
if not errorlevel 1 goto open

echo [1/3] 正在启动行情分析终端...
start "AQuant-Service" /min "%STREAMLIT%" run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false

echo [2/3] 等待服务就绪（最多20秒）...
for /l %%i in (1,1,20) do (
    timeout /t 1 /nobreak >nul
    %PORT_CHECK%
    if not errorlevel 1 goto open
)
echo.
echo [错误] 启动超时。请查看最小化的 "AQuant-Service" 窗口里的报错信息。
pause
exit /b 1

:open
echo [3/3] 打开浏览器...
start "" http://127.0.0.1:%PORT%
echo.
echo ============================================================
echo   终端已就绪:  http://127.0.0.1:%PORT%
echo   关闭最小化的 AQuant-Service 窗口即可停止服务。
echo ============================================================
timeout /t 3 /nobreak >nul
exit /b 0
'''.replace("%PORT_CHECK%", PORT_CHECK).replace("%STREAMLIT%", PY_STREAMLIT).replace("%PORT%", str(PORT))

# ---------- 桌面版：绝对路径（因为 %~dp0 会指向桌面） ----------
START_DESKTOP = START_LOCAL.replace(
    'cd /d "%~dp0"', 'cd /d "%s"' % ROOT
).replace("A-Quant Terminal Launcher", "A股题材情绪量化终端")

# ---------- 停止脚本 ----------
STOP = r'''@echo off
setlocal
title A-Quant Terminal - Stop
echo 正在停止 8501 端口上的终端服务...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo   结束进程 PID %%p
    taskkill /PID %%p /F >nul 2>&1
)
echo 完成。按任意键关闭。
pause >nul
'''.replace("%PORT%", str(PORT))


def write_gbk(path: str, text: str):
    with open(path, "w", encoding="gbk", newline="\r\n") as fh:
        fh.write(text)
    print(f"  生成 {path}  ({os.path.getsize(path)} bytes, GBK)")


if __name__ == "__main__":
    print("生成启动器：")
    write_gbk(os.path.join(ROOT, "启动终端.bat"), START_LOCAL)
    write_gbk(os.path.join(ROOT, "停止终端.bat"), STOP)

    if os.path.isdir(DESKTOP):
        desk = os.path.join(DESKTOP, "A股量化终端.bat")
        try:
            write_gbk(desk, START_DESKTOP)
            print(f"  桌面按钮已生成：{desk}")
        except Exception as e:
            print(f"  [桌面生成失败，仅项目内可用] {type(e).__name__}: {e}")
    else:
        print(f"  未找到桌面目录 {DESKTOP}，跳过桌面版")
