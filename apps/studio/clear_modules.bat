@echo off

REM Get the directory of the current script
SET DIR=%~dp0

REM Change to that directory
cd /d %DIR%

REM Remove nodes
rmdir /s /q node_modules
rmdir /s /q .next
del /f bun.lockb