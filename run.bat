@echo off
REM ==============================================================================
REM VitalGuard AI — Remote Patient Monitoring System (Batch Launcher)
REM ==============================================================================
title VitalGuard AI System Launcher
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
