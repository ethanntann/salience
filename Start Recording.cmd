@echo off
rem Records the game window, then cuts and ranks highlights by your taste.
rem Stop recording with Ctrl+C in this window. Nothing touches the game process.
cd /d "%~dp0"
title Salience recorder
python -m pipeline_v2.capture.record %*
pause
