@echo off
rem Puts Salience in the tray: it records Fortnite whenever the game is open
rem and makes highlights when you quit it. No console window stays open.
cd /d "%~dp0"
start "" pythonw -m pipeline_v2.capture.app
