@echo off
rem Opens Salience: your highlights, rating and settings. Its Auto-clip switch
rem runs the recorder in the background, which keeps going when the window closes.
cd /d "%~dp0"
start "" pythonw -m pipeline_v2.capture.desktop
