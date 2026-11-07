# Installer

`python installer/build.py` makes `dist/Salience-Setup-<version>.exe` (about
205 MB): the official embeddable Python with only the libraries the app
imports, the app itself, and FFmpeg (copied from the one on PATH, with its
licence). It installs for the current user into
`%LOCALAPPDATA%\Programs\Salience`, needs no admin rights, adds a Start menu
entry, and has an uninstaller. Recordings and settings
(`Videos\Salience`, `%APPDATA%\Salience`) are the player's and are never
removed.

Needs Inno Setup 6 (`winget install JRSoftware.InnoSetup`) and FFmpeg 8+.

The installer is not code-signed, so Windows SmartScreen shows "Windows
protected your PC" the first time: testers click **More info → Run anyway**.
