; Salience's installer, built by installer/build.py (it passes BuildDir, OutDir
; and AppVersion). Installs for this user only: no admin rights.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{7F1C3E52-9B7A-4C55-A0E3-5A1E1D2B9C41}
AppName=Salience
AppVersion={#AppVersion}
AppPublisher=Salience
DefaultDirName={localappdata}\Programs\Salience
DefaultGroupName=Salience
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutDir}
OutputBaseFilename=Salience-Setup-{#AppVersion}
SetupIconFile={#BuildDir}\salience.ico
UninstallDisplayIcon={app}\salience.ico
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041
CloseApplications=yes
WizardStyle=modern

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Salience"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m pipeline_v2.capture.desktop"; WorkingDir: "{app}\app"; IconFilename: "{app}\salience.ico"
Name: "{userdesktop}\Salience"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m pipeline_v2.capture.desktop"; WorkingDir: "{app}\app"; IconFilename: "{app}\salience.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Add a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m pipeline_v2.capture.desktop"; WorkingDir: "{app}\app"; Description: "Open Salience"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; Python leaves compiled files beside the app; recordings and settings are the
; player's (Videos\Salience, %APPDATA%\Salience) and stay.
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\python"
