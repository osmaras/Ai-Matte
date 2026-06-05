
AI Matte Bridge (SCRATCH + MatAnyone2)

This project automates matte extraction in Assimilate SCRATCH using MatAnyone2.

Workflow implemented by the bridge:
1. Read selected shot from current SCRATCH construct.
2. Render clean plates from output node.
3. Render one-frame mask from mask layer.
4. Run MatAnyone2 inference on GPU (CUDA required if configured).
5. Write matte sequence and add a shot note.

Requirements
1. Windows
2. Assimilate SCRATCH with REST API enabled
3. uv installed
4. NVIDIA GPU drivers installed for CUDA execution

Install uv (Windows PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

If uv is not on PATH, use:
set Path=C:\Users\<user>\.local\bin;%Path%

Project Setup (Recommended)
1. Open PowerShell in this project folder.
2. Run installer:
    powershell -ExecutionPolicy Bypass -File .\install.ps1
3. Answer prompts:
    - Project folder
    - uv.exe path
    - Cache folder location
    - Require CUDA yes/no
4. Installer writes launcher.bat for this machine.

SCRATCH Setup
1. Enable SCRATCH REST API auto-start.
2. Add custom command:
    - Command: full path to launcher.bat
    - Arguments: -project %PRJ -group %GRP -construct %CON -shot %SHT
3. Keep Wait till finished disabled if you do not want SCRATCH UI blocked.

Mask Layer
1. Preferred layer names include matte or MatAnyone_Mask.
2. If naming differs, adjust script args --mask-layer-name.

Run Manually
Use launcher (recommended because it sets CUDA wheel/index and cache locations):
launcher.bat -project <PRJ> -group <GRP> -construct <CON> -shot <SHT>
