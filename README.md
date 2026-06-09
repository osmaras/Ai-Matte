
AI Matte Bridge (SCRATCH + MatAnyone2 + SAM2)

This project automates matte extraction in Assimilate SCRATCH using MatAnyone2, with an interactive SAM2 mask editor for precise subject selection.

Workflow
1. Read selected shot from current SCRATCH construct.
2. **Pass A** — Render clean plates (all layers disabled → raw source).
3. Collect rendered clean plate frames.
4. **Pass B** — Launch interactive SAM2 mask editor in the browser:
   - Click on subjects to segment with SAM2.
   - Multi-object support with per-object colour overlays.
   - Erode/dilate sliders for edge refinement.
   - Click **Save & Continue** to write `mask.png` and resume the pipeline.
5. Run MatAnyone2 inference on GPU using the saved mask.
6. Write matte sequence and add a shot note.

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

SAM2 Mask Editor
The editor launches automatically during Pass B. It opens a browser window with the first clean plate frame.

Controls:
- **Click** on the image to place positive segmentation points.
- **Negative mode** removes regions from the mask.
- **+ Add Object** for multi-subject shots (each gets its own colour).
- **Erode/Dilate sliders** refine mask edges.
- **Save & Continue** writes `mask.png` and resumes the pipeline.

SAM2 model: `facebook/sam2.1-hiera-large` (~2 GB, downloaded from HuggingFace on first run). Works on 8 GB VRAM at 1080p.

CLI Flags
| Flag | Description |
|------|-------------|
| `--skip-sam` | Reuse existing `mask.png` without opening editor |
| `--sam-port N` | Gradio port (default 7860) |
| `--no-browser` | Don't auto-open browser |

Run Manually
Use launcher (recommended because it sets CUDA wheel/index and cache locations):
launcher.bat -project <PRJ> -group <GRP> -construct <CON> -shot <SHT>
