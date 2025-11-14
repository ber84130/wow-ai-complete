# Contributing to WoW AI Complete

Thanks for your interest in improving **WoW AI Complete - Leveling & Endgame Bot (All Classes)**.

This project is meant as an **educational** AI/automation playground for World of Warcraft:
- Computer vision (OpenCV + screenshots)
- Control automation (pynput)
- Deep Learning / RL experiments (PyTorch)

> **Warning:** Using bots is generally against Blizzard's Terms of Service. This project is for learning and experimentation only.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ber84130/wow-ai-complete.git
cd wow-ai-complete
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# or CMD
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If installation fails for Torch or TorchVision, check the official PyTorch website for commands adapted to your GPU/CPU.

### 4. Run the bot

```bash
python wow_ai_complete_fixed.py
```

Or on Windows, double-click:

- `install_dependencies.bat` (first time)
- `launch_wow_ai.bat` (to run the UI)

---

## Project Overview

Main files:

- `wow_ai_complete_fixed.py`
  - Tkinter UI with tabs:
    - **Configuration** (PC specs, modules)
    - **Agent Leveling** (autonomous leveling bot)
    - **IA Endgame** (experimental endgame AI)
    - **Logs** (runtime logs)
  - `LevelingAgent`:
    - Vision (screen capture + AdvancedVisionSystem)
    - Keyboard/mouse control (pynput)
    - Questing, combat, looting, basic inventory handling
  - `EndgameAgent`:
    - Modes: PvE, PvP, Arena
    - Farm modes: Quests, Professions/Resources (placeholders ready for improvements)
  - `CharacterConfig`:
    - faction, race, class, spec, expansion, professions
    - compatibility logic for race/class/expansion

- `install_dependencies.bat`, `launch_wow_ai.bat`:
  - Windows helpers for installing and running the project.

- `readme_wow_ai.md`:
  - Detailed documentation in French.

- `README.md`:
  - Short GitHub-friendly overview.

---

## How to Contribute

### 1. Issues

- Use **GitHub Issues** to report:
  - Bugs (with logs and steps to reproduce)
  - Feature requests (clear description + possible UI/logic)
  - Ideas for AI improvements (RL, vision, rotations, pathfinding, etc.)

When reporting a bug, please include:

- OS version
- Python version
- GPU/CPU type
- Steps to reproduce
- Relevant log output (from the **Logs** tab or terminal)

### 2. Pull Requests (PRs)

Recommended workflow:

1. **Fork** the repository on GitHub.
2. Create a **feature branch**:
   ```bash
   git checkout -b feature/better-endgame-quests
   ```
3. Make your changes:
   - Keep UI consistent with existing style (Tkinter + tabs)
   - Avoid breaking the current behavior
   - Add logs where behavior changes significantly
4. Run basic tests:
   - `python check_installation.py`
   - `python wow_ai_complete_fixed.py` and ensure the UI starts without errors
5. Commit with a clear message:
   ```bash
   git commit -m "Improve endgame quest farming behavior"
   ```
6. Push your branch and open a Pull Request:
   - Describe what you changed
   - Mention any breaking changes or new dependencies

### 3. Areas Where Contributions Are Welcome

Some ideas (not exhaustive):

- **Pathfinding / Navigation**
  - Waypoint-based paths for questing or profession farming
  - Simple path-following logic that uses movement keys

- **Vision & OCR**
  - Better detection of quest objectives, resources, enemies
  - Integration with OCR (e.g. Tesseract) to read tooltips and quest texts

- **Rotations & Roles**
  - Class/spec-specific combat rotations (Tank / Heal / DPS)
  - Configurable profiles for each class/spec

- **Talents & Professions**
  - Real talent allocation per spec
  - Real logic for leveling professions (nodes detection, routes)

- **RL / Deep Learning**
  - Use `DuelingDQN` and `replay_buffer` for actual RL training
  - Scripts to train models offline using recorded gameplay

- **Better UI/UX**
  - More controls in the UI for tuning behaviors
  - Profile saving/loading (per character)

---

## Code Style & Guidelines

- Python 3.10+ compatible code (project currently tested on 3.12).
- Keep imports at the top of `wow_ai_complete_fixed.py`.
- Avoid hardcoding user-specific paths.
- Use logging via `log_message` instead of `print` when possible (to keep the UI informative).
- Comments and docstrings are welcome, but keep them concise.

---

## Legal & Ethical Note

This project is intended for **learning purposes**. Any real use in online games may violate the game's Terms of Service and can lead to account sanctions. Use responsibly and at your own risk.

Thanks again for contributing and helping improve WoW AI Complete!
