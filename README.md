# 🎮 WoW AI Complete - Leveling & Endgame Bot (All Classes)

> English first — French version below.

## 🇬🇧 English Overview

**WoW AI Complete** is a multi-class AI bot for World of Warcraft, combining:

- Autonomous leveling (quests, combat, loot, basic inventory management)
- Character configuration: faction, race, class, spec, role (Tank/Heal/DPS), professions, gear priority
- Experimental endgame AI (PvE / PvP / Arena) using screen vision (OpenCV/PIL) + keyboard/mouse control (pynput)
- Architecture ready for Deep Reinforcement Learning (Dueling DQN, replay buffer, GPU/CPU via PyTorch)

This file is a **short GitHub-friendly README**. For full details (UI tabs, troubleshooting, roadmap, etc.), see:

- `readme_wow_ai.md`

### 🚀 Quick Install

```bash
pip install -r requirements.txt
python wow_ai_complete_fixed.py
```

On Windows, you can also use the helper scripts:

- `install_dependencies.bat` → install Python dependencies (pynput, pillow, opencv-python, torch, etc.)
- `launch_wow_ai.bat` → launch the main UI

### 🤝 Contributing

- This repository is **public and open to contributions**
- Issues and Pull Requests are welcome (better rotations, more classes/specs, real RL training loops, OCR integration, UI/UX improvements, etc.)
- See `CONTRIBUTING.md` for more detailed guidelines

### ⚠️ Warning / Disclaimer

- Using bots is generally **against the Terms of Service** of online games (including Blizzard games)
- This project is provided for **educational purposes only** (computer vision, reinforcement learning, automation)
- You are fully responsible for how you use this code

---

## 🇫🇷 Vue d’ensemble (Français)

**WoW AI Complete** est une IA complète multi-classes pour World of Warcraft, combinant leveling autonome, IA endgame expérimentale et vision par ordinateur (OpenCV) avec PyTorch.

Ce fichier est une version courte, prête pour GitHub. Pour la documentation détaillée, voir `readme_wow_ai.md`.

### ✨ Aperçu rapide

- Leveling autonome (quêtes, combat, loot, gestion d'inventaire)
- Configuration personnage : faction, race, classe, spé, rôle (Tank/Heal/DPS), métiers, priorité d'équipement
- IA Endgame expérimentale (PvE / PvP / Arena) basée sur la vision écran + contrôle clavier/souris
- Architecture prête pour le Deep Reinforcement Learning (Dueling DQN, replay buffer, GPU/CPU)

### 🚀 Installation rapide

```bash
pip install -r requirements.txt
python wow_ai_complete_fixed.py
```

ou utilisez les scripts Windows :

- `install_dependencies.bat` pour installer automatiquement les dépendances
- `launch_wow_ai.bat` pour lancer l'application

### 🤝 Contribution

- Dépôt prévu pour être **public** et modifiable par tout le monde
- Issues et Pull Requests bienvenues (ajout de classes, meilleures rotations, vrai RL, intégration OCR, améliorations UI/UX, etc.)

### ⚠️ Avertissement

- L'utilisation de bots est généralement contraire aux CGU des jeux en ligne
- Ce projet est fourni à des fins **éducatives** uniquement (vision par ordinateur, RL, automatisation)
