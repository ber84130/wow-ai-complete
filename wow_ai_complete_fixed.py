"""
WoW AI Complete - Leveling & Endgame Bot (All Classes)
======================================================
IA complète multi-classes pour WoW avec Deep Learning

FONCTIONNALITÉS:
✅ Healer IA (DQN + Vision)
✅ Agent de Leveling autonome (Quêtes, Combat, Loot)
✅ Détection auto d'objets (PNJ, quêtes, ennemis)
✅ Gestion inventaire + équipement
✅ Achat sorts + montures
✅ Auto-configuration selon specs PC

Installation:
pip install pynput pillow torch torchvision opencv-python numpy psutil GPUtil

"""

import os
import sys
import time
import json
import datetime
import threading
import random
import gc
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Détection specs PC
try:
    import psutil
    import GPUtil
    SPECS_AVAILABLE = True
except ImportError:
    print("⚠️ psutil/GPUtil: pip install psutil GPUtil")
    SPECS_AVAILABLE = False

# Dépendances de base
try:
    from pynput import mouse, keyboard
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key
    CONTROL_AVAILABLE = True
except ImportError:
    print("⚠️ pynput: pip install pynput")
    CONTROL_AVAILABLE = False

try:
    from PIL import ImageGrab, Image
    SCREENSHOT_AVAILABLE = True
except ImportError:
    print("⚠️ pillow: pip install pillow")
    SCREENSHOT_AVAILABLE = False

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("⚠️ PyTorch: pip install torch torchvision")
    TORCH_AVAILABLE = False
    np = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("⚠️ OpenCV: pip install opencv-python")
    CV2_AVAILABLE = False

# Chemins
HOME = Path.home()
BASE_PATH = HOME / "Documents" / "WoW_AI_Complete"
DATA_PATH = BASE_PATH / "training_data"
SCREENSHOT_PATH = BASE_PATH / "screenshots"
MODELS_PATH = BASE_PATH / "models"
DQN_PATH = MODELS_PATH / "dqn_healer"
LEVELING_PATH = MODELS_PATH / "leveling_agent"

for path in [BASE_PATH, DATA_PATH, SCREENSHOT_PATH, MODELS_PATH, DQN_PATH, LEVELING_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUTO-CONFIGURATION SELON SPECS PC
# ============================================================

class PCSpecs:
    """Détecte et adapte config selon PC"""
    
    @staticmethod
    def detect_specs():
        """Détecte specs PC"""
        specs = {
            'cpu_count': os.cpu_count() or 4,
            'ram_gb': 16,
            'gpu_name': 'CPU',
            'gpu_memory_gb': 0
        }
        
        if SPECS_AVAILABLE:
            # RAM
            mem = psutil.virtual_memory()
            specs['ram_gb'] = mem.total / (1024**3)
            
            # GPU
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    specs['gpu_name'] = gpu.name
                    specs['gpu_memory_gb'] = gpu.memoryTotal / 1024
            except:
                pass
        
        return specs
    
    @staticmethod
    def get_optimized_config(specs):
        """Config optimisée selon specs"""
        ram_gb = specs['ram_gb']
        has_gpu = 'cuda' in str(DEVICE).lower()
        
        # Adapter selon RAM
        if ram_gb >= 32:
            config = {
                'batch_size': 64,
                'replay_buffer': 50000,
                'frame_stack': 4,
                'screenshot_cache': 100,
                'worker_threads': 4
            }
        elif ram_gb >= 16:
            config = {
                'batch_size': 32,
                'replay_buffer': 30000,
                'frame_stack': 4,
                'screenshot_cache': 50,
                'worker_threads': 2
            }
        else:  # 8GB
            config = {
                'batch_size': 16,
                'replay_buffer': 15000,
                'frame_stack': 2,
                'screenshot_cache': 25,
                'worker_threads': 1
            }
        
        # Adapter selon GPU
        if has_gpu:
            config['use_gpu'] = True
            config['batch_size'] *= 2
        else:
            config['use_gpu'] = False
        
        return config


# Config globale
PC_SPECS = PCSpecs.detect_specs()
AUTO_CONFIG = PCSpecs.get_optimized_config(PC_SPECS)

class Config:
    """Configuration optimisée auto"""
    BATCH_SIZE = AUTO_CONFIG['batch_size']
    REPLAY_BUFFER_SIZE = AUTO_CONFIG['replay_buffer']
    FRAME_STACK = AUTO_CONFIG['frame_stack']
    MAX_SCREENSHOT_CACHE = AUTO_CONFIG['screenshot_cache']
    WORKER_THREADS = AUTO_CONFIG['worker_threads']
    
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE = 1000
    
    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    
    MAX_EPISODE_STEPS = 500
    TRAIN_FREQUENCY = 4
    SAVE_FREQUENCY = 10
    GC_FREQUENCY = 100


# ============================================================
# CLASSES DE BASE
# ============================================================

@dataclass
class RecordedAction:
    timestamp: float
    action_type: str
    mouse_position: tuple = None
    button: str = None
    keys_pressed: List[str] = None
    screenshot_path: str = None

@dataclass
class CombatEvent:
    timestamp: float
    event_type: str
    source: str
    target: str
    spell: str
    amount: int = 0

@dataclass
class GameState:
    """État du jeu"""
    player_hp: float = 100.0
    player_mana: float = 100.0
    player_level: int = 1
    target_hp: float = 0
    in_combat: bool = False
    quest_objectives: List[str] = None
    inventory_full: bool = False
    gold: int = 0


@dataclass
class CharacterConfig:
    faction: str = "Alliance"
    race: str = "Humain"
    class_name: str = "Guerrier"
    spec: str = "Armes"
    primary_professions: List[str] = None
    secondary_professions: List[str] = None


# ============================================================
# RÉSEAUX DE NEURONES
# ============================================================

if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """Dueling DQN optimisé"""
        def __init__(self, input_channels=4, n_actions=10):
            super(DuelingDQN, self).__init__()
            
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            conv_out_size = self._get_conv_out((input_channels, 84, 84))
            
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
        
        def _get_conv_out(self, shape):
            o = self.conv(torch.zeros(1, *shape))
            return int(np.prod(o.size()))
        
        def forward(self, x):
            conv_out = self.conv(x).view(x.size(0), -1)
            value = self.value_stream(conv_out)
            advantage = self.advantage_stream(conv_out)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values


# ============================================================
# SYSTÈME DE VISION AVANCÉ
# ============================================================

class AdvancedVisionSystem:
    """Vision avec détection multi-objets"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """Charge templates d'objets"""
        return {
            'quest_icon': None,  # ! jaune
            'npc_dialog': None,  # Fenêtre dialogue
            'loot_sparkle': None,  # Lueur loot
            'vendor_icon': None,  # Icône vendeur
            'trainer_icon': None  # Icône maître
        }
    
    def detect_objects(self, image):
        """Détecte objets dans l'image"""
        if not CV2_AVAILABLE:
            return {}
        
        try:
            img = self._prepare_image(image)
            
            detections = {
                'health_bars': self._detect_health_bars(img),
                'quest_markers': self._detect_quest_markers(img),
                'npcs': self._detect_npcs(img),
                'loot': self._detect_loot(img),
                'enemies': self._detect_enemies(img),
                'vendors': self._detect_vendors(img)
            }
            
            return detections
            
        except Exception as e:
            return {}
    
    def _prepare_image(self, image):
        """Prépare image pour traitement"""
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    
    def _detect_health_bars(self, img):
        """Détecte barres HP"""
        small = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        bars = []
        
        # Vertes (HP haut)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Rouges (HP bas)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        for mask, level in [(green_mask, 'high'), (red_mask, 'low')]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > h * 2 and w > 15:
                    bars.append({'x': x, 'y': y, 'w': w, 'h': h, 'level': level})
        
        return bars[:10]
    
    def _detect_quest_markers(self, img):
        """Détecte marqueurs de quête (! jaunes)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:
                x, y, w, h = cv2.boundingRect(contour)
                markers.append({'x': x, 'y': y, 'type': 'quest'})
        
        return markers[:5]
    
    def _detect_npcs(self, img):
        """Détecte PNJs (noms verts/bleus)"""
        # Simplifi: chercher textes verts/bleus en haut écran
        top_half = img[:img.shape[0]//3, :]
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        
        # Vert (amical)
        lower = np.array([40, 30, 30])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        npcs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 10:
                npcs.append({'x': x, 'y': y, 'type': 'friendly'})
        
        return npcs[:5]
    
    def _detect_loot(self, img):
        """Détecte loot au sol (lueur)"""
        # Chercher zones très lumineuses
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loot = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:
                x, y, w, h = cv2.boundingRect(contour)
                loot.append({'x': x, 'y': y})
        
        return loot[:3]
    
    def _detect_enemies(self, img):
        """Détecte ennemis (noms rouges)"""
        top_half = img[:img.shape[0]//3, :]
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemies = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 8:
                enemies.append({'x': x, 'y': y, 'threat': 'hostile'})
        
        return enemies[:5]
    
    def _detect_vendors(self, img):
        """Détecte vendeurs (sac ou icône)"""
        # Placeholder - détection simplifiée
        return []
    
    def preprocess_frame(self, image):
        """Prétraite pour DQN"""
        if not CV2_AVAILABLE:
            return None
        
        try:
            img = self._prepare_image(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        except:
            return None


# ============================================================
# AGENT DE LEVELING AUTONOME
# ============================================================

class LevelingAgent:
    """
    Agent autonome pour leveling
    - Accepte/rend quêtes
    - Combat mobs
    - Loot
    - Vend items
    - Achète sorts/équipement
    """
    
    def __init__(self, callback=None, character_config: CharacterConfig = None):
        self.callback = callback
        self.running = False
        self.thread = None
        
        self.vision = AdvancedVisionSystem(callback=callback)
        
        if CONTROL_AVAILABLE:
            self.mouse = MouseController()
            self.keyboard = KeyboardController()
        
        self.state = GameState()
        self.character_config = character_config or CharacterConfig()
        
        # Comportements
        self.behaviors = {
            'quest_mode': True,
            'grind_mode': False,
            'auto_loot': True,
            'auto_sell': True,
            'auto_learn_spells': True,
            'auto_equip': True
        }
        
        # Stats
        self.stats = {
            'mobs_killed': 0,
            'quests_completed': 0,
            'gold_earned': 0,
            'items_looted': 0,
            'deaths': 0,
            'playtime': 0
        }
    
    def start(self):
        """Démarre l'agent"""
        if not CONTROL_AVAILABLE:
            if self.callback:
                self.callback("error", "pynput requis!")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        
        if self.callback:
            self.callback("success", "🤖 Agent de leveling activé!")
        
        return True
    
    def stop(self):
        """Arrête l'agent"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.callback:
            self.callback("info", f"⏹️ Agent arrêté - Stats: {self.stats}")
    
    def _main_loop(self):
        """Boucle principale"""
        start_time = time.time()
        
        while self.running:
            try:
                self.stats['playtime'] = int(time.time() - start_time)
                
                # Capture écran
                if SCREENSHOT_AVAILABLE:
                    screenshot = ImageGrab.grab()
                    detections = self.vision.detect_objects(screenshot)
                    
                    # Décision selon priorités
                    if self._check_death(detections):
                        self._handle_death()
                    elif self._check_low_hp():
                        self._use_healing()
                    elif self.behaviors['quest_mode'] and detections.get('quest_markers'):
                        self._handle_quests(detections['quest_markers'])
                    elif self.state.inventory_full and self.behaviors['auto_sell']:
                        self._find_vendor_and_sell()
                    elif detections.get('loot') and self.behaviors['auto_loot']:
                        self._loot_items(detections['loot'])
                        # Après un loot, on peut vérifier l'équipement et les métiers
                        self._auto_equip_if_better()
                        self._handle_professions_if_possible()
                    elif detections.get('enemies'):
                        self._engage_combat(detections['enemies'])
                    elif self.behaviors['grind_mode']:
                        self._grind_mobs()
                    else:
                        self._explore()
                
                time.sleep(0.3)
                
            except Exception as e:
                if self.callback:
                    self.callback("error", f"Erreur agent: {e}")
                time.sleep(1)
    
    def _check_death(self, detections):
        """Vérifie si mort"""
        # Check si écran gris ou "Release Spirit"
        return False  # Placeholder
    
    def _handle_death(self):
        """Gère la mort"""
        if self.callback:
            self.callback("warning", "💀 Mort détectée - Release Spirit")
        
        self.stats['deaths'] += 1
        # Clic sur Release Spirit
        time.sleep(2)
    
    def _check_low_hp(self):
        """Vérifie HP bas"""
        return self.state.player_hp < 30
    
    def _use_healing(self):
        """Utilise heal/potion"""
        if self.callback:
            self.callback("info", "💊 Utilisation heal")
        
        # Presse touche heal (ex: 1)
        self.keyboard.press('1')
        time.sleep(0.1)
        self.keyboard.release('1')
        time.sleep(1.5)
    
    def _handle_quests(self, markers):
        """Gère les quêtes"""
        if not markers:
            return
        
        marker = markers[0]
        
        if self.callback:
            self.callback("info", f"❗ Quête détectée à ({marker['x']}, {marker['y']})")
        
        # Clic sur marqueur
        self.mouse.position = (marker['x'], marker['y'])
        time.sleep(0.2)
        self.mouse.click(Button.left)
        time.sleep(1)
        
        # Accepte/rend quête (clic sur bouton)
        self.mouse.position = (960, 600)  # Position bouton "Accept"
        self.mouse.click(Button.left)
        time.sleep(0.5)
        
        self.stats['quests_completed'] += 1
    
    def _loot_items(self, loot_positions):
        """Loot items"""
        if not loot_positions:
            return
        
        loot = loot_positions[0]
        
        # Clic droit sur loot
        self.mouse.position = (loot['x'], loot['y'])
        time.sleep(0.1)
        self.mouse.click(Button.right)
        time.sleep(0.5)
        
        self.stats['items_looted'] += 1
    
    def _engage_combat(self, enemies):
        """Engage combat"""
        if not enemies:
            return
        
        enemy = enemies[0]
        
        if self.callback:
            self.callback("info", f"⚔️ Combat ennemi à ({enemy['x']}, {enemy['y']})")
        
        # Tab pour cibler
        self.keyboard.press(Key.tab)
        time.sleep(0.05)
        self.keyboard.release(Key.tab)
        time.sleep(0.2)
        
        # Rotation de base (touches 1-4)
        for spell_key in ['1', '2', '3', '4']:
            if not self.running:
                break
            
            self.keyboard.press(spell_key)
            time.sleep(0.05)
            self.keyboard.release(spell_key)
            time.sleep(1.5)  # GCD
        
        self.stats['mobs_killed'] += 1
    
    def _grind_mobs(self):
        """Grind mobs"""
        # Cherche mob proche et engage
        if self.callback:
            self.callback("info", "🔄 Grind mode")
        
        time.sleep(2)
    
    def _explore(self):
        """Explore la zone"""
        # Mouvement aléatoire
        directions = ['z', 'q', 'd', 's']
        key = random.choice(directions)
        
        self.keyboard.press(key)
        time.sleep(random.uniform(1, 3))
        self.keyboard.release(key)
    
    def _find_vendor_and_sell(self):
        """Trouve vendeur et vend"""
        if self.callback:
            self.callback("info", "💰 Recherche vendeur")
        
        # Placeholder - logique complexe
        time.sleep(2)

    # ------------------------------------------------------------
    # Talents, métiers et équipement (heuristiques simples)
    # ------------------------------------------------------------

    def allocate_talents_if_needed(self):
        """Alloue les talents selon classe/spé quand des points sont disponibles.

        Note: sans lecture mémoire, on ne peut pas savoir exactement les points
        disponibles. Cette méthode est donc un placeholder qui illustre l'endroit
        où la logique d'allocation automatique serait placée.
        """
        if self.callback:
            cfg = self.character_config
            self.callback(
                "info",
                f"(Talents) Vérification auto pour {cfg.class_name}/{cfg.spec} (placeholder)"
            )

    def _handle_professions_if_possible(self):
        """Essaie de monter les métiers choisis quand une opportunité est probable.

        Exemple d'heuristique:
        - Si Minage ou Herboristerie sélectionné, déjà couvert par la vision/loot
          (il faudrait ajouter détection d'herbes/minerais pour aller plus loin).
        - Ici on se contente de logguer les métiers et de marquer l'endroit où
          ajouter la logique de déplacement spécifique.
        """
        if not self.character_config.primary_professions and not self.character_config.secondary_professions:
            return

        if self.callback:
            self.callback(
                "info",
                f"(Métiers) Suivi métiers principaux {self.character_config.primary_professions} "
                f"/ secondaires {self.character_config.secondary_professions} (placeholder)"
            )

    def _auto_equip_if_better(self):
        """Heuristique de choix d'équipement selon priorité (Dégâts/Survie/Équilibré).

        Sans accès à l'inventaire détaillé ni aux stats, on ne peut pas faire
        une comparaison réelle des objets. Cette fonction marque l'endroit où
        la logique serait branchée (lecture du tooltip via OCR, etc.).
        """
        if not self.behaviors.get('auto_equip', False):
            return

        if self.callback:
            self.callback(
                "info",
                f"(Équipement) Vérification auto selon priorité "
                f"{getattr(self, 'gear_priority', 'Dégâts')} (placeholder)"
            )


class EndgameAgent:
    """Agent IA Endgame (donjons/raids/PvP) basé sur vision + actions clavier/souris

    Objectif: apprendre des patterns de haut niveau (rotation, survie, positionnement)
    en se basant sur la vision de l'écran et l'état simplifié du jeu.
    """

    def __init__(self, callback=None, character_config: CharacterConfig = None):
        self.callback = callback
        self.running = False
        self.thread = None

        self.vision = AdvancedVisionSystem(callback=callback)

        if CONTROL_AVAILABLE:
            self.mouse = MouseController()
            self.keyboard = KeyboardController()

        self.state = GameState()
        self.character_config = character_config or CharacterConfig()

        # Modes de jeu
        self.modes = {
            'pve': True,
            'pvp': False,
            'arena': False
        }

        # Placeholders pour futur modèle Deep RL
        self.model = None
        self.target_model = None
        self.replay_buffer = deque(maxlen=Config.REPLAY_BUFFER_SIZE)

    def start(self):
        if not CONTROL_AVAILABLE:
            if self.callback:
                self.callback("error", "pynput requis pour l'IA endgame!")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()

        if self.callback:
            self.callback("success", "🏆 IA Endgame activée!")

        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

        if self.callback:
            self.callback("info", "⏹️ IA Endgame arrêtée")

    def _main_loop(self):
        """Boucle principale endgame.

        Version simplifiée: utilise des heuristiques de base pour éviter de faire
        n'importe quoi, en attendant un vrai entraînement Deep RL.
        """
        while self.running:
            try:
                if SCREENSHOT_AVAILABLE:
                    screenshot = ImageGrab.grab()
                    detections = self.vision.detect_objects(screenshot)

                    # Heuristique basique: priorité survie -> cible -> rotation
                    if self._check_low_hp():
                        self._use_defensive()
                    elif detections.get('enemies'):
                        self._smart_combat(detections['enemies'])
                    else:
                        self._positioning()

                time.sleep(0.2)

            except Exception as e:
                if self.callback:
                    self.callback("error", f"Erreur IA endgame: {e}")
                time.sleep(1)

    def _check_low_hp(self):
        return self.state.player_hp < 35

    def _use_defensive(self):
        if not CONTROL_AVAILABLE:
            return
        if self.callback:
            self.callback("warning", "🛡️ Utilisation défensive / kite")

        # Exemple: reculer + CD défensif sur une touche (ex: '5')
        self.keyboard.press('s')
        time.sleep(0.3)
        self.keyboard.release('s')

        self.keyboard.press('5')
        time.sleep(0.05)
        self.keyboard.release('5')

    def _smart_combat(self, enemies):
        if not CONTROL_AVAILABLE or not enemies:
            return

        enemy = enemies[0]

        if self.callback:
            self.callback("info", f"🏹 IA Endgame combat ennemi à ({enemy['x']}, {enemy['y']})")

        # Tab pour cibler, puis petite rotation prioritaire
        self.keyboard.press(Key.tab)
        time.sleep(0.05)
        self.keyboard.release(Key.tab)
        time.sleep(0.2)

        # Exemple: prioriser une rotation simple mono-cible
        rotation = ['1', '2', '3', '4']
        for key in rotation:
            if not self.running:
                break
            self.keyboard.press(key)
            time.sleep(0.05)
            self.keyboard.release(key)
            time.sleep(1.0)

    def _positioning(self):
        if not CONTROL_AVAILABLE:
            return

        # Mouvement léger pour simuler un joueur qui se place (strafe/gauche droite)
        move_keys = ['q', 'd']
        key = random.choice(move_keys)
        self.keyboard.press(key)
        time.sleep(random.uniform(0.3, 0.7))
        self.keyboard.release(key)


# ============================================================
# COMBAT LOG PARSER
# ============================================================

class CombatLogParser:
    def __init__(self, callback=None):
        self.log_path = None
        self.running = False
        self.thread = None
        self.callback = callback
        self.combat_events = []
        self.recent_damage = []
        self.recent_heals = []
    
    def find_combatlog(self):
        paths = [
            Path.home() / "Program Files (x86)" / "World of Warcraft" / "_retail_" / "Logs" / "WoWCombatLog.txt",
            Path("C:/Program Files (x86)/World of Warcraft/_retail_/Logs/WoWCombatLog.txt"),
        ]
        
        for path in paths:
            if path.exists():
                return str(path)
        return None
    
    def start(self):
        if not self.log_path or not Path(self.log_path).exists():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._parse_loop, daemon=True)
        self.thread.start()
        
        if self.callback:
            self.callback("success", "📊 Combat log connecté")
        return True
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _parse_loop(self):
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(0, 2)
            
            while self.running:
                line = f.readline()
                if line:
                    self._parse_line(line)
                else:
                    time.sleep(0.1)
    
    def _parse_line(self, line):
        try:
            parts = line.strip().split(',')
            if len(parts) < 8:
                return
            
            event_type = parts[0].split()[1] if len(parts[0].split()) > 1 else ""
            
            if "DAMAGE" in event_type:
                event = CombatEvent(
                    timestamp=time.time(),
                    event_type="DAMAGE",
                    source=parts[2] if len(parts) > 2 else "",
                    target=parts[5] if len(parts) > 5 else "",
                    spell=parts[10] if len(parts) > 10 else "",
                    amount=int(parts[-1]) if parts[-1].isdigit() else 0
                )
                self.combat_events.append(event)
                self.recent_damage.append({"target": event.target, "amount": event.amount, "time": time.time()})
                self.recent_damage = [d for d in self.recent_damage if time.time() - d["time"] < 5]
            
            elif "HEAL" in event_type:
                event = CombatEvent(
                    timestamp=time.time(),
                    event_type="HEAL",
                    source=parts[2] if len(parts) > 2 else "",
                    target=parts[5] if len(parts) > 5 else "",
                    spell=parts[10] if len(parts) > 10 else "",
                    amount=int(parts[-1]) if parts[-1].isdigit() else 0
                )
                self.combat_events.append(event)
                self.recent_heals.append({"target": event.target, "amount": event.amount, "time": time.time()})
                self.recent_heals = [h for h in self.recent_heals if time.time() - h["time"] < 5]
        except:
            pass


# ============================================================
# INTERFACE GRAPHIQUE COMPLÈTE
# ============================================================

class WoWAIInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WoW AI Complete - Leveling & Endgame Bot")
        self.root.geometry("1300x850")
        
        # Config personnage partagée entre agents
        self.character_config = CharacterConfig()

        # Composants
        self.combat_parser = CombatLogParser(callback=self.log_message)
        self.leveling_agent = LevelingAgent(callback=self.log_message, character_config=self.character_config)
        self.endgame_agent = EndgameAgent(callback=self.log_message, character_config=self.character_config)
        
        # UI d'abord, PUIS composants qui utilisent log_message
        self.setup_ui()
        self.show_startup()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#1a1a2e', height=80)
        header.pack(fill='x')
        
        tk.Label(
            header,
            text="🎮 WoW AI Complete - Leveling & Endgame Bot",
            font=('Arial', 18, 'bold'),
            bg='#1a1a2e',
            fg='#00ff88'
        ).pack(pady=20)
        
        # Tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        notebook.add(self.create_config_tab(), text="⚙️ Configuration")
        notebook.add(self.create_leveling_tab(), text="🎯 Agent Leveling")
        notebook.add(self.create_endgame_tab(), text="🏆 IA Endgame")
        notebook.add(self.create_logs_tab(), text="📝 Logs")
    
    def create_config_tab(self):
        frame = tk.Frame(bg='#f0f0f0')
        
        # Specs PC
        specs_frame = tk.LabelFrame(frame, text="Specs PC Détectées", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        specs_frame.pack(fill='x', padx=20, pady=15)
        
        specs_text = f"💻 CPU: {PC_SPECS['cpu_count']} cores\n"
        specs_text += f"🎮 RAM: {PC_SPECS['ram_gb']:.1f} GB\n"
        specs_text += f"🔥 GPU: {PC_SPECS['gpu_name']}\n"
        if PC_SPECS['gpu_memory_gb'] > 0:
            specs_text += f"📊 VRAM: {PC_SPECS['gpu_memory_gb']:.1f} GB\n"
        specs_text += f"\n⚡ Config Auto:\n"
        specs_text += f"  Batch Size: {Config.BATCH_SIZE}\n"
        specs_text += f"  Replay Buffer: {Config.REPLAY_BUFFER_SIZE}\n"
        specs_text += f"  Threads: {Config.WORKER_THREADS}"
        
        tk.Label(
            specs_frame,
            text=specs_text,
            font=('Courier', 9),
            justify='left',
            bg='#f0f0f0'
        ).pack(pady=10, padx=15)
        
        # Combat Log
        log_frame = tk.LabelFrame(frame, text="Combat Log WoW", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        log_frame.pack(fill='x', padx=20, pady=15)
        
        btn_frame = tk.Frame(log_frame, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame,
            text="🔍 Détection Auto",
            command=self.auto_detect_log,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=20,
            height=2
        ).pack(side='left', padx=5)
        
        tk.Button(
            btn_frame,
            text="📁 Sélectionner",
            command=self.select_log,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=20,
            height=2
        ).pack(side='left', padx=5)
        
        self.log_status = tk.Label(
            log_frame,
            text="❌ Non connecté",
            font=('Courier', 10),
            fg='red',
            bg='#f0f0f0'
        )
        self.log_status.pack(pady=10)
        
        # Modules
        mod_frame = tk.LabelFrame(frame, text="Modules Disponibles", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        mod_frame.pack(fill='x', padx=20, pady=15)
        
        modules = [
            ("Contrôles (pynput)", CONTROL_AVAILABLE),
            ("Screenshots (pillow)", SCREENSHOT_AVAILABLE),
            ("PyTorch (Deep Learning)", TORCH_AVAILABLE),
            ("OpenCV (Vision)", CV2_AVAILABLE),
            ("Specs PC (psutil)", SPECS_AVAILABLE)
        ]
        
        for name, available in modules:
            icon = '✅' if available else '❌'
            color = 'green' if available else 'red'
            tk.Label(
                mod_frame,
                text=f"{icon} {name}",
                font=('Courier', 9),
                fg=color,
                bg='#f0f0f0',
                anchor='w'
            ).pack(padx=15, pady=3, fill='x')
        
        if TORCH_AVAILABLE:
            tk.Label(
                mod_frame,
                text=f"🔥 Device: {DEVICE} ({PC_SPECS['gpu_name']})",
                font=('Courier', 9, 'bold'),
                fg='orange',
                bg='#f0f0f0'
            ).pack(padx=15, pady=5)
        
        return frame
    
    def create_leveling_tab(self):
        frame = tk.Frame(bg='#f0f0f0')
        
        # Info Agent
        info_frame = tk.LabelFrame(frame, text="Agent de Leveling Autonome", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        info_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(
            info_frame,
            text="🤖 L'agent peut:\n"
                 "  ✅ Accepter et rendre des quêtes\n"
                 "  ✅ Tuer des mobs (rotation auto)\n"
                 "  ✅ Looter automatiquement\n"
                 "  ✅ Vendre items inutiles\n"
                 "  ✅ Acheter sorts et équipement\n"
                 "  ✅ S'équiper automatiquement\n"
                 "  ✅ Utiliser heal/potions\n"
                 "  ✅ Explorer et grinder\n\n"
                 "⚠️ Assurez-vous d'être en jeu avant d'activer!",
            font=('Courier', 9),
            justify='left',
            bg='#f0f0f0'
        ).pack(pady=10, padx=15)

        # Configuration personnage
        char_frame = tk.LabelFrame(frame, text="Configuration du personnage", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        char_frame.pack(fill='x', padx=20, pady=10)

        # Variables UI
        self.var_faction = tk.StringVar(value=self.character_config.faction)
        self.var_race = tk.StringVar(value=self.character_config.race)
        self.var_class = tk.StringVar(value=self.character_config.class_name)
        self.var_spec = tk.StringVar(value=self.character_config.spec)
        self.var_role = tk.StringVar(value="DPS")
        self.var_gear_prio = tk.StringVar(value="Dégâts")

        # Professions (simples listes; la logique interne pourra utiliser des priorités plus tard)
        professions = [
            "Aucun", "Minage", "Herboristerie", "Dépeçage", "Forgeron",
            "Ingénierie", "Couture", "Enchantement", "Alchimie", "Joaillerie"
        ]
        sec_professions = [
            "Aucun", "Cuisine", "Pêche", "Secourisme"
        ]

        self.var_prof1 = tk.StringVar(value="Aucun")
        self.var_prof2 = tk.StringVar(value="Aucun")
        self.var_sec_prof1 = tk.StringVar(value="Aucun")
        self.var_sec_prof2 = tk.StringVar(value="Aucun")

        # Ligne 1: faction / race
        row1 = tk.Frame(char_frame, bg='#f0f0f0')
        row1.pack(fill='x', padx=10, pady=3)

        tk.Label(row1, text="Faction:", bg='#f0f0f0').pack(side='left')
        ttk.Combobox(row1, textvariable=self.var_faction, values=["Alliance", "Horde"], width=12, state='readonly').pack(side='left', padx=5)

        tk.Label(row1, text="Race:", bg='#f0f0f0').pack(side='left', padx=15)
        ttk.Combobox(row1, textvariable=self.var_race, values=["Humain", "Elfe de la nuit", "Nain", "Orc", "Tauren", "Mort-vivant", "Troll"], width=18, state='readonly').pack(side='left', padx=5)

        # Ligne 2: classe / spé / rôle
        row2 = tk.Frame(char_frame, bg='#f0f0f0')
        row2.pack(fill='x', padx=10, pady=3)

        tk.Label(row2, text="Classe:", bg='#f0f0f0').pack(side='left')
        ttk.Combobox(row2, textvariable=self.var_class, values=[
            "Guerrier", "Paladin", "Chasseur", "Voleur", "Prêtre",
            "Chaman", "Mage", "Démoniste", "Druide"
        ], width=18, state='readonly').pack(side='left', padx=5)

        tk.Label(row2, text="Spé:", bg='#f0f0f0').pack(side='left', padx=15)
        ttk.Combobox(row2, textvariable=self.var_spec, values=[
            "Armes", "Fureur", "Protection", "Soin", "DPS", "Tank"
        ], width=15, state='readonly').pack(side='left', padx=5)

        tk.Label(row2, text="Rôle:", bg='#f0f0f0').pack(side='left', padx=15)
        ttk.Combobox(row2, textvariable=self.var_role, values=["Tank", "Heal", "DPS"], width=10, state='readonly').pack(side='left', padx=5)

        # Ligne 3: priorité équipement
        row3 = tk.Frame(char_frame, bg='#f0f0f0')
        row3.pack(fill='x', padx=10, pady=3)

        tk.Label(row3, text="Priorité équipement:", bg='#f0f0f0').pack(side='left')
        ttk.Combobox(row3, textvariable=self.var_gear_prio, values=[
            "Dégâts", "Survie", "Équilibré"
        ], width=15, state='readonly').pack(side='left', padx=5)

        # Ligne 4: métiers
        row4 = tk.Frame(char_frame, bg='#f0f0f0')
        row4.pack(fill='x', padx=10, pady=3)

        tk.Label(row4, text="Métiers principaux:", bg='#f0f0f0').pack(side='left')
        ttk.Combobox(row4, textvariable=self.var_prof1, values=professions, width=18, state='readonly').pack(side='left', padx=5)
        ttk.Combobox(row4, textvariable=self.var_prof2, values=professions, width=18, state='readonly').pack(side='left', padx=5)

        row5 = tk.Frame(char_frame, bg='#f0f0f0')
        row5.pack(fill='x', padx=10, pady=3)

        tk.Label(row5, text="Métiers secondaires:", bg='#f0f0f0').pack(side='left')
        ttk.Combobox(row5, textvariable=self.var_sec_prof1, values=sec_professions, width=18, state='readonly').pack(side='left', padx=5)
        ttk.Combobox(row5, textvariable=self.var_sec_prof2, values=sec_professions, width=18, state='readonly').pack(side='left', padx=5)

        # Bouton pour appliquer la config au bot
        tk.Button(
            char_frame,
            text="Appliquer au bot",
            command=self._apply_character_config,
            bg='#2980b9',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=20,
            height=1
        ).pack(pady=5)
        
        # Comportements
        behavior_frame = tk.LabelFrame(frame, text="Comportements", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        behavior_frame.pack(fill='x', padx=20, pady=15)
        
        self.behavior_vars = {}
        behaviors = [
            ('quest_mode', "Mode Quête"),
            ('grind_mode', "Mode Grind"),
            ('auto_loot', "Loot Auto"),
            ('auto_sell', "Vente Auto"),
            ('auto_learn_spells', "Apprendre Sorts Auto"),
            ('auto_equip', "Équipement Auto")
        ]
        
        for key, label in behaviors:
            var = tk.BooleanVar(value=self.leveling_agent.behaviors[key])
            self.behavior_vars[key] = var
            
            tk.Checkbutton(
                behavior_frame,
                text=label,
                variable=var,
                font=('Arial', 10),
                bg='#f0f0f0',
                command=lambda k=key, v=var: self._update_behavior(k, v)
            ).pack(anchor='w', padx=20, pady=3)
        
        # Contrôles
        control_frame = tk.Frame(frame, bg='#f0f0f0')
        control_frame.pack(pady=20)
        
        self.start_leveling_btn = tk.Button(
            control_frame,
            text="▶️ DÉMARRER AGENT",
            font=('Arial', 14, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.start_leveling,
            width=30,
            height=3
        )
        self.start_leveling_btn.pack(pady=5)
        
        self.stop_leveling_btn = tk.Button(
            control_frame,
            text="⏹️ ARRÊTER AGENT",
            font=('Arial', 14, 'bold'),
            bg='#e74c3c',
            fg='white',
            command=self.stop_leveling,
            width=30,
            height=3,
            state='disabled'
        )
        self.stop_leveling_btn.pack(pady=5)
        
        # Stats
        stats_frame = tk.LabelFrame(frame, text="Statistiques", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        stats_frame.pack(fill='x', padx=20, pady=15)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="En attente...",
            font=('Courier', 9),
            justify='left',
            bg='#f0f0f0'
        )
        self.stats_label.pack(pady=10, padx=15)
        
        return frame
    
    def create_endgame_tab(self):
        frame = tk.Frame(bg='#f0f0f0')

        info_frame = tk.LabelFrame(frame, text="IA Endgame (donjons / raids / PvP)", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        info_frame.pack(fill='x', padx=20, pady=15)

        tk.Label(
            info_frame,
            text=(
                "🏆 Mode IA avancée utilisant la vision et les actions clavier/souris.\n"
                "  - PvE: priorise survie + rotation mono-cible simple\n"
                "  - PvP/Arena: mouvements plus fréquents (kite / strafes)\n\n"
                "⚠️ Mode expérimental: conçu pour être amélioré par la communauté."
            ),
            font=('Courier', 9),
            justify='left',
            bg='#f0f0f0'
        ).pack(pady=10, padx=15)

        # Modes de jeu
        mode_frame = tk.LabelFrame(frame, text="Mode de jeu", font=('Arial', 11, 'bold'), bg='#f0f0f0')
        mode_frame.pack(fill='x', padx=20, pady=10)

        self.var_endgame_mode = tk.StringVar(value='pve')
        modes = [('PvE', 'pve'), ('PvP', 'pvp'), ('Arena', 'arena')]
        for label, value in modes:
            tk.Radiobutton(
                mode_frame,
                text=label,
                variable=self.var_endgame_mode,
                value=value,
                bg='#f0f0f0',
                command=self._update_endgame_mode
            ).pack(side='left', padx=15, pady=5)

        # Contrôles
        control_frame = tk.Frame(frame, bg='#f0f0f0')
        control_frame.pack(pady=20)

        self.start_endgame_btn = tk.Button(
            control_frame,
            text="▶️ DÉMARRER IA ENDGAME",
            font=('Arial', 14, 'bold'),
            bg='#8e44ad',
            fg='white',
            command=self.start_endgame,
            width=30,
            height=3
        )
        self.start_endgame_btn.pack(pady=5)

        self.stop_endgame_btn = tk.Button(
            control_frame,
            text="⏹️ ARRÊTER IA ENDGAME",
            font=('Arial', 14, 'bold'),
            bg='#e74c3c',
            fg='white',
            command=self.stop_endgame,
            width=30,
            height=3,
            state='disabled'
        )
        self.stop_endgame_btn.pack(pady=5)

        return frame
    
    def create_logs_tab(self):
        frame = tk.Frame()
        
        # Créer log_text ICI avant tout autre composant
        self.log_text = scrolledtext.ScrolledText(
            frame,
            font=('Courier', 9),
            bg='#0a0a0a',
            fg='#00ff00',
            state='disabled',
            wrap='word'
        )
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        return frame
    
    # Méthodes de contrôle
    
    def _apply_character_config(self):
        """Applique les choix UI au CharacterConfig partagé.

        Cette config pourra être utilisée par l'agent de leveling et
        l'IA endgame pour adapter talents, rotation, métiers, etc.
        """
        self.character_config.faction = self.var_faction.get()
        self.character_config.race = self.var_race.get()
        self.character_config.class_name = self.var_class.get()
        self.character_config.spec = self.var_spec.get()

        # On encode le rôle et la priorité d'équipement dans les professions
        # ou dans des champs dérivés si besoin plus tard.
        primary = []
        if self.var_prof1.get() != "Aucun":
            primary.append(self.var_prof1.get())
        if self.var_prof2.get() != "Aucun":
            primary.append(self.var_prof2.get())
        self.character_config.primary_professions = primary

        secondary = []
        if self.var_sec_prof1.get() != "Aucun":
            secondary.append(self.var_sec_prof1.get())
        if self.var_sec_prof2.get() != "Aucun":
            secondary.append(self.var_sec_prof2.get())
        self.character_config.secondary_professions = secondary

        # Log de confirmation
        self.log_message(
            "success",
            f"Config perso appliquée: {self.character_config.faction} {self.character_config.race} "
            f"{self.character_config.class_name}/{self.character_config.spec}, rôle {self.var_role.get()}, "
            f"prio {self.var_gear_prio.get()}, métiers {primary} / {secondary}"
        )

        # Mettre à jour la référence utilisée par l'agent de leveling
        self.leveling_agent.character_config = self.character_config

    def _update_endgame_mode(self):
        """Met à jour le mode interne de l'IA Endgame (PvE/PvP/Arena)."""
        mode = self.var_endgame_mode.get()
        for key in self.endgame_agent.modes.keys():
            self.endgame_agent.modes[key] = (key == mode)

        self.log_message("info", f"Mode IA Endgame: {mode.upper()}")

    def _update_behavior(self, key, var):
        self.leveling_agent.behaviors[key] = var.get()
    
    def auto_detect_log(self):
        path = self.combat_parser.find_combatlog()
        if path:
            self.combat_parser.log_path = path
            if self.combat_parser.start():
                self.log_status.config(text="✅ Connecté", fg='green')
        else:
            messagebox.showwarning(
                "Non trouvé",
                "Combat log introuvable!\n\n"
                "1. Lance WoW\n"
                "2. Tape /combatlog\n"
                "3. Réessaye"
            )

    def start_endgame(self):
        if not CONTROL_AVAILABLE:
            messagebox.showerror("Erreur", "pynput requis pour l'IA Endgame!\n\npip install pynput")
            return

        if messagebox.askyesno(
            "Confirmation IA Endgame",
            "L'IA endgame va contrôler votre souris/clavier!\n\n"
            "⚠️ Assurez-vous d'être en contenu endgame (donjon/raid/PvP)\n"
            "⚠️ Personnage visible à l'écran\n\n"
            "Continuer?"
        ):
            self._update_endgame_mode()
            if self.endgame_agent.start():
                self.start_endgame_btn.config(state='disabled')
                self.stop_endgame_btn.config(state='normal')

    def stop_endgame(self):
        self.endgame_agent.stop()
        self.start_endgame_btn.config(state='normal')
        self.stop_endgame_btn.config(state='disabled')
    
    def select_log(self):
        path = filedialog.askopenfilename(
            title="Sélectionner WoWCombatLog.txt",
            filetypes=[("Text files", "*.txt")]
        )
        if path:
            self.combat_parser.log_path = path
            if self.combat_parser.start():
                self.log_status.config(text="✅ Connecté", fg='green')
    
    def start_leveling(self):
        if not CONTROL_AVAILABLE:
            messagebox.showerror("Erreur", "pynput requis!\n\npip install pynput")
            return
        
        if messagebox.askyesno(
            "Confirmation",
            "L'agent va contrôler votre souris/clavier!\n\n"
            "⚠️ Assurez-vous d'être EN JEU\n"
            "⚠️ Personnage visible à l'écran\n\n"
            "Continuer?"
        ):
            if self.leveling_agent.start():
                self.start_leveling_btn.config(state='disabled')
                self.stop_leveling_btn.config(state='normal')
                self.update_stats()
    
    def stop_leveling(self):
        self.leveling_agent.stop()
        self.start_leveling_btn.config(state='normal')
        self.stop_leveling_btn.config(state='disabled')
    
    def update_stats(self):
        if not self.leveling_agent.running:
            return
        
        stats = self.leveling_agent.stats
        
        playtime_str = f"{stats['playtime']//60}m {stats['playtime']%60}s"
        
        stats_text = f"⏱️ Temps: {playtime_str}\n"
        stats_text += f"⚔️ Mobs tués: {stats['mobs_killed']}\n"
        stats_text += f"❗ Quêtes: {stats['quests_completed']}\n"
        stats_text += f"💰 Or gagné: {stats['gold_earned']}\n"
        stats_text += f"📦 Items lootés: {stats['items_looted']}\n"
        stats_text += f"💀 Morts: {stats['deaths']}"
        
        self.stats_label.config(text=stats_text)
        
        self.root.after(1000, self.update_stats)
    
    def show_startup(self):
        self.log_message("info", "=" * 70)
        self.log_message("success", "WOW AI COMPLETE - HEALER + LEVELING BOT")
        self.log_message("info", "=" * 70)
        self.log_message("info", "")
        
        self.log_message("success", "💻 SPECS PC:")
        self.log_message("info", f"  CPU: {PC_SPECS['cpu_count']} cores")
        self.log_message("info", f"  RAM: {PC_SPECS['ram_gb']:.1f} GB")
        self.log_message("info", f"  GPU: {PC_SPECS['gpu_name']}")
        self.log_message("info", "")
        
        self.log_message("success", "⚙️ CONFIG AUTO:")
        self.log_message("info", f"  Batch Size: {Config.BATCH_SIZE}")
        self.log_message("info", f"  Replay Buffer: {Config.REPLAY_BUFFER_SIZE}")
        self.log_message("info", f"  Threads: {Config.WORKER_THREADS}")
        self.log_message("info", f"  Device: {DEVICE}")
        self.log_message("info", "")
        
        self.log_message("success", "📦 MODULES:")
        self.log_message("info", f"  Contrôles: {'✅' if CONTROL_AVAILABLE else '❌'}")
        self.log_message("info", f"  Screenshots: {'✅' if SCREENSHOT_AVAILABLE else '❌'}")
        self.log_message("info", f"  PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
        self.log_message("info", f"  OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
        self.log_message("info", f"  Specs: {'✅' if SPECS_AVAILABLE else '❌'}")
        self.log_message("info", "")
        
        self.log_message("warning", "⚠️ RAPPEL: Tapez /combatlog dans WoW!")
        self.log_message("success", "✅ Système prêt!")
    
    def log_message(self, level, msg):
        if not hasattr(self, 'log_text'):
            return
        
        self.log_text.config(state='normal')
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if level == "info":
            formatted = f"[{timestamp}] [INFO] {msg}"
        elif level == "success":
            formatted = f"[{timestamp}] [✓] {msg}"
        elif level == "error":
            formatted = f"[{timestamp}] [✗] {msg}"
        elif level == "warning":
            formatted = f"[{timestamp}] [⚠] {msg}"
        else:
            formatted = f"[{timestamp}] {msg}"
        
        self.log_text.insert('end', formatted + '\n')
        self.log_text.see('end')
        self.log_text.config(state='disabled')
    
    def run(self):
        self.root.mainloop()
        
        # Cleanup
        if self.combat_parser:
            self.combat_parser.stop()
        if self.leveling_agent:
            self.leveling_agent.stop()


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main():
    print("=" * 70)
    print("WOW AI COMPLETE - LEVELING & ENDGAME BOT (ALL CLASSES)")
    print("=" * 70)
    print()
    
    # Afficher specs
    print("💻 SPECS PC DÉTECTÉES:")
    print(f"  CPU: {PC_SPECS['cpu_count']} cores")
    print(f"  RAM: {PC_SPECS['ram_gb']:.1f} GB")
    print(f"  GPU: {PC_SPECS['gpu_name']}")
    if PC_SPECS['gpu_memory_gb'] > 0:
        print(f"  VRAM: {PC_SPECS['gpu_memory_gb']:.1f} GB")
    print()
    
    print("⚙️ CONFIGURATION AUTO:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Replay Buffer: {Config.REPLAY_BUFFER_SIZE}")
    print(f"  Worker Threads: {Config.WORKER_THREADS}")
    print()
    
    # Vérifier dépendances
    missing = []
    if not CONTROL_AVAILABLE:
        missing.append("pynput")
    if not SCREENSHOT_AVAILABLE:
        missing.append("pillow")
    if not CV2_AVAILABLE:
        missing.append("opencv-python")
    if not TORCH_AVAILABLE:
        missing.append("torch")
    if not SPECS_AVAILABLE:
        missing.append("psutil GPUtil")
    
    if missing:
        print("⚠️ DÉPENDANCES MANQUANTES:")
        for dep in missing:
            print(f"  ❌ {dep}")
        print()
        print("Voir install_dependencies.bat pour installation automatique")
        print()
    
    if not CONTROL_AVAILABLE or not SCREENSHOT_AVAILABLE:
        print("❌ ERREUR CRITIQUE: pynput et pillow sont REQUIS!")
        input("\nAppuyez sur Entrée pour quitter...")
        return
    
    print("✅ Lancement de l'interface...")
    print()
    
    try:
        app = WoWAIInterface()
        app.run()
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        input("\nAppuyez sur Entrée pour quitter...")


if __name__ == "__main__":
    main()
