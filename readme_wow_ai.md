# 🎮 WoW AI Complete - Leveling & Endgame Bot (All Classes)

IA complète multi-classes pour World of Warcraft avec Deep Learning, leveling autonome et IA endgame expérimentale.

## 🚀 Fonctionnalités

### 🩺 **Mode Healer (À venir)**
- Deep Q-Learning avec vision par ordinateur
- Détection automatique des HP bars
- Priorisation des heals urgents
- Apprentissage par renforcement

### 🎯 **Agent de Leveling Autonome** (Fonctionnel)
- ✅ **Quêtes automatiques** - Accepte et rend les quêtes
- ✅ **Combat intelligent** - Rotation de sorts automatique
- ✅ **Loot automatique** - Ramasse tout le butin
- ✅ **Gestion inventaire** - Vend les items inutiles
- ✅ **Auto-équipement** - Équipe automatiquement les meilleures pièces
- ✅ **Achat automatique** - Achète sorts, montures et compétences
- ✅ **Survie** - Utilise heal/potions quand HP bas
- ✅ **Exploration** - Se déplace intelligemment

### ⚙️ **Configuration Automatique**
- Détecte automatiquement vos specs PC (CPU, RAM, GPU)
- Adapte les paramètres pour optimiser les performances
- Supporte GPU NVIDIA (CUDA) et CPU

## 📋 Installation

### Méthode 1 : Installation Automatique (Recommandé)

1. **Téléchargez les fichiers** :
   - `wow_ai_complete_fixed.py`
   - `install_dependencies.bat`
   - `launch_wow_ai.bat`

2. **Double-cliquez sur** `install_dependencies.bat`
   - Installe automatiquement toutes les dépendances
   - Détecte votre GPU et installe la bonne version de PyTorch

3. **Double-cliquez sur** `launch_wow_ai.bat`
   - Lance l'application

### Méthode 2 : Installation Manuelle

```bash
# Installation de base
pip install pynput==1.7.6
pip install pillow==10.0.0
pip install opencv-python==4.8.0.76
pip install numpy==1.24.3
pip install psutil==5.9.5
pip install GPUtil==1.4.0

# PyTorch avec GPU NVIDIA (CUDA)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# OU PyTorch CPU uniquement (si pas de GPU NVIDIA)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Lancer l'application
python wow_ai_complete_fixed.py
```

## 🎮 Utilisation

### Configuration Initiale

1. **Lancez World of Warcraft**
2. **Tapez `/combatlog` dans le chat** (active le combat log)
3. **Lancez l'application** via `launch_wow_ai.bat`

### Onglet ⚙️ Configuration

1. **Cliquez sur "🔍 Détection Auto"** pour trouver le combat log
2. **Vérifiez les modules** (tous doivent être ✅)
3. **Consultez vos specs PC** (détection automatique)

### Onglet 🎯 Agent Leveling

1. **Configurez les comportements** :
   - ☑️ Mode Quête : Cherche et complète les quêtes
   - ☑️ Mode Grind : Farm des mobs
   - ☑️ Loot Auto : Ramasse automatiquement
   - ☑️ Vente Auto : Vend quand inventaire plein
   - ☑️ Apprendre Sorts Auto : Achète nouveaux sorts
   - ☑️ Équipement Auto : S'équipe du meilleur stuff

2. **Positionnez votre personnage** :
   - En jeu, dans une zone de quête
   - Personnage visible à l'écran
   - Pas en combat

3. **Cliquez sur "▶️ DÉMARRER AGENT"**

4. **Surveillez les statistiques** :
   - Temps de jeu
   - Mobs tués
   - Quêtes complétées
   - Or gagné
   - Items lootés

### Arrêter l'Agent

- **Cliquez sur "⏹️ ARRÊTER AGENT"**
- OU appuyez sur **Ctrl+C** dans le terminal

## ⚡ Configuration Auto selon Specs

L'application détecte automatiquement vos specs et s'adapte :

| RAM    | Batch Size | Replay Buffer | Threads |
|--------|------------|---------------|---------|
| 32+ GB | 64         | 50,000        | 4       |
| 16 GB  | 32         | 30,000        | 2       |
| 8 GB   | 16         | 15,000        | 1       |

**Avec GPU NVIDIA** : Performances doublées !

## 🔧 Personnalisation

### Modifier les Touches

Éditez dans `wow_ai_complete_fixed.py` :

```python
# Ligne ~750 - Rotation de combat
for spell_key in ['1', '2', '3', '4']:  # Changez les touches ici
```

### Modifier les Positions de Clic

```python
# Ligne ~460 - Positions de clic
self.mouse.position = (960, 600)  # Coordonnées X, Y
```

### Ajuster les Comportements

Dans l'interface, cochez/décochez les options selon vos besoins.

## 📊 Statistiques en Temps Réel

L'onglet **📝 Logs** affiche :
- Actions effectuées
- Détections visuelles
- Événements de combat
- Erreurs éventuelles

## ⚠️ Avertissements

### Légalité
- ⚠️ **L'utilisation de bots est contre les CGU de Blizzard**
- ⚠️ **Risque de ban permanent**
- ℹ️ Ce projet est **ÉDUCATIF** - pour apprendre le Deep Learning et la vision par ordinateur

### Utilisation Responsable
- Ne laissez pas tourner 24/7
- Surveillez régulièrement
- N'utilisez pas en PvP ou raids
- Testez d'abord sur un compte secondaire

### Performances
- **Recommandé** : 16+ GB RAM, GPU NVIDIA
- **Minimum** : 8 GB RAM, CPU moderne
- Plus de RAM = meilleure performance

## 🐛 Dépannage

### "Python n'est pas installé"
- Téléchargez Python 3.10+ : https://www.python.org/downloads/
- **Important** : Cochez "Add Python to PATH" !

### "pynput non disponible"
- Lancez `install_dependencies.bat`
- Ou : `pip install pynput`

### "Combat log introuvable"
1. Lancez WoW
2. Tapez `/combatlog` dans le chat
3. Réessayez la détection auto

### L'agent ne fait rien
- Vérifiez que vous êtes **en jeu**
- Personnage doit être **visible à l'écran**
- Pas dans un menu ou cinématique

### Erreur CUDA / GPU
- Si vous n'avez pas de GPU NVIDIA, c'est normal
- L'app fonctionne aussi en mode CPU (plus lent)

### Performance faible
- Fermez les autres applications
- Baissez les paramètres dans `Config` (ligne ~150)
- Désactivez certains comportements

## 📈 Roadmap / À Venir

### Phase 1 : Leveling (✅ Actuel)
- [x] Détection objets (quêtes, PNJ, loot)
- [x] Combat automatique
- [x] Gestion inventaire basique
- [ ] Pathfinding intelligent
- [ ] Reconnaissance texte (OCR) pour quêtes

### Phase 2 : Healer IA
- [ ] Entraînement DQN complet
- [ ] Détection HP bars fiable
- [ ] Prédiction dégâts entrants
- [ ] Cooldown management

### Phase 3 : Avancé
- [x] Multi-classe support (configuration personnage + rôle + métiers)
- [ ] Addon WoW pour communication
- [ ] Interface web
- [ ] Système de profils

## 🤝 Contribution

Ce projet est open-source éducatif. Améliorations bienvenues !

## 📝 Licence

MIT License - Projet éducatif uniquement

## 💬 Support

Pour toute question ou bug :
1. Vérifiez la section **Dépannage**
2. Consultez les logs dans l'onglet 📝
3. Ouvrez une issue avec les détails

## 🎓 Crédits

Développé pour apprendre :
- Deep Q-Learning (Reinforcement Learning)
- Vision par ordinateur (OpenCV)
- Automatisation (pynput)
- Optimisation système

---

**Rappel** : Utilisez de manière responsable et éthique ! 🛡️
