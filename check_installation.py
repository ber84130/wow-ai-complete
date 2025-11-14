"""
Script de vérification de l'installation
Teste tous les modules et affiche les specs PC
"""

import sys
import platform

print("=" * 70)
print("WoW AI Complete - Vérification de l'installation")
print("=" * 70)
print()

# Python version
print("🐍 PYTHON:")
print(f"   Version: {sys.version}")
print(f"   Plateforme: {platform.platform()}")
print()

# Tests modules
print("📦 MODULES:")
modules_to_test = [
    ("pynput", "Contrôles clavier/souris"),
    ("PIL", "Screenshots (pillow)"),
    ("cv2", "Vision (opencv-python)"),
    ("numpy", "Calculs"),
    ("torch", "Deep Learning (PyTorch)"),
    ("torchvision", "PyTorch Vision"),
    ("psutil", "Specs système"),
    ("GPUtil", "Détection GPU")
]

all_ok = True
for module_name, description in modules_to_test:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "N/A")
        print(f"   ✅ {description:30} - v{version}")
    except ImportError:
        print(f"   ❌ {description:30} - NON INSTALLÉ")
        all_ok = False

print()

# PyTorch CUDA
if all_ok:
    try:
        import torch
        print("🔥 PYTORCH:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA disponible: {'✅ OUI' if torch.cuda.is_available() else '❌ NON (CPU uniquement)'}")
        if torch.cuda.is_available():
            print(f"   Device CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        print()
    except:
        pass

# Specs PC
try:
    import psutil
    import GPUtil
    
    print("💻 SPECS PC:")
    
    # CPU
    print(f"   CPU:")
    print(f"     Cores: {psutil.cpu_count()}")
    print(f"     Utilisation: {psutil.cpu_percent()}%")
    
    # RAM
    mem = psutil.virtual_memory()
    print(f"   RAM:")
    print(f"     Total: {mem.total / (1024**3):.1f} GB")
    print(f"     Disponible: {mem.available / (1024**3):.1f} GB")
    print(f"     Utilisée: {mem.percent}%")
    
    # GPU
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            print(f"   GPU:")
            for i, gpu in enumerate(gpus):
                print(f"     [{i}] {gpu.name}")
                print(f"         VRAM: {gpu.memoryTotal}MB (utilisée: {gpu.memoryUsed}MB)")
                print(f"         Load: {gpu.load*100:.1f}%")
        else:
            print(f"   GPU: Aucun GPU détecté")
    except:
        print(f"   GPU: Erreur détection")
    
    print()
except ImportError:
    print("⚠️ psutil/GPUtil non disponibles - Impossible d'afficher les specs")
    print()

# Configuration recommandée
print("⚙️ CONFIGURATION RECOMMANDÉE:")
try:
    mem_gb = psutil.virtual_memory().total / (1024**3)
    
    if mem_gb >= 32:
        print("   Votre PC: 🟢 EXCELLENT (32+ GB RAM)")
        print("   Recommandation: Tous paramètres au maximum")
    elif mem_gb >= 16:
        print("   Votre PC: 🟡 BON (16 GB RAM)")
        print("   Recommandation: Paramètres par défaut OK")
    else:
        print("   Votre PC: 🟠 MINIMUM (8 GB RAM)")
        print("   Recommandation: Réduire batch size et buffer")
except:
    print("   Impossible de déterminer")

print()

# Résumé
print("=" * 70)
if all_ok:
    print("✅ INSTALLATION COMPLÈTE ET FONCTIONNELLE!")
    print()
    print("Vous pouvez lancer l'application:")
    print("   - Double-cliquez sur launch_wow_ai.bat")
    print("   - Ou: python wow_ai_complete_fixed.py")
else:
    print("❌ INSTALLATION INCOMPLÈTE")
    print()
    print("Lancez install_dependencies.bat pour installer les modules manquants")
print("=" * 70)

input("\nAppuyez sur Entrée pour fermer...")
