# Computer Vision: CNN, ViT & YOLO

Une implémentation complète d'architectures de vision par ordinateur pour la classification d'images, incluant **CNNs personnalisés**, **Vision Transformers** et préparation pour **YOLO**.

## 📋 Vue d'ensemble

Ce projet explore et compare différentes architectures de deep learning pour la vision par ordinateur :

- **CNN (Convolutional Neural Networks)** - 6 architectures classiques et modernes
- **ViT (Vision Transformers)** - Architecture Transformer pour vision (en développement)
- **YOLO** - Préparation pour détection d'objets (en développement)

### Modèles CNN implémentés

| Modèle | Couches Conv | Filtres max | Dropout | Use Case |
|--------|--------------|-------------|---------|----------|
| **CNN1** | 3 | 16 | ❌ | Baseline simple |
| **CNN2** | 3 | 8 | ❌ | Étude de la profondeur |
| **CNN3** | 3 | 8 | ❌ | Petits filtres (3×3) |
| **CNN4** | 1 | 4 | ❌ | Modèle minimaliste |
| **LeNet** | 2 | 16 | ❌ | Classique (1998) |
| **mycnn (cnnTD3)** | 3 | 192 | ✅ | Moderne avec régularisation |

## 🚀 Démarrage rapide

### Prérequis

- Python 3.8+
- PyTorch 2.0+
- GPU recommandé (CPU supporté)

### Installation

```bash
# Cloner le repository
git clone https://github.com/etiennechevrolat/computer-vision-cnn-vit-yolo.git
cd computer-vision-cnn-vit-yolo

# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation basique

```bash
# Lancer l'entraînement
python main.py

# Visualiser les métriques avec TensorBoard
tensorboard --logdir=./logs
# Accédez à http://localhost:6006
```

## 📁 Structure du projet

```
computer-vision-cnn-vit-yolo/
├── main.py                 # Point d'entrée principal
├── requirements.txt        # Dépendances Python
├── README.md              # Cette documentation
│
├── src/                   # Code source
│   ├── cnn/
│   │   ├── models.py      # Définitions des modèles CNN
│   │   └── __pycache__/
│   │
│   ├── utils/
│   │   ├── trainer.py     # Boucle d'entraînement & évaluation
│   │   └── __pycache__/
│   │
│   ├── vit/               # Vision Transformers (en développement)
│   ├── vlm/               # Vision Language Models (en développement)
│   └── data/              # Utilitaires de données
│
├── data/                  # Datasets
│   └── MNIST/
│       └── raw/           # Données brutes téléchargées
│
├── logs/                  # TensorBoard logs
│   └── mycnn/             # Logs d'entraînement par modèle
│
└── notebooks/
    └── pres.ipynb         # Notebooks d'exploration
```

## ⚙️ Configuration

Modifiez `main.py` pour ajuster les hyperparamètres :

```python
# Hyperparamètres d'entraînement
epochs = 100           # Nombre d'epochs
batch_size = 128       # Taille des batches
lr = 1e-3              # Learning rate
MODEL_NAME = "mycnn"   # Modèle à utiliser

# Choisir le modèle
model = mycnn().to(device)  # Autres options : CNN1, CNN2, CNN3, CNN4, LeNet

# Sélectionner le dataset
# Options : MNIST, CIFAR10, CIFAR100
train_data = datasets.CIFAR10(root="./data", train=True, download=True, ...)
```

## 📊 Suivi de l'entraînement

### TensorBoard

Les métriques d'entraînement sont enregistrées en temps réel :

```bash
tensorboard --logdir=./logs
```

**Métriques disponibles :**
- Loss/train - Perte d'entraînement par batch
- Logs sauvegardés dans `logs/{MODEL_NAME}/`

### Structure des logs

```
logs/
├── mycnn/
│   ├── events.out.tfevents...
│   └── ...autres fichiers TensorBoard
└── autre_modele/
```

## 🔧 Détails techniques

### Dépendances principales

- **torch** (≥2.0) - Framework deep learning
- **torchvision** (≥0.15) - Datasets et transformations d'images
- **tensorboard** - Visualisation des métriques
- **tqdm** - Barres de progression
- **timm** (≥0.9) - Vision models pré-entraînés
- **einops** - Opérations tenseur flexibles

### Architectures CNN détaillées

#### mycnn (Recommended)
- 3 couches convolutives avec dropout
- Filtres progressifs : 32 → 64 → 192
- Dropout 2D dans les couches conv (0.2)
- Dropout FC dans les couches denses (0.2)
- Parfait pour CIFAR-10

#### LeNet (Classique)
- Architecture LeNet-5 originale (1998)
- 2 couches convolutives + 3 FC
- Baseline pour comparaisons historiques

#### CNN1-4
- Variations pour études d'architecture
- CNN1 : Baseline avec 16 filtres
- CNN2 : Étude de la profondeur (8 filtres)
- CNN3 : Petits filtres 3×3
- CNN4 : Une seule couche convolutive

## 🎯 Utilisation avancée

### Entraîner un modèle spécifique

```python
# main.py
from cnn.models import CNN3, LeNet

model = CNN3().to(device)  # Ou LeNet, CNN1, etc.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

trainer(
    train_data, 
    model, 
    optimizer, 
    loss_fn, 
    epochs=50,
    batch_size=64,
    run_name="CNN3_experiment"
)

# Évaluer le modèle
accuracy = success_rate(model, test_data)
print(f"Accuracy: {accuracy:.2%}")
```

### Sauvegarder et charger un modèle

```python
# Sauvegarder
torch.save(model.state_dict(), 'models/mycnn.pth')

# Charger
model = mycnn()
model.load_state_dict(torch.load('models/mycnn.pth'))
model.eval()
```

## 📈 Résultats attendus

Sur **CIFAR-10** avec 100 epochs :
- mycnn : ~85-90% accuracy
- LeNet : ~70-75% accuracy
- CNN3 : ~75-80% accuracy

*Les résultats varient selon les hyperparamètres et l'initialisation*

## 🚧 Développement futur

- [ ] Vision Transformers (ViT) - Implémentation complète
- [ ] Vision Language Models (VLM) - Modèles multimodaux
- [ ] YOLO v8 - Détection d'objets
- [ ] Augmentation de données avancée
- [ ] Transfer learning avec modèles pré-entraînés
- [ ] Évaluation et comparaison des architectures
- [ ] Support multi-GPU

## 🤝 Contribution

Les contributions sont bienvenues ! Vous pouvez :
- Ajouter de nouveaux modèles
- Optimiser les hyperparamètres
- Améliorer la documentation
- Rapporter des bugs

## 📝 License

Ce projet est fourni à titre éducatif.

## 📧 Contact

Pour des questions ou suggestions : etiennechevrolat@example.com

---

**Dernière mise à jour :** Avril 2026  
**Version :** 1.0.0  
**Status :** Actif - En développement