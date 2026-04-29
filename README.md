# NOM ET PRENOM: KRIKET NASSIRA 
# 🧠 ProjetMLP — Cartographie d'une Fonction Mystère

Implémentation complète d'un **Perceptron Multicouche (MLP)** from scratch en utilisant uniquement **NumPy**, sans aucun framework de haut niveau (PyTorch, TensorFlow, etc.).

---

##  Objectif

Entraîner un MLP à apprendre et reconstruire la surface mathématique suivante :

$$f(x, y) = \sin\left(\sqrt{x^2 + y^2}\right) + 0.5 \cdot \cos(2x + 2y)$$

---

## 📁 Structure du Projet

```
ProjetMLP/
│
├── processed_data.npz
├── norm_params.json
│
├── SEMAINE1_GENERATION_DATASET/
│   ├── consultation1.py
│   └── data/
│       ├── generate.py          # Génération des 2000 points
│       ├── normalize.py         # Normalisation z-score
│       └── visualize.py         # Scatter 3D
│
├── SEMAINE2_ARCHITECTURE_RESEAU/
│   ├── consultation2.py
│   └── architecture_réseau/
│       ├── mlp.py               # Classe MLP complète
│       ├── activations.py       # ReLU, Linear
│       ├── initialisation.py    # He, Xavier
│       └── loss.py              # MSE Loss
│
├── SEMAINE3_BACKPROPAGATION/
│   ├── consultation3.py
│   └── backpropagation/
│       ├── train.py             # Mini-batch SGD
│       ├── gradients_sortie.py  # Gradient couche de sortie
│       ├── gradients_cachees.py # Gradient couches cachées
│       └── optimizers.py        # SGD update
│
├── SEMAINE4_ENTRAINEMENT_ET_VISUALISATION/
│   ├── test_final.py
│   ├── evaluation.py
│   └── visualisation.py
│
└── questions_réflexion/
    ├── experiment_linear.py         # Q1 : Activation linéaire
    ├── experiment_small_network.py  # Q2 : Petit réseau [2,4,1]
    └── experiment_momentum.py       # Q3 : SGD + Momentum
```

---

##  Installation

```bash
pip install numpy matplotlib
```

---

##  Exécution

```bash
# Étape 1 — Générer les données
cd SEMAINE1_GENERATION_DATASET && python consultation1.py

# Étape 2 — Tester l'architecture
cd SEMAINE2_ARCHITECTURE_RESEAU && python consultation2.PY

# Étape 3 — Backpropagation
cd SEMAINE3_BACKPROPAGATION && python consultation3.py

# Étape 4 — Test final
cd SEMAINE4_ENTRAINEMENT_ET_VISUALISATION && python test_final.py
```

---

##  Questions de Réflexion

### Q1 — Activation linéaire
```bash
cd questions_réflexion && python experiment_linear.py
```
> **Résultat** : Loss stagne à ≈ 1.0, prédiction = surface plate.  
> Un réseau linéaire ≡ régression linéaire → **impossible d'apprendre une fonction sinusoïdale**.

### Q2 — Petit réseau [2, 4, 1]
```bash
cd questions_réflexion && python experiment_small_network.py
```
> **Résultat** : Loss finale ≈ 0.75 (vs 0.49 pour le grand réseau).  
> Seulement 17 paramètres → **underfitting**.

### Q3 — SGD + Momentum (β = 0.9)
```bash
cd questions_réflexion && python experiment_momentum.py
```
> **Résultat** : Loss finale ≈ 0.38, convergence plus rapide avec oscillations.

---

##  Architecture

| Couche | Neurones | Activation |
|--------|----------|------------|
| Entrée | 2 | — |
| Cachée 1 | 64 | ReLU |
| Cachée 2 | 64 | ReLU |
| Sortie | 1 | Linéaire |

---

##  Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Learning rate | 0.01 | Compromis optimal vitesse/stabilité |
| Batch size | 64 | 32 mises à jour/époque |
| Époques | 500 | Convergence observée |
| Initialisation | He | Adaptée à ReLU |
| Loss | MSE | Tâche de régression |

---

##  Résultats Comparatifs

| Configuration | Loss finale |
|---|---|
| SGD standard `[2, 64, 64, 1]` | ≈ 0.49 |
| Activation linéaire | ≈ 1.00 ❌ |
| Petit réseau `[2, 4, 1]` | ≈ 0.75 ⚠️ |
| SGD + Momentum | ≈ 0.38 ✅ |

---

