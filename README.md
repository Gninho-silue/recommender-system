# 🎬 MovieLens Recommendation System on Amazon SageMaker

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Système de recommandation de films intelligent utilisant PyTorch et Amazon SageMaker**
> 
> Projet de fin de semestre - Module Virtualisation & Cloud Computing  
> ENSAH - Génie Informatique Option Logiciel | 2025/2026

---

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Structure du Projet](#structure-du-projet)
- [Technologies](#technologies)
- [Auteur](#auteur)

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système de recommandation hybride** (Collaborative Filtering + Content-Based) pour prédire les préférences cinématographiques des utilisateurs. Le système est entraîné sur le dataset **MovieLens 100K** et déployé sur **Amazon SageMaker**.

### Objectifs du Projet

- ✅ Construire un modèle de recommandation performant avec PyTorch
- ✅ Exploiter les services AWS (S3, SageMaker, IAM)
- ✅ Démontrer la maîtrise du Cloud Computing et de la Virtualisation
- ✅ Créer une interface utilisateur interactive

---

## ⚡ Fonctionnalités

### 🤖 Modèle de Machine Learning

- **Architecture hybride** : Collaborative Filtering + Content-Based Filtering
- **Deep Neural Network** : 3 couches cachées [256, 128, 64]
- **Embeddings** : 128 dimensions pour utilisateurs et films
- **Feature Engineering** : 21 features (démographiques, temporelles, interactions)
- **Optimisation** : Adam optimizer avec Learning Rate Scheduler

### 📊 Métriques de Performance

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **RMSE** | 0.6247 | Erreur moyenne de 0.62 étoiles |
| **MAE** | 0.4492 | Précision absolue de 0.45 étoiles |
| **Hit Rate** | 70.4% | 70% de prédictions correctes |

### 🎨 Interface Utilisateur

- **Application Streamlit** interactive
- Profils utilisateurs détaillés
- Recommandations Top-K personnalisées
- Visualisations interactives (Plotly)
- Design moderne et responsive

---

## 🏗️ Architecture

### Architecture Système

```
┌─────────────────────────────────────────────────────────────┐
│                     Amazon SageMaker                        |
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │   Notebook   │──▶│   Training   │──▶│   Endpoint  │     │  
│  │   Instance   │   │     Job      │   │   (Deploy)   │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│    Development          Model              Inference        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Amazon S3     │
                    │  (Data Storage)  │
                    └──────────────────┘
```

### Architecture du Modèle

```
Input Layer
    │
    ├─▶ User Embedding (128D) ──┐
    │                            │
    └─▶ Item Embedding (128D) ──┤
                                 ├──▶ Concatenation
    ┌─▶ Features (19D) ─────────┘
    │
    ▼
Fully Connected Layers
    │
    ├─▶ FC1 (256 neurons) + ReLU + Dropout + BatchNorm
    ├─▶ FC2 (128 neurons) + ReLU + Dropout + BatchNorm
    ├─▶ FC3 (64 neurons)  + ReLU + Dropout + BatchNorm
    │
    ▼
Output Layer (1 neuron)
    │
    ▼
Predicted Rating (1-5)
```

---

## 🚀 Installation

### Prérequis

- Python 3.11+
- Compte AWS (Free Tier suffisant)
- Git

### Installation Locale

```bash
# Cloner le repository
git clone https://github.com/Gninho-silue/recommmender-system.git
cd/recommmender-system

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration AWS

```bash
# Configurer AWS CLI
aws configure

# Variables d'environnement
export AWS_REGION=us-east-1
export S3_BUCKET=your-bucket-name
```

---

## 📖 Utilisation

### 1. Exploration des Données

```bash
# Ouvrir le notebook dans SageMaker Studio
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Entraînement du Modèle

```bash
# Lancer l'entraînement
python src/train.py --epochs 15 --batch-size 256 --lr 0.001
```

### 3. Interface Streamlit

```bash
# Lancer l'application web
streamlit run app_streamlit.py
```

### 4. Génération de Recommandations

```python
from src.recommendation import recommend_top_k

# Recommander 10 films pour l'utilisateur 196
recommendations = recommend_top_k(user_id=196, top_k=10)

for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie['title']} (Score: {movie['predicted_rating']:.2f})")
```

---

## 📊 Résultats

### Performance du Modèle

- **Dataset** : MovieLens 100K (100,000 ratings, 943 utilisateurs, 1,682 films)
- **Split** : 80% train (80,000) / 20% test (20,000) - Split temporel
- **Sparsité** : 93.7% (matrice creuse)

### Évolution des Métriques

| Epoch | Train Loss | Test Loss | RMSE | MAE | Hit Rate |
|-------|------------|-----------|------|-----|----------|
| 1 | 5.4620 | 1.3550 | 1.1660 | 0.9696 | 44.3% |
| 5 | 0.6414 | 0.4380 | 0.6641 | 0.5185 | 70.1% |
| 9 | 0.5380 | 0.3875 | **0.6247** | **0.4492** | **75.2%** |
| 15 | 0.4628 | 0.4286 | 0.6572 | 0.4809 | 70.4% |

### Exemples de Recommandations

**Utilisateur #610** (22 ans, étudiant, aime les films classiques)

| Rang | Film | Score | Genres |
|------|------|-------|--------|
| 1 | Pather Panchali (1955) | 4.48⭐ | Drama |
| 2 | Shawshank Redemption (1994) | 4.32⭐ | Drama |
| 3 | Rear Window (1954) | 4.24⭐ | Mystery, Thriller |

---

## 📁 Structure du Projet

```
movielens-sagemaker/
│
├── data/
│   ├── raw/                        # Données brutes
│   │   └── ml-100k/
│   ├── processed/                  # Données prétraitées
│   │   ├── train_features.csv
│   │   ├── test_features.csv
│   │   ├── movies_metadata.csv
│   │   └── users_metadata.csv
│   └── sample/                     # Échantillons pour tests
│
├── models/
│   ├── saved_models/               # Modèles entraînés
│   │   └── best_model.pth
│   ├── encoders/                   # Encoders (LabelEncoder, Scaler)
│   │   ├── user_encoder.pkl
│   │   ├── item_encoder.pkl
│   │   └── feature_scaler.pkl
│   └── checkpoints/                # Checkpoints d'entraînement
│
├── notebooks/
│   ├── 00_diagnostic_environnement.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training_advanced_metrics.ipynb
│   ├── 04_recommendation_system_topk.ipynb
│   └── 05_interface_demo_interactive.ipynb
│
├── src/
│   ├── data_processing.py          # Prétraitement des données
│   ├── model.py                    # Architecture du modèle
│   ├── train.py                    # Script d'entraînement
│   ├── inference.py                # Script d'inférence
│   └── recommendation.py           # Fonctions de recommandation
│
├── outputs/
│   ├── plots/                      # Visualisations
│   │   ├── 01_exploration_overview.png
│   │   ├── 02_genres_demographie.png
│   │   ├── 03_feature_engineering.png
│   │   ├── 04_training_metrics.png
│   │   └── 05_prediction_analysis.png
│   ├── metrics/                    # Rapports JSON
│   │   ├── exploration_report.json
│   │   ├── preprocessing_report.json
│   │   ├── training_report.json
│   │   └── recommendation_report.json
│   └── logs/                       # Logs d'exécution
│
├── app_streamlit.py                # Application web Streamlit
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
└── LICENSE                         # Licence MIT
```

---

## 🛠️ Technologies

### Machine Learning & Data Science

- **PyTorch 2.6.0** - Deep Learning framework
- **pandas 2.3.1** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **scikit-learn** - Preprocessing & metrics

### Cloud & Infrastructure

- **Amazon SageMaker** - ML platform
- **Amazon S3** - Object storage
- **AWS IAM** - Access management
- **boto3** - AWS SDK for Python

### Visualisation & Interface

- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static plots

### Development Tools

- **Jupyter Notebook** - Interactive development
- **Git & GitHub** - Version control
- **VS Code** - Code editor

---

## 📈 Améliorations Futures

- [ ] Implémenter un pipeline SageMaker complet
- [ ] Ajouter le déploiement avec SageMaker Endpoint
- [ ] Intégrer des modèles pré-entraînés (BERT pour descriptions)
- [ ] Ajouter l'explicabilité des recommandations (SHAP values)
- [ ] Supporter plusieurs datasets (MovieLens 1M, 10M)
- [ ] Créer une API RESTful avec FastAPI
- [ ] Ajouter des tests unitaires (pytest)
- [ ] Monitoring avec CloudWatch

---

## 🎓 Contexte Académique

### Lien avec le Cours "Virtualisation & Cloud Computing"

Ce projet illustre les concepts clés du cours :

#### Virtualisation
- **Hyperviseur Type 1** : SageMaker utilise des instances EC2 virtualisées
- **Isolation** : Environnements d'entraînement cloisonnés
- **Conteneurs Docker** : PyTorch packagé dans des images Docker

#### Cloud Computing
- **IaaS** : Infrastructure EC2 sous-jacente
- **PaaS** : SageMaker comme plateforme gérée
- **Caractéristiques du Cloud** :
  - ✅ Élasticité : Scalabilité automatique des ressources
  - ✅ Pay-as-you-go : Facturation à l'usage
  - ✅ Self-service : Provisionnement via console/SDK
  - ✅ Mesurabilité : Monitoring via CloudWatch

#### Stockage Cloud
- **S3 (Object Storage)** : Équivalent AWS de Swift (OpenStack)
- **Redondance** : Données répliquées automatiquement
- **Accès programmatique** : API boto3

---

## 👨‍💻 Auteur

**Gninninmaguignon Silué**  
Étudiant en Génie Informatique - Option Logiciel  
ENSAH (École Nationale des Sciences Appliquées d'Al Hoceima)  
Promotion 2025/2026

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Professeur Routaib Hayat** - Enseignant du module Virtualisation & Cloud Computing
- **GroupLens Research** - Pour le dataset MovieLens
- **AWS Educate** - Pour les crédits AWS gratuits
- **Community PyTorch** - Pour la documentation et les exemples

---

## 📞 Contact

Pour toute question ou suggestion :
- 📧 Email : gninhosilue@gmail.coom
- 💼 LinkedIn : [Votre profil LinkedIn]
- 🐙 GitHub : [@Gninho-silue](https://github.com/Gninho-silue)

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ and ☕ by Gninninmaguignon Silué

</div>

