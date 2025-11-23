# MovieLens Recommendation System on Amazon SageMaker

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Système de recommandation de films intelligent utilisant PyTorch et Amazon SageMaker**

---

## 📋 Projet de Fin de Semestre
**Module :** Virtualisation & Cloud Computing  
**École :** ENSAH - Génie Informatique Option Logiciel | 2025/2026  
**Auteur :** Gninninmaguignon Silué  
**Encadrant :** Pr. Routaib Hayat

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système de recommandation hybride** (Collaborative Filtering + Content-Based) pour prédire les préférences cinématographiques des utilisateurs. Le système est entraîné sur le dataset MovieLens 100K et **déployé sur Amazon SageMaker**.

### Objectifs du Projet
✅ Construire un modèle de recommandation performant avec PyTorch  
✅ Exploiter les services AWS (S3, SageMaker, IAM, CloudWatch)  
✅ Démontrer la maîtrise du Cloud Computing et de la Virtualisation  
✅ Créer une interface utilisateur interactive  
✅ **Déployer un endpoint SageMaker pour l'inférence en temps réel**

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
| **RMSE** | **0.6247** | Erreur moyenne de 0.62 étoiles ⭐⭐⭐⭐⭐ |
| **MAE** | **0.4492** | Précision absolue de 0.45 étoiles ⭐⭐⭐⭐⭐ |
| **Hit Rate** | **70.4%** | 70% de prédictions correctes ⭐⭐⭐⭐ |

### 🎨 Interface Utilisateur
- Application **Streamlit** interactive
- Profils utilisateurs détaillés
- Recommandations Top-K personnalisées
- Visualisations interactives (Plotly)
- Design moderne et responsive

### ☁️ Déploiement Cloud
- **Endpoint SageMaker** déployé avec succès
- Instance : ml.t3.medium
- Status : InService ✓
- API REST pour inférence temps réel
- Monitoring CloudWatch intégré

---

## 🏗️ Architecture

### Architecture Système

```
┌─────────────────────────────────────────────────────────────┐
│                     Amazon SageMaker                        │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   Notebook   │──▶│   Training   │──▶│   Endpoint   │   │  
│  │   Instance   │   │     Job      │   │   (Deploy)   │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│    Development          Model              Inference       │
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
git clone https://github.com/Gninho-silue/recommender-system.git
cd recommender-system

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
|-------|-----------|-----------|------|-----|----------|
| 1 | 5.4620 | 1.3550 | 1.1660 | 0.9696 | 44.3% |
| 5 | 0.6414 | 0.4380 | 0.6641 | 0.5185 | 70.1% |
| **9** | **0.5380** | **0.3875** | **0.6247** | **0.4492** | **75.2%** |
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
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training_advanced_metrics.ipynb
│   ├── 04_recommendation_system_topk.ipynb
│   ├── 05_interface_demo_interactive.ipynb
│   └── 07_deployment_sagemaker_endpoint.ipynb
│
├── src/
│   ├── data_processing.py          # Prétraitement des données
│   ├── model.py                    # Architecture du modèle
│   ├── train.py                    # Script d'entraînement
│   ├── inference.py                # Script d'inférence
│   └── recommendation.py           # Fonctions de recommandation
│
├── deployment/
│   ├── code/
│   │   ├── inference.py            # Script SageMaker
│   │   └── requirements.txt
│   └── model/                      # Artifacts pour déploiement
│
├── outputs/
│   ├── plots/                      # Visualisations
│   ├── metrics/                    # Rapports JSON
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
- **Amazon SageMaker** - ML platform (PaaS)
- **Amazon S3** - Object storage
- **AWS IAM** - Access management
- **CloudWatch** - Monitoring et logs
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

## ☁️ Déploiement sur AWS SageMaker

### Architecture de Déploiement

Le modèle a été déployé avec succès sur un **endpoint SageMaker** pour l'inférence en temps réel.

#### Configuration

- **Nom endpoint** : `movielens-recommender-endpoint`
- **Instance** : ml.t3.medium (2 vCPU, 4 GB RAM)
- **Framework** : PyTorch 2.0
- **Status** : InService ✓
- **Coût** : $0.05/heure

#### API Format

**Requête (JSON) :**
```json
{
  "user_id": 196,
  "top_k": 10
}
```

**Réponse (JSON) :**
```json
{
  "user_id": 196,
  "top_k": 10,
  "recommendations": [
    {"rank": 1, "item_id": 408, "predicted_rating": 4.52},
    {"rank": 2, "item_id": 169, "predicted_rating": 4.48},
    ...
  ]
}
```

### Challenge Technique : Trade-off Latence vs Précision

Lors du déploiement, un **problème de timeout** a été rencontré :
- **Cause** : Prédiction sur l'ensemble du catalogue (1,682 films) trop coûteuse
- **Apprentissage** : Trade-off fondamental entre qualité et temps réel en production

**Solutions envisagées pour la production :**
1. **Batch processing** : Pré-calcul des recommandations offline
2. **Caching** : Mise en cache des embeddings des items
3. **Architecture ANN** : Approximate Nearest Neighbors pour recherche rapide
4. **Instance GPU** : Utilisation de ml.g4dn.xlarge

> Ce défi illustre un problème réel du MLOps et démontre la compréhension des contraintes de déploiement en production.

---

## 📈 Améliorations Futures

### Court terme (1-3 mois)
- [ ] Optimiser l'inférence (batch processing + caching)
- [ ] Ajouter mécanismes d'attention
- [ ] Implémenter NDCG@K et autres métriques avancées
- [ ] Utiliser embeddings BERT pour les titres de films

### Moyen terme (3-6 mois)
- [ ] Pipeline SageMaker complet automatisé
- [ ] Endpoint avec auto-scaling
- [ ] Migration vers MovieLens 1M (dataset plus récent)
- [ ] API RESTful avec FastAPI

### Long terme (6-12 mois)
- [ ] Recommandations contextuelles (moment, humeur, contexte social)
- [ ] Apprentissage par renforcement
- [ ] Explainability avec SHAP values
- [ ] Déploiement multi-région

---

## 🎓 Contexte Académique

### Lien avec le Cours "Virtualisation & Cloud Computing"

Ce projet illustre les concepts clés du cours :

#### 1. Virtualisation
- **Hyperviseur Type 1** : SageMaker utilise AWS Nitro System (équivalent KVM)
- **Conteneurisation** : Docker pour isolation et portabilité
- **Isolation des ressources** : Environnements d'entraînement cloisonnés

#### 2. Cloud Computing
- **IaaS** : Infrastructure EC2 sous-jacente
- **PaaS** : SageMaker comme plateforme gérée
- **Caractéristiques du Cloud** :
  - ✅ Élasticité : Scalabilité automatique des ressources
  - ✅ Pay-as-you-go : Facturation à l'usage
  - ✅ Self-service : Provisionnement via console/SDK
  - ✅ Mesurabilité : Monitoring via CloudWatch

#### 3. Stockage Cloud
- **S3 (Object Storage)** : Équivalent AWS de Swift (OpenStack)
- **Redondance** : Données répliquées automatiquement (3+ AZ)
- **Accès programmatique** : API boto3

### Comparaison IaaS vs PaaS

| Tâche | Sans SageMaker (IaaS) | Avec SageMaker (PaaS) |
|-------|----------------------|----------------------|
| Provisioning EC2 | Manuel | Automatique |
| Installation PyTorch | Manuel | Pré-configuré |
| Configuration réseau | Manuel | Automatique |
| Load balancing | Manuel | Intégré |
| Monitoring | Setup CloudWatch | Automatique |
| **Temps de déploiement** | **~2 heures** | **~10 minutes** |

---

## 💰 Coûts du Projet

| Resource | Utilisation | Coût unitaire | Total |
|----------|-------------|---------------|-------|
| SageMaker Studio | 20h | $0.05/h | $1.00 |
| SageMaker Endpoint | 8h | $0.05/h | $0.40 |
| S3 Storage | 500 MB | $0.023/GB/mois | ~$0.01 |
| **Total** | - | - | **~$1.50** |

**Crédits AWS utilisés :** $1.50 / $100  
**Crédits restants :** $98.50 ✅

---

## 👨‍💻 Auteur

**Gninninmaguignon Silué**  
Étudiant en Génie Informatique - Option Logiciel  
ENSAH (École Nationale des Sciences Appliquées d'Al Hoceima)  
Promotion 2025/2026

📧 Email : gninhosilue@gmail.com  
🐙 GitHub : [@Gninho-silue](https://github.com/Gninho-silue)  
💼 LinkedIn : [Votre profil LinkedIn]

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Pr. Routaib Hayat** - Enseignant du module Virtualisation & Cloud Computing
- **GroupLens Research** - Pour le dataset MovieLens
- **AWS Educate** - Pour les crédits AWS gratuits
- **Community PyTorch** - Pour la documentation et les exemples

---

## 📞 Contact

Pour toute question ou suggestion :

📧 Email : gninhosilue@gmail.com  
🐙 GitHub : [@Gninho-silue](https://github.com/Gninho-silue)

---

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !** ⭐

---

*Made with ❤️ and ☕ by Gninninmaguignon Silué*