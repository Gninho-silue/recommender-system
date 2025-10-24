"""
=====================================================================
INTERFACE WEB - SYSTÈME DE RECOMMANDATION MOVIELENS
Projet : Système de Recommandation MovieLens sur Amazon SageMaker
Auteur : Gninninmaguignon Silué
Date : Octobre 2025

Usage:
    streamlit run app.py
=====================================================================
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🎬 MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e50914;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .movie-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #e50914;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #b20710;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CHARGEMENT DES DONNÉES ET MODÈLE
# ============================================

@st.cache_resource
def load_model_and_data():
    """Charger le modèle et les données (mis en cache)"""
    
    # Architecture du modèle
    class HybridRecommenderNet(nn.Module):
        def __init__(self, n_users, n_items, n_features, 
                     embedding_dim=128, hidden_dims=[256, 128, 64]):
            super(HybridRecommenderNet, self).__init__()
            
            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)
            self.user_bn = nn.BatchNorm1d(embedding_dim)
            self.item_bn = nn.BatchNorm1d(embedding_dim)
            
            self.feature_fc = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(64)
            )
            
            total_input = embedding_dim * 2 + 64
            layers = []
            input_dim = total_input
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.BatchNorm1d(hidden_dim)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, 1))
            self.fc_layers = nn.Sequential(*layers)
        
        def forward(self, user, item, features):
            user_emb = self.user_embedding(user)
            item_emb = self.item_embedding(item)
            user_emb = self.user_bn(user_emb)
            item_emb = self.item_bn(item_emb)
            feat_emb = self.feature_fc(features)
            x = torch.cat([user_emb, item_emb, feat_emb], dim=1)
            output = self.fc_layers(x)
            return output.squeeze()
    
    device = torch.device('cpu')
    
    # Charger le checkpoint
    checkpoint = torch.load('models/saved_models/best_model.pth', 
                            map_location=device)
    
    n_users = checkpoint['n_users']
    n_items = checkpoint['n_items']
    n_features = checkpoint['n_features']
    
    # Créer et charger le modèle
    model = HybridRecommenderNet(n_users, n_items, n_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les encoders
    with open('models/encoders/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('models/encoders/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    
    # Charger les métadonnées
    movies_meta = pd.read_csv("data/processed/movies_metadata.csv")
    users_meta = pd.read_csv("data/processed/users_metadata.csv")
    data_full = pd.read_csv("data/processed/train_features.csv")
    
    return {
        'model': model,
        'checkpoint': checkpoint,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'movies_meta': movies_meta,
        'users_meta': users_meta,
        'data_full': data_full,
        'n_users': n_users,
        'n_items': n_items,
        'n_features': n_features,
        'device': device
    }

# Charger tout
with st.spinner('🔄 Chargement du modèle et des données...'):
    resources = load_model_and_data()

model = resources['model']
checkpoint = resources['checkpoint']
user_encoder = resources['user_encoder']
item_encoder = resources['item_encoder']
movies_meta = resources['movies_meta']
users_meta = resources['users_meta']
data_full = resources['data_full']
n_users = resources['n_users']
n_items = resources['n_items']
n_features = resources['n_features']
device = resources['device']

# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def get_user_profile(user_id_original):
    """Récupérer le profil d'un utilisateur"""
    user_data = users_meta[users_meta['user_id'] == user_id_original]
    if len(user_data) == 0:
        return None
    
    user_info = user_data.iloc[0]
    user_ratings = data_full[data_full['user_id'] == user_id_original]
    
    return {
        'user_id': int(user_id_original),
        'age': int(user_info['age']),
        'gender': user_info['gender'],
        'occupation': user_info['occupation'],
        'n_ratings': len(user_ratings),
        'avg_rating': float(user_ratings['rating'].mean()) if len(user_ratings) > 0 else 0
    }

def get_movie_info(item_id):
    """Récupérer les informations d'un film"""
    movie = movies_meta[movies_meta['item_id'] == item_id]
    if len(movie) == 0:
        return None
    
    movie_info = movie.iloc[0]
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
    
    genres = [col for col in genre_cols if movie_info[col] == 1]
    item_ratings = data_full[data_full['item_id'] == item_id]
    
    return {
        'item_id': int(item_id),
        'title': movie_info['title'],
        'genres': genres,
        'n_ratings': len(item_ratings),
        'avg_rating': float(item_ratings['rating'].mean()) if len(item_ratings) > 0 else 0
    }

def recommend_top_k(user_id_original, top_k=10, exclude_rated=True):
    """Recommander les top-K films"""
    if user_id_original not in user_encoder.classes_:
        return None
    
    user_id_encoded = user_encoder.transform([user_id_original])[0]
    
    user_tensor = torch.tensor([user_id_encoded] * n_items, 
                               dtype=torch.long).to(device)
    item_tensor = torch.arange(n_items, dtype=torch.long).to(device)
    
    # Features simplifiées
    user_data = data_full[data_full['user_id'] == user_id_original]
    if len(user_data) > 0:
        feature_cols = checkpoint['feature_cols']
        user_features_mean = user_data[feature_cols].mean().values
        features = np.tile(user_features_mean, (n_items, 1))
    else:
        features = np.zeros((n_items, n_features))
    
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor, features_tensor)
        predictions = predictions.cpu().numpy()
    
    item_ids_original = item_encoder.inverse_transform(range(n_items))
    recommendations_df = pd.DataFrame({
        'item_id': item_ids_original,
        'predicted_rating': predictions
    })
    
    if exclude_rated:
        rated_items = data_full[data_full['user_id'] == user_id_original]['item_id'].values
        recommendations_df = recommendations_df[~recommendations_df['item_id'].isin(rated_items)]
    
    recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False)
    top_recommendations = recommendations_df.head(top_k)
    
    recommendations = []
    for _, row in top_recommendations.iterrows():
        movie_info = get_movie_info(row['item_id'])
        if movie_info:
            movie_info['predicted_rating'] = float(row['predicted_rating'])
            recommendations.append(movie_info)
    
    return recommendations

# ============================================
# INTERFACE UTILISATEUR
# ============================================

# Header
st.markdown('<h1 class="main-header">🎬 MovieLens Recommender System</h1>', 
            unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=100)
    st.title("⚙️ Configuration")
    
    # Sélection de l'utilisateur
    all_users = sorted(data_full['user_id'].unique())
    user_id = st.selectbox(
        "🆔 Sélectionner un utilisateur",
        options=all_users,
        index=0
    )
    
    # Nombre de recommandations
    top_k = st.slider(
        "📊 Nombre de recommandations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # Exclure films notés
    exclude_rated = st.checkbox(
        "🚫 Exclure les films déjà notés",
        value=True
    )
    
    st.markdown("---")
    
    # Informations du modèle
    st.subheader("📈 Performances du Modèle")
    st.metric("RMSE", f"{checkpoint['rmse']:.4f}")
    st.metric("MAE", f"{checkpoint['mae']:.4f}")
    
    st.markdown("---")
    st.markdown("**Développé par:** Gninninmaguignon Silué")
    st.markdown("**Projet:** Cloud Computing & ML")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("👤 Profil Utilisateur")
    
    profile = get_user_profile(user_id)
    
    if profile:
        st.markdown(f"""
        <div class="metric-card">
            <h2>Utilisateur #{profile['user_id']}</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                👤 {profile['age']} ans | {'👨' if profile['gender'] == 'M' else '👩'} {profile['gender']}
            </p>
            <p style="font-size: 1rem;">
                💼 {profile['occupation']}<br>
                📊 {profile['n_ratings']} films notés<br>
                ⭐ Note moyenne: {profile['avg_rating']:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Films préférés
        st.markdown("### ❤️ Films Préférés")
        user_ratings = data_full[data_full['user_id'] == user_id].sort_values(
            'rating', ascending=False
        ).head(5)
        
        for idx, row in user_ratings.iterrows():
            movie = get_movie_info(row['item_id'])
            if movie:
                st.markdown(f"""
                <div class="movie-card">
                    <strong>{movie['title']}</strong><br>
                    ⭐ {row['rating']}/5 | 🎭 {', '.join(movie['genres'][:2])}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("❌ Utilisateur introuvable")

with col2:
    st.subheader(f"🎯 Top {top_k} Recommandations")
    
    # Bouton de génération
    if st.button("🚀 Générer les Recommandations", use_container_width=True):
        with st.spinner('🔮 Analyse en cours...'):
            recommendations = recommend_top_k(user_id, top_k=top_k, 
                                             exclude_rated=exclude_rated)
            
            if recommendations:
                # Afficher les recommandations
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"#{i} - {rec['title']} ⭐ {rec['predicted_rating']:.2f}"):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**Genres:** {', '.join(rec['genres'])}")
                            st.markdown(f"**Note moyenne réelle:** {rec['avg_rating']:.2f}/5")
                            st.markdown(f"**Nombre d'évaluations:** {rec['n_ratings']}")
                        
                        with col_b:
                            # Jauge de score
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=rec['predicted_rating'],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [1, 5]},
                                    'bar': {'color': "darkred"},
                                    'steps': [
                                        {'range': [1, 3], 'color': "lightgray"},
                                        {'range': [3, 4], 'color': "gray"},
                                        {'range': [4, 5], 'color': "lightgreen"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)
                
                # Graphique de distribution
                st.markdown("---")
                st.subheader("📊 Distribution des Scores Prédits")
                
                scores = [r['predicted_rating'] for r in recommendations]
                fig = px.histogram(
                    x=scores,
                    nbins=10,
                    title="Distribution des Scores",
                    labels={'x': 'Score Prédit', 'y': 'Nombre de Films'},
                    color_discrete_sequence=['#e50914']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Graphique des genres
                all_genres = []
                for rec in recommendations:
                    all_genres.extend(rec['genres'])
                
                from collections import Counter
                genre_counts = Counter(all_genres)
                
                fig = px.bar(
                    x=list(genre_counts.keys()),
                    y=list(genre_counts.values()),
                    title="Genres Recommandés",
                    labels={'x': 'Genre', 'y': 'Fréquence'},
                    color_discrete_sequence=['#764ba2']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Impossible de générer des recommandations")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>🎓 Projet de fin de semestre - Virtualisation & Cloud Computing</p>
    <p>ENSAH - Génie Informatique Option Logiciel | 2025/2026</p>
</div>
""", unsafe_allow_html=True)