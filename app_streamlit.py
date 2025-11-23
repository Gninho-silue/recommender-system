"""
=====================================================================
APPLICATION STREAMLIT - SYSTÈME DE RECOMMANDATION
Version Premium - Interface Moderne
Auteur : Gninninmaguignon Silué
Date : Novembre 2025

Installation:
    pip install streamlit torch pandas numpy plotly scikit-learn

Lancement:
    streamlit run app_streamlit.py
=====================================================================
"""

import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🎬 MovieLens AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PREMIUM avec animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Background gradient animé */
    .main {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Header avec effet glassmorphism */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px #667eea); }
        to { filter: drop-shadow(0 0 20px #764ba2); }
    }
    
    .subtitle {
        text-align: center;
        color: #b8b8d1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Carte glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(102, 126, 234, 0.4);
    }
    
    /* Carte de profil avec effet néon */
    .profile-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        color: white;
        margin-bottom: 1.5rem;
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    .profile-card h2 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    
    .profile-card p {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #e0e0ff;
    }
    
    /* Carte de film premium */
    .movie-card {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(178, 7, 16, 0.1) 100%);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 15px;
        border-left: 4px solid #e50914;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        color: white;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .movie-card:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.4);
        border-left-width: 6px;
    }
    
    .movie-card:hover::before {
        left: 100%;
    }
    
    /* Bouton premium avec effet */
    .stButton>button {
        background: linear-gradient(135deg, #e50914 0%, #b20710 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 50px;
        padding: 0.8rem 3rem;
        border: none;
        box-shadow: 0 10px 30px rgba(229, 9, 20, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(229, 9, 20, 0.6);
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    /* Métrique avec effet glassmorphism */
    .metric-premium {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-premium:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b8b8d1;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Expander personnalisé */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Badge de genre */
    .genre-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0ff;
    }
    
    /* Progress bar animé */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradient 2s ease infinite;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b8b8d1;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #b8b8d1;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animations de chargement */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loader {
        border: 4px solid rgba(255, 255, 255, 0.1);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# CHARGEMENT DES DONNÉES ET MODÈLE
# ============================================

@st.cache_resource
def load_model_and_data():
    """Charger le modèle et les données (mis en cache)"""

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

    try:
        checkpoint = torch.load('models/saved_models/best_model.pth',
                                map_location=device, weights_only=False)

        n_users = checkpoint['n_users']
        n_items = checkpoint['n_items']
        n_features = checkpoint['n_features']

        model = HybridRecommenderNet(n_users, n_items, n_features).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        with open('models/encoders/user_encoder.pkl', 'rb') as f:
            user_encoder = pickle.load(f)
        with open('models/encoders/item_encoder.pkl', 'rb') as f:
            item_encoder = pickle.load(f)

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
            'device': device,
            'loaded': True
        }
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return {'loaded': False, 'error': str(e)}


# Charger tout
with st.spinner('🔄 Chargement du système IA...'):
    resources = load_model_and_data()

if not resources['loaded']:
    st.error(f"❌ Impossible de charger les données")
    st.stop()

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
# INTERFACE UTILISATEUR PREMIUM
# ============================================

# Header Premium
st.markdown('<h1 class="main-header">🎬 MovieLens AI Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ Propulsé par Deep Learning & Amazon SageMaker</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Premium
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    all_users = sorted(data_full['user_id'].unique())
    user_id = st.selectbox(
        "🆔 Utilisateur",
        options=all_users,
        index=0
    )

    top_k = st.slider(
        "📊 Recommandations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

    exclude_rated = st.checkbox(
        "🚫 Exclure films notés",
        value=True
    )

    st.markdown("---")

    # Métriques premium
    st.markdown("### 📈 Performances Modèle")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-premium">
            <div class="metric-value">{checkpoint['rmse']:.3f}</div>
            <div class="metric-label">RMSE</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-premium">
            <div class="metric-value">{checkpoint['mae']:.3f}</div>
            <div class="metric-label">MAE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**👨‍💻 Développé par**")
    st.markdown("Gninninmaguignon Silué")
    st.markdown("**🎓 Projet ENSAH 2025**")
    st.markdown("Cloud Computing & ML")

# Main content avec tabs
tab1, tab2 = st.tabs(["🎯 Recommandations", "📊 Statistiques"])

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 👤 Profil Utilisateur")

        profile = get_user_profile(user_id)

        if profile:
            gender_emoji = '👨' if profile['gender'] == 'M' else '👩'
            st.markdown(f"""
            <div class="profile-card">
                <h2>{gender_emoji} Utilisateur #{profile['user_id']}</h2>
                <p>
                    🎂 <strong>{profile['age']} ans</strong><br>
                    💼 <strong>{profile['occupation']}</strong><br>
                    📊 <strong>{profile['n_ratings']} films notés</strong><br>
                    ⭐ <strong>Note moyenne: {profile['avg_rating']:.2f}/5</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Films préférés
            st.markdown("### ❤️ Films Favoris")
            user_ratings = data_full[data_full['user_id'] == user_id].sort_values(
                'rating', ascending=False
            ).head(5)

            for idx, row in user_ratings.iterrows():
                movie = get_movie_info(row['item_id'])
                if movie:
                    genres_badges = ' '.join([f'<span class="genre-badge">{g}</span>' for g in movie['genres'][:3]])
                    st.markdown(f"""
                    <div class="movie-card">
                        <strong style="font-size: 1.1rem;">{movie['title'][:50]}</strong><br>
                        <div style="margin-top: 0.5rem;">
                            ⭐ {row['rating']}/5 | {genres_badges}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"### 🎯 Top {top_k} Recommandations IA")

        if st.button("🚀 Générer les Recommandations", use_container_width=True):
            with st.spinner('🔮 Intelligence Artificielle en action...'):
                recommendations = recommend_top_k(user_id, top_k=top_k, exclude_rated=exclude_rated)

                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"#{i} - {rec['title'][:50]} ⭐ {rec['predicted_rating']:.2f}",
                                         expanded=(i <= 3)):
                            col_a, col_b = st.columns([2, 1])

                            with col_a:
                                genres_badges = ' '.join(
                                    [f'<span class="genre-badge">{g}</span>' for g in rec['genres']])
                                st.markdown(genres_badges, unsafe_allow_html=True)
                                st.markdown(f"**Note moyenne:** {rec['avg_rating']:.2f}/5 ⭐")
                                st.markdown(f"**Popularité:** {rec['n_ratings']} évaluations 📊")

                            with col_b:
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=rec['predicted_rating'],
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [1, 5]},
                                        'bar': {'color': "#667eea"},
                                        'steps': [
                                            {'range': [1, 2.5], 'color': "#ff6b6b"},
                                            {'range': [2.5, 3.5], 'color': "#ffd93d"},
                                            {'range': [3.5, 5], 'color': "#6bcf7f"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "white", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 4.0
                                        }
                                    }
                                ))
                                fig.update_layout(
                                    height=150,
                                    margin=dict(l=5, r=5, t=5, b=5),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font={'color': "white"}
                                )
                                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 📊 Analyse des Recommandations")

    if st.button("📈 Analyser", use_container_width=True):
        with st.spinner('📊 Génération des statistiques...'):
            recommendations = recommend_top_k(user_id, top_k=top_k, exclude_rated=exclude_rated)

            if recommendations:
                col1, col2 = st.columns(2)

                with col1:
                    # Distribution des scores
                    scores = [r['predicted_rating'] for r in recommendations]
                    fig1 = px.histogram(
                        x=scores,
                        nbins=15,
                        title="📊 Distribution des Scores Prédits",
                        labels={'x': 'Score', 'y': 'Fréquence'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig1.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white"}
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    # Genres
                    all_genres = []
                    for rec in recommendations:
                        all_genres.extend(rec['genres'])

                    genre_counts = Counter(all_genres)

                    fig2 = px.bar(
                        x=list(genre_counts.keys()),
                        y=list(genre_counts.values()),
                        title="🎭 Genres Recommandés",
                        labels={'x': 'Genre', 'y': 'Fréquence'},
                        color_discrete_sequence=['#764ba2']
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "white"}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# Footer Premium
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.7;">
        Propulsé par PyTorch 2.6.0 & Amazon SageMaker ☁️
    </p>
     <p style="font-size: 1.1rem; font-weight: 600;">🎓 Projet de Fin de Semestre</p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">Virtualisation & Cloud Computing</p>
    <p style="font-size: 0.85rem; color: #667eea;">ENSAH - Génie Informatique Option Logiciel | 2025/2026</p>
   
</div>
""", unsafe_allow_html=True)
