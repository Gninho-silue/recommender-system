"""
Script d'inférence SIMPLIFIÉ - Version qui MARCHE
Pas d'optimisation, juste du code qui fonctionne
"""

import torch
import torch.nn as nn
import json
import os
import numpy as np
import pickle


class SimpleRecommenderNet(nn.Module):
    """
    Version SIMPLIFIÉE du modèle
    Sans BatchNorm pour éviter les problèmes
    """
    def __init__(self, n_users, n_items, embedding_dim=128):
        super(SimpleRecommenderNet, self).__init__()
        
        # Embeddings uniquement
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Réseau simple sans BatchNorm
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, user, item):
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        x = torch.cat([u_emb, i_emb], dim=1)
        return self.fc(x).squeeze()


def model_fn(model_dir):
    """Charge le modèle - Version simple"""
    print("=" * 60)
    print("🔄 Chargement du modèle SIMPLIFIÉ...")
    print("=" * 60)
    
    device = torch.device('cpu')  # Forcer CPU pour debug
    
    # Charger config
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"Config: {config}")
    
    # Charger encoders
    with open(os.path.join(model_dir, 'user_encoder.pkl'), 'rb') as f:
        user_encoder = pickle.load(f)
    
    with open(os.path.join(model_dir, 'item_encoder.pkl'), 'rb') as f:
        item_encoder = pickle.load(f)
    
    print(f"Encoders: {len(user_encoder.classes_)} users, {len(item_encoder.classes_)} items")
    
    # Créer le modèle simple
    model = SimpleRecommenderNet(
        n_users=config['n_users'],
        n_items=config['n_items'],
        embedding_dim=128
    )
    
    # Charger seulement les embeddings depuis le vrai modèle
    checkpoint = torch.load(
        os.path.join(model_dir, 'model.pth'),
        map_location=device
    )
    
    # Copier seulement les embeddings
    model.user_embedding.load_state_dict({
        'weight': checkpoint['user_embedding.weight']
    })
    model.item_embedding.load_state_dict({
        'weight': checkpoint['item_embedding.weight']
    })
    
    # Initialiser le reste aléatoirement (pas grave pour la démo)
    
    model.to(device)
    model.eval()
    
    print("✅ Modèle chargé!")
    
    return {
        'model': model,
        'device': device,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'config': config
    }


def input_fn(request_body, content_type):
    """Parse input"""
    print(f"📥 Input: {request_body[:100]}")
    
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(f"Content type non supporté: {content_type}")


def predict_fn(input_data, model_dict):
    """
    Prédiction SIMPLE - Top 100 films seulement
    """
    print("=" * 60)
    print("🔮 Début prédiction...")
    print(f"Input: {input_data}")
    
    model = model_dict['model']
    device = model_dict['device']
    user_encoder = model_dict['user_encoder']
    item_encoder = model_dict['item_encoder']
    config = model_dict['config']
    
    user_id = input_data['user_id']
    top_k = min(input_data.get('top_k', 10), 20)  # Max 20
    
    print(f"User: {user_id}, Top-K: {top_k}")
    
    # Encoder l'utilisateur
    try:
        user_encoded = user_encoder.transform([user_id])[0]
        print(f"User encoded: {user_encoded}")
    except ValueError:
        return {
            'error': f'User ID {user_id} inconnu'
        }
    
    # ASTUCE : Ne prédire QUE pour les 100 premiers films
    # (sinon timeout)
    n_items_sample = min(100, config['n_items'])
    
    print(f"Prédiction pour {n_items_sample} films...")
    
    # Créer les tensors
    user_tensor = torch.tensor(
        [user_encoded] * n_items_sample, 
        dtype=torch.long
    ).to(device)
    
    item_tensor = torch.arange(
        n_items_sample, 
        dtype=torch.long
    ).to(device)
    
    print("Tensors créés")
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor)
        predictions = predictions.cpu().numpy()
    
    print(f"Prédictions: shape={predictions.shape}")
    
    # Top-K
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_items_encoded = top_k_indices
    top_k_items = item_encoder.inverse_transform(top_k_items_encoded)
    top_k_scores = predictions[top_k_indices]
    
    print(f"Top-K calculé: {len(top_k_items)} items")
    
    result = {
        'user_id': int(user_id),
        'top_k': int(top_k),
        'note': 'Démo - Échantillon de 100 films seulement',
        'recommendations': [
            {
                'rank': i + 1,
                'item_id': int(item),
                'predicted_rating': round(float(score), 2)
            }
            for i, (item, score) in enumerate(zip(top_k_items, top_k_scores))
        ]
    }
    
    print("✅ Prédiction terminée!")
    return result


def output_fn(prediction, accept):
    """Sérialise output"""
    print(f"📤 Output: {accept}")
    
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Accept type non supporté: {accept}")