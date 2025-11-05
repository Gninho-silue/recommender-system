"""
Script d'inférence OPTIMISÉ pour SageMaker Endpoint
MovieLens Recommender System - Version Fast
"""

import torch
import torch.nn as nn
import json
import os
import numpy as np
import pickle


class HybridRecommenderNet(nn.Module):
    """Modèle hybride - Architecture identique"""
    def __init__(self, n_users, n_items, n_features, embedding_dim=128):
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
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, user, item, features):
        u_emb = self.user_bn(self.user_embedding(user))
        i_emb = self.item_bn(self.item_embedding(item))
        f_emb = self.feature_fc(features)
        
        x = torch.cat([u_emb, i_emb, f_emb], dim=1)
        return self.fc_layers(x).squeeze()


def model_fn(model_dir):
    """
    Charge le modèle - OPTIMISÉ avec mise en cache
    """
    print("🔄 [FAST] Chargement du modèle...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger config
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    # Charger encoders
    with open(os.path.join(model_dir, 'user_encoder.pkl'), 'rb') as f:
        user_encoder = pickle.load(f)
    
    with open(os.path.join(model_dir, 'item_encoder.pkl'), 'rb') as f:
        item_encoder = pickle.load(f)
    
    # Initialiser le modèle
    model = HybridRecommenderNet(
        n_users=config['n_users'],
        n_items=config['n_items'],
        n_features=config['n_features'],
        embedding_dim=128
    )
    
    # Charger les poids
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, 'model.pth'),
            map_location=device
        )
    )
    
    model.to(device)
    model.eval()
    
    # 🚀 OPTIMISATION : Précalculer les embeddings des items
    print("🚀 [FAST] Précalcul des embeddings items...")
    n_items = config['n_items']
    item_tensor = torch.arange(n_items, dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Embeddings pré-calculés pour tous les items
        item_embeddings = model.item_bn(model.item_embedding(item_tensor))
    
    print("✅ [FAST] Modèle chargé avec précalcul!")
    
    return {
        'model': model,
        'device': device,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'config': config,
        'item_embeddings': item_embeddings  # 🚀 Cache
    }


def input_fn(request_body, content_type):
    """Parse les données d'entrée"""
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError(f"Content type non supporté: {content_type}")


def predict_fn(input_data, model_dict):
    """
    Prédiction OPTIMISÉE - Batch processing
    """
    import time
    start = time.time()
    
    model = model_dict['model']
    device = model_dict['device']
    user_encoder = model_dict['user_encoder']
    item_encoder = model_dict['item_encoder']
    config = model_dict['config']
    item_embeddings = model_dict['item_embeddings']  # 🚀 Cache
    
    user_id = input_data['user_id']
    top_k = input_data.get('top_k', 10)
    
    try:
        user_encoded = user_encoder.transform([user_id])[0]
    except ValueError:
        return {
            'error': f'User ID {user_id} inconnu',
            'valid_range': f'1-943'
        }
    
    # 🚀 OPTIMISATION : Traiter par batches de 256 items
    n_items = config['n_items']
    batch_size = 256
    all_predictions = []
    
    with torch.no_grad():
        # User embedding (calculé une seule fois)
        user_tensor_single = torch.tensor([user_encoded], dtype=torch.long).to(device)
        user_emb = model.user_bn(model.user_embedding(user_tensor_single))
        
        # Traiter par batches
        for start_idx in range(0, n_items, batch_size):
            end_idx = min(start_idx + batch_size, n_items)
            batch_size_actual = end_idx - start_idx
            
            # User embeddings répétés pour le batch
            user_emb_batch = user_emb.repeat(batch_size_actual, 1)
            
            # Item embeddings du batch (depuis le cache)
            item_emb_batch = item_embeddings[start_idx:end_idx]
            
            # Features (simplifiées)
            features_batch = torch.zeros(
                batch_size_actual, 
                config['n_features'], 
                dtype=torch.float32
            ).to(device)
            
            # Forward pass pour features
            f_emb = model.feature_fc(features_batch)
            
            # Concatenation
            x = torch.cat([user_emb_batch, item_emb_batch, f_emb], dim=1)
            
            # Prédiction
            batch_predictions = model.fc_layers(x).squeeze()
            all_predictions.append(batch_predictions.cpu().numpy())
        
        # Combiner tous les batches
        predictions = np.concatenate(all_predictions)
    
    # Top-K
    top_k_indices = np.argsort(predictions)[::-1][:top_k]
    top_k_items = item_encoder.inverse_transform(top_k_indices)
    top_k_scores = predictions[top_k_indices]
    
    elapsed = time.time() - start
    
    result = {
        'user_id': int(user_id),
        'top_k': int(top_k),
        'inference_time_ms': round(elapsed * 1000, 2),
        'recommendations': [
            {
                'rank': i + 1,
                'item_id': int(item),
                'predicted_rating': round(float(score), 4)
            }
            for i, (item, score) in enumerate(zip(top_k_items, top_k_scores))
        ]
    }
    
    print(f"✅ [FAST] Prédiction en {elapsed*1000:.0f}ms")
    return result


def output_fn(prediction, accept):
    """Sérialise la sortie"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Accept type non supporté: {accept}")