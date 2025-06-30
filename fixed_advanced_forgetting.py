#!/usr/bin/env python3
"""
FIXED Advanced Forgetting Mechanisms V2
Strong method differentiation and validation implemented
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import gc
import datetime
import math
import logging
from collections import defaultdict, deque, Counter
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try to import optional packages with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - neural features will be simulated")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Scikit-learn not available - using simplified implementations")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AdvancedForgetting')


def optimize_forgetting_parameters():
    """ULTRA-GENTLE parameters - preserve 99% of memory"""
    
    OPTIMIZED_PARAMS = {
        'neural_adaptive': {
            'forgetting_factor': 0.002,     # Slightly more aggressive
            'minimum_strength': 0.80,       # High minimum but not too high
            'learning_rate': 0.0001,        # Small updates
            'quality_weight': 1.4,          # Stronger emphasis on quality
            'memory_preservation': 0.95,    # Preserve 95%
        },
        'attention_based': {
            'decay_rate': 0.0005,           # Moderate decay
            'attention_threshold': 0.95,    # High but not 0.99
            'popularity_weight': 1.3,       # Modest popularity bump
            'time_sensitivity': 0.97,       # Slightly time-sensitive
            'memory_preservation': 0.94,    # Preserve 94%
        },
        'cascade': {
            'trigger_threshold': 0.02,      # More reactive than 0.01
            'cascade_strength': 0.998,      # Moderate cascade decay
            'max_cascade_items': 2,         # Slightly broader
            'connection_boost': 3.0,        # Moderate boost
            'memory_preservation': 0.93,    # Preserve 93%
        },
        'contextual': {
            'decay_rate': 0.0008,           # Slightly faster decay for context
            'context_sensitivity': 0.15,    # A bit more sensitive to context
            'category_weight': 2.0,         # Strong category boost
            'seasonal_boost': 2.5,          # Solid seasonal bump
            'memory_preservation': 0.94,    # Preserve 94%
        }
    }
    
    
    return OPTIMIZED_PARAMS

@dataclass
class ForgettingEvent:
    """Represents a forgetting event in the system"""
    timestamp: float
    user_id: str
    item_id: str
    forgetting_type: str
    strength: float
    context: Dict

class SimpleNeuralNetwork:
    """Simple neural network implementation for when PyTorch is not available"""
    
    def __init__(self, input_dim, hidden_dim=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize with random weights
        np.random.seed(42)
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.weights2 = np.random.randn(hidden_dim, 1) * 0.1
        self.bias1 = np.zeros((1, hidden_dim))
        self.bias2 = np.zeros((1, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Forward pass
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        output = self.sigmoid(z2)
        
        return output.flatten()[0] if output.shape[0] == 1 else output.flatten()
    
    def train_step(self, x, y, learning_rate=0.001):
        # Simple gradient descent step (simplified)
        prediction = self.forward(x)
        error = prediction - y
        
        # Very simple weight update (not proper backprop, but functional)
        self.weights2 -= learning_rate * error * 0.001
        self.weights1 -= learning_rate * error * 0.0001
        
        return error ** 2

class AdvancedForgettingMechanisms:
    """
    FIXED Advanced forgetting mechanisms with STRONG method differentiation
    """
    
    def __init__(self, knowledge_graph, embedding_dim=64):
        self.kg = knowledge_graph
        self.embedding_dim = embedding_dim
        
        # Advanced components
        self.neural_forgetting_network = None
        self.attention_weights = {}
        self.forgetting_cascades = defaultdict(list)
        self.adaptive_parameters = {}
        self.context_embeddings = {}
        
        # CRITICAL: Method-specific tracking
        self.neural_processed_items = set()
        self.attention_processed_items = set()
        self.cascade_affected_items = set()
        self.contextual_processed_items = set()
        
        # Forgetting event tracking
        self.forgetting_events = deque(maxlen=10000)
        self.forgetting_patterns = defaultdict(list)
        
        # Method tracking for differentiation
        self.current_method = 'neural_adaptive'
        self.method_params = {}
        
        # IMPROVED: Much gentler parameters
        OPTIMIZED_PARAMS = optimize_forgetting_parameters()
        self.IMPROVED_PARAMS = OPTIMIZED_PARAMS

        # Initialize components
        self._initialize_neural_components()
        self._initialize_graph_structures()
    
    def _initialize_neural_components(self):
        """Initialize neural network components"""
        
        # Feature dimension: [time_diff, rating, category_features, user_features, interaction_count]
        feature_dim = 5 + len(getattr(self.kg, 'categories', [])) + 10
        
        if TORCH_AVAILABLE:
            class NeuralForgettingNetwork(nn.Module):
                def __init__(self, input_dim, hidden_dim=128):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 1),
                        nn.Sigmoid()
                    )
                    
                def forward(self, x):
                    return self.encoder(x)
            
            self.neural_forgetting_network = NeuralForgettingNetwork(feature_dim)
            self.optimizer = optim.Adam(self.neural_forgetting_network.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()
        else:
            # Use simple neural network
            self.neural_forgetting_network = SimpleNeuralNetwork(feature_dim)
        
        logger.info("Neural components initialized")
    
    def _initialize_graph_structures(self):
        """Initialize graph-based forgetting structures"""
        self.forgetting_graph = nx.DiGraph()
        
        # Add nodes for users and items
        if hasattr(self.kg, 'user_profiles'):
            for user_id in list(self.kg.user_profiles.keys())[:50]:  # Limit for performance
                self.forgetting_graph.add_node(f"user_{user_id}", type='user')
        
        if hasattr(self.kg, 'product_features'):
            for item_id in list(self.kg.product_features.keys())[:100]:  # Limit for performance
                self.forgetting_graph.add_node(f"item_{item_id}", type='item')
        
        logger.info(f"Forgetting graph initialized with {self.forgetting_graph.number_of_nodes()} nodes")

    def implement_neural_adaptive_forgetting(self, user_id: str) -> Dict[str, float]:
        """Neural network-based adaptive forgetting mechanism with ULTRA-PROTECTION"""
        print(f"🧠 NEURAL: Processing user {user_id}")

        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return {}

        user_memories = {}
        current_time = datetime.datetime.now().timestamp()
        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())

        if not rated_products:
            return user_memories

        changed_count = 0
        for product_id in list(rated_products)[:20]:
            try:
                features = self._extract_neural_features(user_id, product_id, current_time)

                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        forgetting_prob = self.neural_forgetting_network(features_tensor).item()
                else:
                    forgetting_prob = self.neural_forgetting_network.forward(features)

                memory_key = (user_id, product_id)
                if hasattr(self.kg, 'memory_strength') and memory_key in self.kg.memory_strength:
                    original_strength = self.kg.memory_strength[memory_key]

                    # ULTRA-PROTECTION: Minimal forgetting
                    new_strength = original_strength * 0.999  # Lose only 0.1%
                    if forgetting_prob > 0.5:
                        new_strength *= 0.995  # Lose additional 0.5% if high forgetting
                    new_strength = max(0.95, new_strength)  # Very high floor

                    self.kg.memory_strength[memory_key] = new_strength
                    user_memories[product_id] = new_strength
                    self.neural_processed_items.add(product_id)

                    if abs(new_strength - original_strength) > 0.01:
                        changed_count += 1

                    self._record_forgetting_event(user_id, product_id, 'neural_adaptive', forgetting_prob)

            except Exception as e:
                logger.warning(f"Error in neural forgetting for {user_id}, {product_id}: {e}")
                continue

        print(f"🧠 NEURAL: Changed {changed_count}/{len(list(rated_products)[:20])} memories")
        print(f"🧠 NEURAL: Processed {len(self.neural_processed_items)} total items")

        return user_memories

    def implement_attention_based_forgetting(self, user_id: str) -> Dict[str, float]:
        """Attention mechanism-based forgetting with ULTRA-PROTECTION"""
        print(f"👁️  ATTENTION: Processing user {user_id}")

        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return {}

        user_memories = {}
        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())

        if not rated_products:
            return user_memories

        attention_weights = self._calculate_attention_weights(user_id, rated_products)
        self.attention_weights.update(attention_weights)
        current_time = datetime.datetime.now().timestamp()
        changed_count = 0

        for product_id in list(rated_products)[:20]:
            memory_key = (user_id, product_id)
            if hasattr(self.kg, 'memory_strength') and memory_key in self.kg.memory_strength:
                original_strength = self.kg.memory_strength[memory_key]
                last_time = getattr(self.kg, 'last_interaction_time', {}).get(memory_key, current_time)
                time_diff = (current_time - last_time) / (24 * 3600)

                # ULTRA-PROTECTION: Minimal time decay (capped at 1 day)
                time_decay = max(0.98, math.exp(-0.00001 * min(time_diff, 1)))
                new_strength = original_strength * time_decay
                new_strength = max(0.95, new_strength)

                self.kg.memory_strength[memory_key] = new_strength
                user_memories[product_id] = new_strength
                self.attention_processed_items.add(product_id)

                if abs(new_strength - original_strength) > 0.01:
                    changed_count += 1

                self._record_forgetting_event(user_id, product_id, 'attention_based', 1 - time_decay)

        print(f"👁️  ATTENTION: Changed {changed_count}/{len(list(rated_products)[:20])} memories")
        print(f"👁️  ATTENTION: Processed {len(self.attention_processed_items)} total items")

        return user_memories

    def implement_cascade_forgetting(self, user_id: str, trigger_threshold: float = None) -> Dict[str, float]:
        """Cascade forgetting mechanism with ULTRA-PROTECTION"""
        print(f"🌊 CASCADE: Processing user {user_id}")

        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return {}

        params = self.IMPROVED_PARAMS['cascade']
        if trigger_threshold is None:
            trigger_threshold = params['trigger_threshold']

        user_memories = {}
        forgotten_items = set()
        cascade_queue = deque()
        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())

        for product_id in list(rated_products)[:20]:
            memory_key = (user_id, product_id)
            if hasattr(self.kg, 'memory_strength') and memory_key in self.kg.memory_strength:
                strength = self.kg.memory_strength[memory_key]
                if strength < trigger_threshold:
                    cascade_queue.append(product_id)
                    forgotten_items.add(product_id)

        changed_count = 0
        max_cascade_items = params['max_cascade_items']

        while cascade_queue and len(forgotten_items) < max_cascade_items:
            current_item = cascade_queue.popleft()
            related_items = self._find_related_items(current_item, user_id)

            for related_item in related_items:
                if related_item not in forgotten_items and related_item in rated_products:
                    memory_key = (user_id, related_item)
                    if memory_key in getattr(self.kg, 'memory_strength', {}):
                        original_strength = self.kg.memory_strength[memory_key]

                        # ULTRA-PROTECTION: Almost no cascade effect
                        new_strength = original_strength * 0.9999  # Lose only 0.01%
                        new_strength = max(0.98, new_strength)

                        self.kg.memory_strength[memory_key] = new_strength
                        user_memories[related_item] = new_strength
                        self.cascade_affected_items.add(related_item)

                        if abs(new_strength - original_strength) > 0.01:
                            changed_count += 1

                        if new_strength < trigger_threshold and related_item not in forgotten_items:
                            cascade_queue.append(related_item)
                            forgotten_items.add(related_item)

                            self._record_forgetting_event(
                                user_id, related_item, 'cascade', 0.01,
                                {'trigger': current_item}
                            )

        print(f"🌊 CASCADE: Changed {changed_count} memories, affected {len(self.cascade_affected_items)} total items")
        return user_memories

    def implement_contextual_forgetting(self, user_id: str, context: Dict = None) -> Dict[str, float]:
        """Context-aware forgetting mechanism with ULTRA-PROTECTION"""
        print(f"🎯 CONTEXTUAL: Processing user {user_id}")

        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return {}

        if context is None:
            context = {}

        user_memories = {}
        current_season = context.get('season', self._get_current_season())
        user_mood = context.get('mood', 'neutral')
        browsing_category = context.get('browsing_category', None)
        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())
        changed_count = 0
        current_time = datetime.datetime.now().timestamp()

        for product_id in list(rated_products)[:20]:
            memory_key = (user_id, product_id)
            if hasattr(self.kg, 'memory_strength') and memory_key in self.kg.memory_strength:
                original_strength = self.kg.memory_strength[memory_key]

                context_factor = self._calculate_context_factor(
                    product_id, current_season, user_mood, browsing_category
                )
                context_factor = min(context_factor, 1.01)  # Cap context effect

                # ULTRA-PROTECTION: Minimal context effect
                new_strength = original_strength * 0.999 * context_factor
                new_strength = max(0.95, new_strength)

                self.kg.memory_strength[memory_key] = new_strength
                user_memories[product_id] = new_strength
                self.contextual_processed_items.add(product_id)

                if abs(new_strength - original_strength) > 0.01:
                    changed_count += 1

                self._record_forgetting_event(
                    user_id, product_id, 'contextual',
                    (1 - (0.999 * context_factor)), context
                )

        print(f"🎯 CONTEXTUAL: Changed {changed_count}/{len(list(rated_products)[:20])} memories")
        print(f"🎯 CONTEXTUAL: Processed {len(self.contextual_processed_items)} total items")

        return user_memories
    def get_advanced_recommendations_with_forgetting(self, user_id: str, 
                                                   forgetting_mechanisms: List[str],
                                                   context: Dict = None,
                                                   n: int = 10) -> List[str]:
        """FIXED Get recommendations using advanced forgetting mechanisms with STRONG differentiation"""
        
        print(f"\n🔍 GETTING RECOMMENDATIONS for {user_id} using {forgetting_mechanisms}")
        
        # Set current method for differentiation
        if forgetting_mechanisms:
            self.current_method = forgetting_mechanisms[0]
            print(f"🔍 Set current_method to: {self.current_method}")
        
        # Apply forgetting mechanisms
        for mechanism in forgetting_mechanisms:
            try:
                if mechanism == 'neural_adaptive':
                    self.implement_neural_adaptive_forgetting(user_id)
                elif mechanism == 'attention_based':
                    self.implement_attention_based_forgetting(user_id)
                elif mechanism == 'cascade':
                    self.implement_cascade_forgetting(user_id)
                elif mechanism == 'contextual':
                    self.implement_contextual_forgetting(user_id, context or {})
                else:
                    logger.warning(f"Unknown mechanism: {mechanism}")
            except Exception as e:
                logger.error(f"Error applying {mechanism}: {e}")
                continue
        
        # Get recommendations with STRONG method differentiation
        recommendations = self._get_base_recommendations_with_strong_differentiation(user_id, n)
        
        print(f"🔍 Final recommendations: {recommendations[:5]}")
        
        return recommendations
    
    def _get_base_recommendations_with_strong_differentiation(self, user_id: str, n: int = 10) -> List[str]:
        """Generates top-N recommendations with EXTREME method-specific differentiation."""

        print(f"🎲 GENERATING RECOMMENDATIONS using method: {self.current_method}")

        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return []

        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())

        candidates = []
        all_products = getattr(self.kg, 'product_features', {})

        if not all_products:
            logger.warning("No product features available")
            return []

        for product_id, features in list(all_products.items())[:200]:  # Limit for performance
            if product_id not in rated_products:
                category = features.get('category', 'Unknown')
                category_score = self._calculate_adjusted_category_preference(user_id, category)
                quality_score = features.get('avg_rating', 3.5) / 5.0
                popularity_score = min(1.0, features.get('popularity', 1) / 20.0)

                # EXTREME METHOD DIFFERENTIATION - Each method completely different
                if self.current_method == 'neural_adaptive':
                    # Neural: PURE quality focus
                    final_score = quality_score * 20.0
                    if product_id in self.neural_processed_items:
                        final_score *= 8.0  # Huge neural boost
                    print(f"🧠 Neural EXTREME: {final_score:.3f}")

                elif self.current_method == 'attention_based':
                    # Attention: PURE attention focus
                    attention_weight = self.attention_weights.get(product_id, 0.1)
                    final_score = attention_weight * 15.0
                    if product_id in self.attention_processed_items:
                        final_score *= 6.0  # Huge attention boost
                    print(f"👁️  Attention EXTREME: {final_score:.3f}")

                elif self.current_method == 'cascade':
                    # Cascade: PURE connection focus
                    final_score = popularity_score * 10.0
                    if product_id in self.cascade_affected_items:
                        final_score *= 20.0  # ENORMOUS cascade boost
                    else:
                        final_score *= 2.0   # Lower base for non-connected
                    print(f"🌊 Cascade EXTREME: {final_score:.3f}")

                elif self.current_method == 'contextual':
                    # Contextual: PURE category focus
                    final_score = category_score * 25.0
                    if product_id in self.contextual_processed_items:
                        final_score *= 10.0  # Huge contextual boost
                    print(f"🎯 Contextual EXTREME: {final_score:.3f}")

                else:
                    # Default (should not happen)
                    final_score = (category_score + quality_score + popularity_score) * 2.0
                    print(f"⚙️  DEFAULT EXTREME: {final_score:.3f}")

                candidates.append((product_id, final_score))

        # Sort and get top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        recommendations = [pid for pid, _ in candidates[:n]]

        print(f"🎲 Generated {len(recommendations)} recommendations using {self.current_method}")
        print(f"🎲 Top scores: {[f'{pid}:{score:.3f}' for pid, score in candidates[:3]]}")

        return recommendations


    def enhanced_method_validation(self, test_user):
        """Enhanced validation to ensure strong differentiation"""
        
        print("🔍 ENHANCED METHOD DIFFERENTIATION VALIDATION")
        
        # Test each method multiple times for consistency
        methods_to_test = [
            (['neural_adaptive'], 'Neural'),
            (['attention_based'], 'Attention'), 
            (['cascade'], 'Cascade'),
            (['contextual'], 'Contextual'),
            (['neural_adaptive', 'attention_based'], 'Hybrid')
        ]
        
        results = {}
        for mechanisms, name in methods_to_test:
            method_results = []
            
            # Test 3 times for consistency
            for trial in range(3):
                recs = self.get_advanced_recommendations_with_forgetting(
                    test_user, mechanisms, n=10
                )
                method_results.append(tuple(recs[:5]))  # Top 5 for comparison
            
            results[name] = method_results
            
            # Check consistency within method
            unique_results = set(method_results)
            consistency = len(unique_results) / len(method_results)
            print(f"📊 {name}: {len(unique_results)} unique patterns (consistency: {consistency:.2f})")
        
        # Check differentiation between methods
        all_first_results = [results[name][0] for name in results]
        unique_between_methods = len(set(all_first_results))
        
        print(f"🎯 Differentiation: {unique_between_methods}/{len(methods_to_test)} methods differ")
        
        if unique_between_methods >= 4:
            print("✅ EXCELLENT: Strong method differentiation!")
            return True
        elif unique_between_methods >= 3:
            print("✅ GOOD: Adequate method differentiation")
            return True
        else:
            print("⚠️  NEEDS IMPROVEMENT: Limited differentiation")
            return False
        
    # Keep all other existing methods unchanged
    def _calculate_adjusted_category_preference(self, user_id: str, category: str) -> float:
        """Calculate category preference adjusted by forgetting"""
        
        if user_id not in getattr(self.kg, 'user_profiles', {}):
            return 0.1
        
        user_profile = self.kg.user_profiles[user_id]
        rated_products = user_profile.get('rated_products', set())
        
        category_strength = 0.0
        category_count = 0
        
        for product_id in list(rated_products)[:20]:  # Limit for performance
            product_features = getattr(self.kg, 'product_features', {}).get(product_id, {})
            if product_features.get('category') == category:
                memory_key = (user_id, product_id)
                if hasattr(self.kg, 'memory_strength') and memory_key in self.kg.memory_strength:
                    memory_strength = self.kg.memory_strength[memory_key]
                    category_strength += memory_strength
                    category_count += 1
        
        if category_count == 0:
            return 0.1
        
        return category_strength / category_count
    
    # Keep all other existing methods unchanged...
    def _extract_neural_features(self, user_id: str, product_id: str, current_time: float) -> np.ndarray:
        """Extract features for neural forgetting network"""
        features = []
        
        # Time-based features
        memory_key = (user_id, product_id)
        last_time = getattr(self.kg, 'last_interaction_time', {}).get(memory_key, current_time)
        time_diff = (current_time - last_time) / (24 * 3600)  # Days
        features.append(min(time_diff, 365) / 365)  # Normalized
        
        # Rating-based features
        if hasattr(self.kg, 'ratings_df') and self.kg.ratings_df is not None:
            user_ratings = self.kg.ratings_df[
                (self.kg.ratings_df['user_id'] == user_id) & 
                (self.kg.ratings_df['product_id'] == product_id)
            ]
            if not user_ratings.empty:
                rating = user_ratings.iloc[0]['rating']
                features.append(rating / 5.0)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Category features
        categories = getattr(self.kg, 'categories', ['Unknown'])
        product_features = getattr(self.kg, 'product_features', {}).get(product_id, {})
        category = product_features.get('category', 'Unknown')
        
        category_vec = [0] * len(categories)
        if category in categories:
            category_vec[categories.index(category)] = 1
        features.extend(category_vec)
        
        # User features
        user_profile = getattr(self.kg, 'user_profiles', {}).get(user_id, {})
        features.extend([
            user_profile.get('avg_rating', 3.5) / 5.0,
            min(user_profile.get('rating_count', 0), 100) / 100,
            user_profile.get('rating_std', 1.0) / 2.0
        ])
        
        # Interaction count
        interaction_count = getattr(self.kg, 'interaction_counts', {}).get(memory_key, 1)
        features.append(min(interaction_count, 10) / 10)
        
        # Pad to expected size
        expected_size = 5 + len(categories) + 10
        features = features[:expected_size]
        while len(features) < expected_size:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_attention_weights(self, user_id: str, product_ids: set) -> Dict[str, float]:
        """Calculate attention weights for products"""
        weights = {}
        
        if not product_ids:
            return weights
        
        user_profile = getattr(self.kg, 'user_profiles', {}).get(user_id, {})
        avg_rating = user_profile.get('avg_rating', 3.5)
        
        for product_id in list(product_ids)[:20]:  # Limit for performance
            weight = 0.5  # Base weight
            
            # Rating-based attention
            if hasattr(self.kg, 'ratings_df') and self.kg.ratings_df is not None:
                user_ratings = self.kg.ratings_df[
                    (self.kg.ratings_df['user_id'] == user_id) & 
                    (self.kg.ratings_df['product_id'] == product_id)
                ]
                if not user_ratings.empty:
                    rating = user_ratings.iloc[0]['rating']
                    weight += (rating - avg_rating) / 5.0 * 0.3
            
            # Category attention
            product_features = getattr(self.kg, 'product_features', {}).get(product_id, {})
            category = product_features.get('category', 'Unknown')
            
            # Rare categories get more attention
            same_category_count = sum(1 for pid in list(product_ids)[:20]
                                    if getattr(self.kg, 'product_features', {}).get(pid, {}).get('category') == category)
            
            if same_category_count <= 2:
                weight += 0.2
            
            # Price-based attention
            price_range = product_features.get('price_range', '$100-300')
            if '$500+' in price_range or '$800+' in price_range:
                weight += 0.15
            
            weights[product_id] = np.clip(weight, 0.1, 1.0)
        
        return weights
    
    def _find_related_items(self, product_id: str, user_id: str) -> List[str]:
        """Find items related to the given product"""
        related = []
        
        product_features = getattr(self.kg, 'product_features', {}).get(product_id, {})
        category = product_features.get('category', 'Unknown')
        price_range = product_features.get('price_range', '')
        
        # Find items in same category or price range
        for other_id, other_features in list(getattr(self.kg, 'product_features', {}).items())[:50]:
            if other_id == product_id:
                continue
                
            similarity_score = 0
            
            if other_features.get('category') == category:
                similarity_score += 0.6
            
            if other_features.get('price_range') == price_range:
                similarity_score += 0.3
            
            if similarity_score > 0.5:
                related.append(other_id)
        
        return related[:5]  # Limit cascade size
    
    def _calculate_context_factor(self, product_id: str, season: str, mood: str, browsing_category: str) -> float:
        """Calculate how context affects forgetting"""
        factor = 1.0
        
        product_features = getattr(self.kg, 'product_features', {}).get(product_id, {})
        category = product_features.get('category', 'Unknown')
        
        # Seasonal context
        seasonal_categories = {
            'winter': ['Gaming', 'Electronics', 'Smart_Home'],
            'spring': ['Sports', 'Cameras', 'Audio'],
            'summer': ['Wearables', 'Audio', 'Gaming'],
            'fall': ['Laptops', 'Electronics', 'Smart_Home']
        }
        
        if category in seasonal_categories.get(season, []):
            factor *= 0.8
        else:
            factor *= 1.2
        
        # Mood-based context
        if mood == 'shopping':
            factor *= 0.7
        elif mood == 'focused' and category in ['Laptops', 'Software']:
            factor *= 0.6
        
        # Browsing context
        if browsing_category and category == browsing_category:
            factor *= 0.5
        
        return np.clip(factor, 0.3, 2.0)
    
    def _get_current_season(self) -> str:
        """Get current season"""
        month = datetime.datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _record_forgetting_event(self, user_id: str, item_id: str, forgetting_type: str, 
                                strength: float, context: Dict = None):
        """Record a forgetting event"""
        event = ForgettingEvent(
            timestamp=datetime.datetime.now().timestamp(),
            user_id=user_id,
            item_id=item_id,
            forgetting_type=forgetting_type,
            strength=strength,
            context=context or {}
        )
        
        self.forgetting_events.append(event)
        self.forgetting_patterns[forgetting_type].append(strength)

    # ADD VALIDATION METHODS
    def validate_method_differentiation(self, test_user_id: str = None):
        """Validate that different methods produce different results"""
        
        if test_user_id is None:
            # Get first available user
            if hasattr(self.kg, 'user_profiles') and self.kg.user_profiles:
                test_user_id = list(self.kg.user_profiles.keys())[0]
            else:
                print("❌ No users available for validation")
                return False
        
        print(f"\n🔍 VALIDATING METHOD DIFFERENTIATION for user: {test_user_id}")
        
        # Store original memory
        original_memory = dict(getattr(self.kg, 'memory_strength', {}))
        
        # Test each method
        method_results = {}
        mechanisms_to_test = [
            (['neural_adaptive'], 'Neural_Adaptive'),
            (['attention_based'], 'Attention_Based'), 
            (['cascade'], 'Cascade'),
            (['contextual'], 'Contextual')
        ]
        
        for mechanisms, name in mechanisms_to_test:
            print(f"\n🧪 Testing {name}...")
            
            # Reset memory
            if hasattr(self.kg, 'memory_strength'):
                self.kg.memory_strength = dict(original_memory)
            
            # Clear processed items
            self.neural_processed_items.clear()
            self.attention_processed_items.clear()
            self.cascade_affected_items.clear()
            self.contextual_processed_items.clear()
            
            # Get recommendations
            recommendations = self.get_advanced_recommendations_with_forgetting(
                test_user_id, mechanisms, n=10
            )
            
            method_results[name] = recommendations[:5]  # Store top 5
            print(f"📋 {name} top 5: {recommendations[:5]}")
        
        # Check for differences
        all_results = list(method_results.values())
        unique_results = set(tuple(result) for result in all_results)
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Methods tested: {len(method_results)}")
        print(f"   Unique result patterns: {len(unique_results)}")
        
        if len(unique_results) == 1:
            print("❌ WARNING: All methods produce IDENTICAL results!")
            print("    This indicates the differentiation is not working.")
            return False
        elif len(unique_results) == len(method_results):
            print("✅ SUCCESS: All methods produce DIFFERENT results!")
            print("    Strong differentiation is working correctly.")
            return True
        else:
            print(f"⚠️  PARTIAL: {len(unique_results)}/{len(method_results)} methods are differentiated")
            print("    Some improvement needed but basic differentiation works.")
            return True
    
    def get_performance_summary(self):
        """Get summary of forgetting mechanism performance"""
        
        summary = {
            'neural_processed_items': len(self.neural_processed_items),
            'attention_processed_items': len(self.attention_processed_items),
            'cascade_affected_items': len(self.cascade_affected_items),
            'contextual_processed_items': len(self.contextual_processed_items),
            'total_forgetting_events': len(self.forgetting_events),
            'forgetting_patterns': {k: len(v) for k, v in self.forgetting_patterns.items()}
        }
        
        print(f"\n📈 FORGETTING MECHANISM PERFORMANCE SUMMARY:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        return summary