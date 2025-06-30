# src/amazon_forgetting_mechanism.py
import numpy as np
import datetime
import math
import logging
from collections import defaultdict
import pickle
import gzip
import time
from skopt import gp_minimize
from skopt.space import Real, Integer

class AmazonForgettingMechanism:
    """
    Implements various forgetting mechanisms for Amazon product recommendation systems.
    
    This class provides methods to simulate memory decay over time,
    allowing for more dynamic and temporally-aware recommendations for products.
    """
    def __init__(self, knowledge_graph):
        """
        Initialize the forgetting mechanism for an Amazon knowledge graph.
        
        Args:
            knowledge_graph: The AmazonProductKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.memory_strength = {}  # Maps (user_id, product_id) to memory strength
        self.last_interaction_time = {}  # Maps (user_id, product_id) to last interaction timestamp
        self.interaction_counts = defaultdict(int)  # Maps (user_id, product_id) to interaction count
        self.user_activity_patterns = {}  # Maps user_id to activity pattern metrics
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('AmazonForgettingMechanism')
        
        # Initialize memory strengths from existing graph data
        self._initialize_memory_strengths()
    
    def _initialize_memory_strengths(self):
        """Initialize memory strengths from existing ratings data."""
        self.logger.info("Initializing memory strengths from ratings data...")
        
        if self.kg.ratings_df is None:
            self.logger.warning("No ratings data available for initialization")
            return
            
        for _, rating in self.kg.ratings_df.iterrows():
            user_id = rating['user_id']
            product_id = rating['product_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
            # Initial memory strength is based on the rating value (normalized to [0,1])
            memory_strength = rating_value / 5.0
            
            self.memory_strength[(user_id, product_id)] = memory_strength
            self.last_interaction_time[(user_id, product_id)] = timestamp
            self.interaction_counts[(user_id, product_id)] += 1
        
        self.logger.info(f"Initialized memory strengths for {len(self.memory_strength)} user-product pairs")
    
    def implement_time_based_decay(self, user_id, decay_parameter=0.1):
        """
        Implement time-based decay for a user's memories.
        
        Args:
            user_id: The user ID to apply decay to
            decay_parameter: Controls how quickly memories decay (smaller values = slower decay)
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Apply decay to all products the user has interacted with
        for (u_id, product_id), strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                time_diff = current_time - last_time
                
                # Exponential decay formula: strength * e^(-decay_parameter * time_diff)
                # Time difference is in seconds, convert to days for more reasonable decay
                days_diff = time_diff / (24 * 60 * 60)
                decayed_strength = strength * math.exp(-decay_parameter * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, product_id)] = max(0.001, decayed_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def implement_ebbinghaus_forgetting_curve(self, user_id, retention=0.9, strength=1.0):
        """
        Implement the classic Ebbinghaus forgetting curve: R = e^(-t/S)
        
        Args:
            user_id: The user ID to apply decay to
            retention: Base retention rate
            strength: Parameter controlling memory strength
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        for (u_id, product_id), memory_strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                time_diff = (current_time - last_time) / (24 * 60 * 60)  # days
                
                # Adjust strength based on rating value and product characteristics
                if (u_id, product_id) in self.last_interaction_time:
                    rating_data = self.kg.ratings_df[
                        (self.kg.ratings_df['user_id'] == u_id) & 
                        (self.kg.ratings_df['product_id'] == product_id)
                    ]
                    if not rating_data.empty:
                        rating = rating_data.iloc[0]['rating']
                        individual_strength = strength * (rating / 5.0)
                    else:
                        individual_strength = strength
                
                # Classic Ebbinghaus formula
                new_strength = retention * np.exp(-time_diff / individual_strength)
                self.memory_strength[(u_id, product_id)] = max(0.001, new_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def implement_power_law_decay(self, user_id, decay_factor=0.75):
        """
        Implement power law decay, which better models long-term forgetting.
        Follows the form: S(t) = S(0) * (1 + t)^(-decay_factor)
        
        Args:
            user_id: The user ID to apply decay to
            decay_factor: Power law exponent controlling decay rate
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        for (u_id, product_id), initial_strength in self.memory_strength.items():
            if u_id == user_id:
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                time_diff = current_time - last_time
                
                # Convert to days and add 1 to avoid division by zero
                days_diff = (time_diff / (24 * 60 * 60)) + 1
                
                # Power law decay
                decayed_strength = initial_strength * (days_diff ** (-decay_factor))
                
                # Update memory strength
                self.memory_strength[(u_id, product_id)] = max(0.001, decayed_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def implement_category_based_decay(self, user_id, category_decay_rates=None):
        """
        Implement category-specific decay for Amazon products.
        Different product categories may have different forgetting patterns.
        
        Args:
            user_id: The user ID to apply decay to
            category_decay_rates: Dictionary mapping categories to decay rates
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        if category_decay_rates is None:
            # Default decay rates for different Amazon categories
            category_decay_rates = {
                'Books': 0.05,      # Books are remembered longer
                'Electronics': 0.08, # Electronics moderate decay
                'Clothing': 0.12,    # Fashion items decay faster
                'Food': 0.15,        # Consumables decay fastest
                'Home': 0.06,        # Home items moderate-slow decay
                'Sports': 0.10,      # Sports items moderate decay
                'Beauty': 0.13,      # Beauty products decay faster
                'Health': 0.08,      # Health items moderate decay
                'Toys': 0.11,        # Toys moderate-fast decay
                'Kitchen': 0.07      # Kitchen items slow decay
            }
        
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        for (u_id, product_id), strength in self.memory_strength.items():
            if u_id == user_id:
                # Get product category
                product_node = f"product_{product_id}"
                category = 'Unknown'
                
                if product_node in self.kg.G:
                    category = self.kg.G.nodes[product_node].get('category', 'Unknown')
                
                # Get appropriate decay rate
                decay_rate = category_decay_rates.get(category, 0.1)  # Default rate
                
                # Apply time-based decay with category-specific rate
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)
                
                decayed_strength = strength * math.exp(-decay_rate * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, product_id)] = max(0.001, decayed_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def implement_price_aware_decay(self, user_id, price_decay_modifier=0.3):
        """
        Implement price-aware decay where expensive items are remembered longer.
        
        Args:
            user_id: The user ID to apply decay to
            price_decay_modifier: How much price affects decay rate
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        user_memories = {}
        
        # Price range mappings (synthetic for demonstration)
        price_multipliers = {
            '$0-25': 1.0,      # Normal decay
            '$25-50': 0.9,     # Slightly slower decay
            '$50-100': 0.8,    # Slower decay
            '$100-250': 0.7,   # Much slower decay
            '$250+': 0.6       # Slowest decay (expensive items remembered longer)
        }
        
        for (u_id, product_id), strength in self.memory_strength.items():
            if u_id == user_id:
                # Get product price range
                product_node = f"product_{product_id}"
                price_range = '$0-25'  # Default
                
                if product_node in self.kg.G:
                    price_range = self.kg.G.nodes[product_node].get('price_range', '$0-25')
                
                # Get price multiplier
                price_multiplier = price_multipliers.get(price_range, 1.0)
                
                # Apply time-based decay with price adjustment
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)
                
                # Adjust decay rate based on price
                adjusted_decay_rate = 0.1 * price_multiplier
                decayed_strength = strength * math.exp(-adjusted_decay_rate * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, product_id)] = max(0.001, decayed_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def implement_seasonal_decay(self, user_id, seasonal_factor=0.2):
        """
        Implement seasonal decay for products that may be seasonal.
        
        Args:
            user_id: The user ID to apply decay to
            seasonal_factor: How much seasonality affects decay
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        current_month = datetime.datetime.now().month
        user_memories = {}
        
        # Seasonal categories (higher values = more seasonal)
        seasonal_categories = {
            'Clothing': 0.8,     # Highly seasonal
            'Sports': 0.6,       # Moderately seasonal
            'Garden': 0.9,       # Very seasonal
            'Toys': 0.7,         # Seasonal (holidays)
            'Books': 0.2,        # Not very seasonal
            'Electronics': 0.3,  # Slightly seasonal
            'Home': 0.4,         # Moderately seasonal
            'Health': 0.1,       # Not seasonal
            'Beauty': 0.5,       # Moderately seasonal
            'Kitchen': 0.3       # Slightly seasonal
        }
        
        for (u_id, product_id), strength in self.memory_strength.items():
            if u_id == user_id:
                # Get product category
                product_node = f"product_{product_id}"
                category = 'Unknown'
                
                if product_node in self.kg.G:
                    category = self.kg.G.nodes[product_node].get('category', 'Unknown')
                
                # Get seasonality factor
                seasonality = seasonal_categories.get(category, 0.3)
                
                # Calculate seasonal adjustment based on time since interaction
                last_time = self.last_interaction_time.get((u_id, product_id), 0)
                last_month = datetime.datetime.fromtimestamp(last_time).month
                
                # Calculate month difference (accounting for year wrap)
                month_diff = abs(current_month - last_month)
                if month_diff > 6:
                    month_diff = 12 - month_diff
                
                # Seasonal decay increases with month difference for seasonal items
                seasonal_decay = 1.0 + (seasonality * seasonal_factor * (month_diff / 6.0))
                
                # Apply time-based decay with seasonal adjustment
                time_diff = current_time - last_time
                days_diff = time_diff / (24 * 60 * 60)
                
                adjusted_decay_rate = 0.1 * seasonal_decay
                decayed_strength = strength * math.exp(-adjusted_decay_rate * days_diff)
                
                # Update memory strength
                self.memory_strength[(u_id, product_id)] = max(0.001, decayed_strength)
                user_memories[product_id] = self.memory_strength[(u_id, product_id)]
        
        return user_memories
    
    def create_amazon_hybrid_decay_function(self, user_id, time_weight=0.3, usage_weight=0.2, 
                                          category_weight=0.2, price_weight=0.15, seasonal_weight=0.15):
        """
        Create a hybrid decay function specific to Amazon products that combines multiple factors.
        
        Args:
            user_id: The user ID to apply decay to
            time_weight: Weight for time-based decay
            usage_weight: Weight for usage-based decay
            category_weight: Weight for category-based decay
            price_weight: Weight for price-based decay
            seasonal_weight: Weight for seasonal decay
            
        Returns:
            Dictionary of updated memory strengths for the user
        """
        current_time = datetime.datetime.now().timestamp()
        current_month = datetime.datetime.now().month
        user_memories = {}
        
        # Get all products the user has interacted with
        user_products = [(p_id, strength) for (u_id, p_id), strength in self.memory_strength.items() if u_id == user_id]
        
        # Calculate average interaction time for novelty component
        avg_timestamp = sum(self.last_interaction_time.get((user_id, p_id), 0) for p_id, _ in user_products) / max(1, len(user_products))
        
        for product_id, strength in user_products:
            # Time-based component
            last_time = self.last_interaction_time.get((user_id, product_id), 0)
            time_diff = current_time - last_time
            days_diff = time_diff / (24 * 60 * 60)
            time_decay = math.exp(-0.05 * days_diff)
            
            # Usage-based component
            interaction_count = self.interaction_counts.get((user_id, product_id), 0)
            usage_factor = min(1.0, interaction_count / 3.0)  # Products may be purchased less frequently
            
            # Category-based component
            product_node = f"product_{product_id}"
            category = 'Unknown'
            if product_node in self.kg.G:
                category = self.kg.G.nodes[product_node].get('category', 'Unknown')
            
            # Category decay rates
            category_factors = {
                'Books': 0.9, 'Electronics': 0.8, 'Clothing': 0.6, 'Food': 0.4,
                'Home': 0.8, 'Sports': 0.7, 'Beauty': 0.6, 'Health': 0.8,
                'Toys': 0.7, 'Kitchen': 0.8
            }
            category_factor = category_factors.get(category, 0.7)
            
            # Price-based component
            price_range = '$0-25'
            if product_node in self.kg.G:
                price_range = self.kg.G.nodes[product_node].get('price_range', '$0-25')
            
            price_factors = {
                '$0-25': 0.6, '$25-50': 0.7, '$50-100': 0.8, '$100-250': 0.9, '$250+': 1.0
            }
            price_factor = price_factors.get(price_range, 0.7)
            
            # Seasonal component
            last_month = datetime.datetime.fromtimestamp(last_time).month if last_time > 0 else current_month
            month_diff = abs(current_month - last_month)
            if month_diff > 6:
                month_diff = 12 - month_diff
            
            seasonal_categories = {
                'Clothing': 0.8, 'Sports': 0.6, 'Garden': 0.9, 'Toys': 0.7
            }
            seasonality = seasonal_categories.get(category, 0.3)
            seasonal_factor = 1.0 - (seasonality * (month_diff / 6.0) * 0.3)
            
            # Combine all factors
            hybrid_factor = (
                time_weight * time_decay + 
                usage_weight * usage_factor + 
                category_weight * category_factor + 
                price_weight * price_factor + 
                seasonal_weight * seasonal_factor
            )
            
            # Apply decay
            new_strength = strength * hybrid_factor
            self.memory_strength[(user_id, product_id)] = max(0.001, min(1.0, new_strength))
            user_memories[product_id] = self.memory_strength[(user_id, product_id)]
        
        return user_memories
    
    def personalize_forgetting_parameters(self, user_id):
        """
        Personalize forgetting mechanism parameters based on user characteristics for Amazon.
        
        Args:
            user_id: The user ID
            
        Returns:
            Dictionary of personalized parameters for the hybrid decay function
        """
        # Get user activity level
        user_ratings = [strength for (u_id, p_id), strength in self.memory_strength.items() if u_id == user_id]
        activity_level = len(user_ratings)
        
        # Get user product preferences diversity
        user_products = [p_id for (u_id, p_id), _ in self.memory_strength.items() if u_id == user_id]
        diversity = self._calculate_category_diversity(user_products)
        
        # Get user's average spending pattern (based on price ranges of purchased products)
        spending_level = self._estimate_spending_level(user_id)
        
        # Adjust weights based on user characteristics
        if activity_level > 30:  # High activity user
            if diversity > 0.6:  # Diverse shopper
                # Active diverse shoppers - balance all factors
                time_weight = 0.25
                usage_weight = 0.2
                category_weight = 0.2
                price_weight = 0.2
                seasonal_weight = 0.15
            else:  # Focused shopper
                # Active focused shoppers - emphasize usage and category
                time_weight = 0.2
                usage_weight = 0.3
                category_weight = 0.3
                price_weight = 0.15
                seasonal_weight = 0.05
        else:  # Low activity user
            if spending_level > 0.7:  # High spender
                # Casual high spenders - emphasize price and time
                time_weight = 0.3
                usage_weight = 0.15
                category_weight = 0.15
                price_weight = 0.3
                seasonal_weight = 0.1
            else:  # Budget shopper
                # Budget shoppers - emphasize time and seasonal factors
                time_weight = 0.4
                usage_weight = 0.2
                category_weight = 0.15
                price_weight = 0.05
                seasonal_weight = 0.2
        
        return {
            'time_weight': time_weight,
            'usage_weight': usage_weight,
            'category_weight': category_weight,
            'price_weight': price_weight,
            'seasonal_weight': seasonal_weight
        }
    
    def _calculate_category_diversity(self, product_ids):
        """Calculate category diversity for a list of product IDs."""
        if not product_ids:
            return 0.0
        
        categories = []
        for product_id in product_ids:
            product_node = f"product_{product_id}"
            if product_node in self.kg.G:
                category = self.kg.G.nodes[product_node].get('category', 'Unknown')
                categories.append(category)
        
        if not categories:
            return 0.0
        
        unique_categories = len(set(categories))
        total_categories = len(self.kg.categories)
        
        return unique_categories / total_categories
    
    def _estimate_spending_level(self, user_id):
        """Estimate user's spending level based on price ranges of purchased products."""
        user_products = [p_id for (u_id, p_id), _ in self.memory_strength.items() if u_id == user_id]
        
        if not user_products:
            return 0.5
        
        price_scores = []
        price_mapping = {'$0-25': 0.1, '$25-50': 0.3, '$50-100': 0.5, '$100-250': 0.8, '$250+': 1.0}
        
        for product_id in user_products:
            product_node = f"product_{product_id}"
            if product_node in self.kg.G:
                price_range = self.kg.G.nodes[product_node].get('price_range', '$0-25')
                price_scores.append(price_mapping.get(price_range, 0.1))
        
        return np.mean(price_scores) if price_scores else 0.5
    
    def apply_forgetting_to_recommendations(self, user_id, recommendation_scores, forgetting_factor=0.4):
        """
        Apply Amazon-specific forgetting to recommendations.
        
        Args:
            user_id: User ID
            recommendation_scores: Dictionary of product_id -> score
            forgetting_factor: Strength of forgetting effect
            
        Returns:
            Adjusted recommendation scores
        """
        adjusted_scores = {}
        
        for product_id, score in recommendation_scores.items():
            memory_strength = self.memory_strength.get((user_id, product_id), 1.0)
            
            # Amazon-specific adjustments
            product_node = f"product_{product_id}"
            
            # Category-based adjustment
            category_boost = 0.0
            if product_node in self.kg.G:
                category = self.kg.G.nodes[product_node].get('category', 'Unknown')
                
                # Some categories benefit more from novelty
                novelty_categories = {'Electronics', 'Clothing', 'Beauty', 'Toys'}
                if category in novelty_categories:
                    category_boost = 0.1
            
            # Balance between familiar products (high memory) and novel products (low memory)
            if memory_strength > 0.7:
                # Very familiar products get slight penalty (user might want something new)
                adjustment = -0.05 * forgetting_factor
            elif memory_strength < 0.3:
                # Novel products get boost
                adjustment = 0.3 * forgetting_factor + category_boost
            else:
                # Products in the middle get smaller adjustments
                adjustment = (0.5 - memory_strength) * forgetting_factor * 0.5
            
            adjusted_scores[product_id] = score * (1.0 + adjustment)
        
        return adjusted_scores
    
    def integrate_forgetting_mechanism_into_recommendation_pipeline(self, recommendation_algorithm, forgetting_parameters):
        """
        Integrate Amazon forgetting mechanism into the recommendation pipeline.
        
        Args:
            recommendation_algorithm: Algorithm type or function for recommendations
            forgetting_parameters: Parameters for the forgetting mechanism
            
        Returns:
            Function that generates recommendations with Amazon forgetting mechanism applied
        """
        def amazon_forgetting_aware_recommendations(user_id, n=10):
            # Get personalized forgetting parameters if not provided
            if not forgetting_parameters or user_id not in self.user_activity_patterns:
                user_params = self.personalize_forgetting_parameters(user_id)
            else:
                user_params = forgetting_parameters

            # Ensure all required parameters are present with defaults
            params = {
                'time_weight': user_params.get('time_weight', 0.3),
                'usage_weight': user_params.get('usage_weight', 0.2),
                'category_weight': user_params.get('category_weight', 0.2),
                'price_weight': user_params.get('price_weight', 0.15),
                'seasonal_weight': user_params.get('seasonal_weight', 0.15),
                'forgetting_factor': user_params.get('forgetting_factor', 0.4)
            }

            # Apply Amazon hybrid decay to update memory strengths
            self.create_amazon_hybrid_decay_function(
                user_id, 
                time_weight=params['time_weight'],
                usage_weight=params['usage_weight'],
                category_weight=params['category_weight'],
                price_weight=params['price_weight'],
                seasonal_weight=params['seasonal_weight']
            )
            
            # Get base recommendations
            if isinstance(recommendation_algorithm, str):
                if recommendation_algorithm == 'personalized':
                    product_ids = self.kg.get_personalized_recommendations(user_id, n=n*2)
                    scores = {pid: (n*2 - i) / (n*2) for i, pid in enumerate(product_ids)}
                    
                elif recommendation_algorithm == 'graph_based':
                    product_ids = self.kg.get_graph_based_recommendations(user_id, n=n*2)
                    scores = {pid: (n*2 - i) / (n*2) for i, pid in enumerate(product_ids)}
                    
                elif recommendation_algorithm == 'hybrid':
                    product_ids = self.kg.get_hybrid_recommendations(user_id, n=n*2)
                    scores = {pid: (n*2 - i) / (n*2) for i, pid in enumerate(product_ids)}
                    
                else:
                    # Default to hybrid recommendations
                    product_ids = self.kg.get_hybrid_recommendations(user_id, n=n*2)
                    scores = {pid: (n*2 - i) / (n*2) for i, pid in enumerate(product_ids)}
            else:
                # Custom recommendation algorithm that returns scores
                scores = recommendation_algorithm(user_id)
            
            # Apply Amazon forgetting mechanism to adjust scores
            adjusted_scores = self.apply_forgetting_to_recommendations(
                user_id, 
                scores, 
                forgetting_factor=params.get('forgetting_factor', 0.4)
            )
            
            # Sort by adjusted scores and return top n
            sorted_recommendations = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:n]
            return [product_id for product_id, _ in sorted_recommendations]
        
        return amazon_forgetting_aware_recommendations
    
    def simulate_right_to_be_forgotten(self, user_id, product_ids=None):
        """
        Simulate a GDPR right to be forgotten request for Amazon product data.

        Args:
            user_id: The user ID requesting to be forgotten
            product_ids: Optional list of specific product IDs to forget (if None, forget all)

        Returns:
            Impact metrics on recommendation quality
        """
        # Store original recommendations for comparison
        original_recs = self.kg.get_hybrid_recommendations(user_id)

        # Store original data
        original_ratings = self.kg.ratings_df.copy()

        # Remove interactions
        if product_ids is None:
            # Remove all user's ratings
            self.kg.ratings_df = self.kg.ratings_df[self.kg.ratings_df['user_id'] != user_id]

            # Also remove from memory strength
            keys_to_remove = []
            for key in self.memory_strength:
                if key[0] == user_id:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.memory_strength[key]
                if key in self.last_interaction_time:
                    del self.last_interaction_time[key]
                if key in self.interaction_counts:
                    del self.interaction_counts[key]

            # Remove from user profiles
            if user_id in self.kg.user_profiles:
                del self.kg.user_profiles[user_id]
        else:
            # Remove specific ratings
            self.kg.ratings_df = self.kg.ratings_df[
                ~((self.kg.ratings_df['user_id'] == user_id) & 
                  (self.kg.ratings_df['product_id'].isin(product_ids)))
            ]

            # Remove from memory strength
            for product_id in product_ids:
                key = (user_id, product_id)
                if key in self.memory_strength:
                    del self.memory_strength[key]
                if key in self.last_interaction_time:
                    del self.last_interaction_time[key]
                if key in self.interaction_counts:
                    del self.interaction_counts[key]

            # Update user profile
            if user_id in self.kg.user_profiles:
                user_ratings = self.kg.ratings_df[self.kg.ratings_df['user_id'] == user_id]
                rated_products = set(user_ratings['product_id'].values)

                if rated_products:
                    category_vectors = []
                    for product_id in rated_products:
                        if product_id in self.kg.product_features:
                            category_vectors.append(self.kg.product_features[product_id])
                    
                    if category_vectors:
                        category_preferences = np.mean(category_vectors, axis=0)
                    else:
                        category_preferences = np.zeros(len(self.kg.categories))
                else:
                    category_preferences = np.zeros(len(self.kg.categories))

                self.kg.user_profiles[user_id]['rated_products'] = rated_products
                self.kg.user_profiles[user_id]['category_preferences'] = category_preferences

        # Get new recommendations
        if user_id in self.kg.user_profiles:
            new_recs = self.kg.get_hybrid_recommendations(user_id)
        else:
            new_recs = []

        # Calculate impact if possible
        if new_recs:
            # Calculate category diversity impact
            category_diversity_before = self._calculate_category_diversity(original_recs)
            category_diversity_after = self._calculate_category_diversity(new_recs)

            jaccard_similarity = len(set(original_recs).intersection(set(new_recs))) / \
                                len(set(original_recs).union(set(new_recs))) if original_recs and new_recs else 0

            new_items = [item for item in new_recs if item not in original_recs]
            new_item_percentage = len(new_items) / len(new_recs) if new_recs else 0

            impact = {
                'category_diversity_before': category_diversity_before,
                'category_diversity_after': category_diversity_after,
                'jaccard_similarity': jaccard_similarity,
                'new_item_percentage': new_item_percentage
            }
        else:
            impact = {
                'category_diversity_before': 0,
                'category_diversity_after': 0,
                'jaccard_similarity': 0,
                'new_item_percentage': 0,
                'complete_forget': True
            }

        # Restore original data
        self.kg.ratings_df = original_ratings

        # Rebuild user profiles
        self.kg._build_user_profiles()

        # Reinitialize memory strengths
        self._initialize_memory_strengths()

        # Count forgotten items
        forgotten_count = 0
        if product_ids is not None:
            forgotten_count = len(product_ids)
        else:
            user_ratings = original_ratings[original_ratings['user_id'] == user_id]
            forgotten_count = user_ratings.shape[0]

        return {
            'user_id': user_id,
            'forgotten_items': forgotten_count,
            'impact_metrics': impact
        }
    
    def serialize_and_store_memory_state(self, file_path, compression_level=0):
        """
        Serialize and store the current memory state for Amazon data.
        
        Args:
            file_path: Path to store the memory state
            compression_level: 0-9 compression level (0=none, 9=max)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'memory_strength': self.memory_strength,
                'last_interaction_time': self.last_interaction_time,
                'interaction_counts': self.interaction_counts,
                'user_activity_patterns': self.user_activity_patterns,
                'amazon_metadata': {
                    'categories': self.kg.categories,
                    'total_products': len(self.kg.product_features),
                    'total_users': self.kg.ratings_df['user_id'].nunique() if self.kg.ratings_df is not None else 0
                }
            }
            
            if compression_level > 0:
                with gzip.open(file_path, 'wb', compresslevel=compression_level) as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing Amazon memory state: {e}")
            return False
    
    def load_and_restore_memory_state(self, file_path, validation_check=True):
        """
        Load and restore a previously saved Amazon memory state.
        
        Args:
            file_path: Path to the stored memory state
            validation_check: Whether to validate the loaded data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load as gzipped first
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            except:
                # If not gzipped, try normal pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Validation check
            if validation_check:
                required_keys = ['memory_strength', 'last_interaction_time', 
                                'interaction_counts', 'user_activity_patterns']
                
                if not all(key in data for key in required_keys):
                    self.logger.error("Invalid Amazon memory state file: missing required data")
                    return False
                
                # Additional Amazon-specific validation
                if 'amazon_metadata' in data:
                    metadata = data['amazon_metadata']
                    if metadata.get('categories') != self.kg.categories:
                        self.logger.warning("Category mismatch in loaded memory state")
            
            # Restore state
            self.memory_strength = data['memory_strength']
            self.last_interaction_time = data['last_interaction_time']
            self.interaction_counts = data['interaction_counts']
            self.user_activity_patterns = data['user_activity_patterns']
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading Amazon memory state: {e}")
            return False
    
    def benchmark_scalability(self, num_users=None, num_interactions=None, repetitions=3):
        """
        Benchmark the scalability of Amazon forgetting mechanisms.
        
        Args:
            num_users: Number of users to test with (if None, use all)
            num_interactions: Number of interactions to test with (if None, use all)
            repetitions: Number of times to repeat each test
            
        Returns:
            Dictionary with benchmarking results
        """
        # Prepare subset of data if needed
        if num_users is not None or num_interactions is not None:
            original_ratings = self.kg.ratings_df.copy()
            
            if num_users is not None:
                user_ids = list(self.kg.ratings_df['user_id'].unique())
                if num_users < len(user_ids):
                    selected_users = user_ids[:num_users]
                    self.kg.ratings_df = self.kg.ratings_df[self.kg.ratings_df['user_id'].isin(selected_users)]
            
            if num_interactions is not None and len(self.kg.ratings_df) > num_interactions:
                self.kg.ratings_df = self.kg.ratings_df.sample(num_interactions)
            
            # Reinitialize with subset
            self._initialize_memory_strengths()
        
        # Define Amazon-specific strategies to benchmark
        strategies = {
            'time_based': lambda u: self.implement_time_based_decay(u),
            'ebbinghaus': lambda u: self.implement_ebbinghaus_forgetting_curve(u),
            'power_law': lambda u: self.implement_power_law_decay(u),
            'category_based': lambda u: self.implement_category_based_decay(u),
            'price_aware': lambda u: self.implement_price_aware_decay(u),
            'seasonal': lambda u: self.implement_seasonal_decay(u),
            'amazon_hybrid': lambda u: self.create_amazon_hybrid_decay_function(u)
        }
        
        # Run benchmarks
        results = defaultdict(list)
        user_ids = list(self.kg.ratings_df['user_id'].unique())
        
        self.logger.info(f"Benchmarking Amazon strategies with {len(user_ids)} users and {len(self.kg.ratings_df)} interactions")
        
        for strategy_name, strategy_fn in strategies.items():
            self.logger.info(f"Benchmarking Amazon {strategy_name} strategy...")
            
            for _ in range(repetitions):
                start_time = time.time()
                
                # Apply to all users
                for user_id in user_ids:
                    strategy_fn(user_id)
                
                end_time = time.time()
                results[strategy_name].append(end_time - start_time)
        
        # Restore original data if needed
        if num_users is not None or num_interactions is not None:
            self.kg.ratings_df = original_ratings
            self._initialize_memory_strengths()
        
        # Process results
        benchmark_results = {}
        for strategy_name, times in results.items():
            benchmark_results[strategy_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times)
            }
        
        benchmark_results['metadata'] = {
            'num_users': len(user_ids),
            'num_interactions': len(self.kg.ratings_df),
            'repetitions': repetitions,
            'dataset_type': 'Amazon Products'
        }
        
        return benchmark_results