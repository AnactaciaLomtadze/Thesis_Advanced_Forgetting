#!/usr/bin/env python3
"""
Data Loading Configuration for Amazon Electronics Dataset
Ensures consistent loading across all components with proper file paths
"""

import pandas as pd
import numpy as np
import os
import datetime
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataConfig')

class AmazonDataConfig:
    """
    Centralized configuration for Amazon Electronics dataset loading
    Ensures all components use the same data and parameters
    """
    
    def __init__(self):
        # FIXED: Standard file paths
        self.DATA_FOLDER = './data'
        self.RATINGS_FILE = 'ratings_Electronics.csv'
        self.FULL_DATA_PATH = os.path.join(self.DATA_FOLDER, self.RATINGS_FILE)
        
        # FIXED: Consistent parameters across all components
        self.MAX_USERS = 5000      # Reasonable size for good connectivity
        self.MAX_PRODUCTS = 10000  # Reasonable size for good connectivity
        self.MIN_RATINGS_PER_USER = 8
        self.MIN_RATINGS_PER_PRODUCT = 5
        
        # Dataset information
        self.loaded_data = None
        self.final_stats = {}
        
        logger.info(f"Configured for Amazon Electronics dataset")
        logger.info(f"Data path: {self.FULL_DATA_PATH}")
        logger.info(f"Target: {self.MAX_USERS} users, {self.MAX_PRODUCTS} products")
    
    def load_and_process_amazon_electronics(self):
        """
        Load and process the Amazon Electronics dataset with consistent parameters
        """
        logger.info("Loading Amazon Electronics dataset...")
        
        # Check if file exists
        if not os.path.exists(self.FULL_DATA_PATH):
            logger.error(f"Dataset not found at: {self.FULL_DATA_PATH}")
            logger.info("Please ensure the file exists at the specified path")
            return None
        
        try:
            # Load the CSV file - handle different possible formats
            logger.info(f"Reading from: {self.FULL_DATA_PATH}")
            
            # Try standard CSV format first
            try:
                df = pd.read_csv(self.FULL_DATA_PATH)
                logger.info(f"Loaded as standard CSV with columns: {list(df.columns)}")
            except:
                # Try without headers (Amazon format)
                df = pd.read_csv(self.FULL_DATA_PATH, header=None, 
                               names=['user_id', 'product_id', 'rating', 'timestamp'])
                logger.info("Loaded as headerless CSV with assigned column names")
            
            logger.info(f"Initial dataset: {len(df):,} ratings")
            
            # Data cleaning
            df = self._clean_amazon_data(df)
            
            # Smart sampling to get target sizes with good connectivity
            df = self._smart_sample_for_connectivity(df)
            
            # Final validation
            df = self._final_validation_and_cleanup(df)
            
            self.loaded_data = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading Amazon Electronics data: {e}")
            return None
    
    def _clean_amazon_data(self, df):
        """Clean the Amazon Electronics dataset"""
        logger.info("Cleaning Amazon Electronics data...")
        
        initial_len = len(df)
        
        # Ensure proper column names
        expected_columns = ['user_id', 'product_id', 'rating', 'timestamp']
        if list(df.columns) != expected_columns:
            df.columns = expected_columns
        
        # Convert data types
        df['user_id'] = df['user_id'].astype(str)
        df['product_id'] = df['product_id'].astype(str)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna()
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        df = df.drop_duplicates()
        
        logger.info(f"Cleaned: {initial_len:,} -> {len(df):,} ratings")
        return df
    
    def _smart_sample_for_connectivity(self, df):
        """Smart sampling to achieve target user/product counts with good connectivity"""
        logger.info("Smart sampling for optimal connectivity...")
        
        # Step 1: Get most active users (better connectivity)
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.MIN_RATINGS_PER_USER]
        
        if len(active_users) > self.MAX_USERS:
            selected_users = active_users.head(self.MAX_USERS).index
            df = df[df['user_id'].isin(selected_users)]
            logger.info(f"Selected {len(selected_users)} most active users")
        
        # Step 2: Get most popular products (better connectivity)
        product_counts = df['product_id'].value_counts()
        popular_products = product_counts[product_counts >= self.MIN_RATINGS_PER_PRODUCT]
        
        if len(popular_products) > self.MAX_PRODUCTS:
            selected_products = popular_products.head(self.MAX_PRODUCTS).index
            df = df[df['product_id'].isin(selected_products)]
            logger.info(f"Selected {len(selected_products)} most popular products")
        
        return df
    
    def _final_validation_and_cleanup(self, df):
        """Final validation and cleanup"""
        logger.info("Final validation and cleanup...")
        
        # Iterative filtering for consistency (max 3 iterations)
        for iteration in range(3):
            initial_len = len(df)
            
            # Filter users
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.MIN_RATINGS_PER_USER].index
            df = df[df['user_id'].isin(valid_users)]
            
            # Filter products
            product_counts = df['product_id'].value_counts()
            valid_products = product_counts[product_counts >= self.MIN_RATINGS_PER_PRODUCT].index
            df = df[df['product_id'].isin(valid_products)]
            
            if len(df) == initial_len:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Calculate final statistics
        self.final_stats = {
            'total_ratings': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_products': df['product_id'].nunique(),
            'avg_rating': df['rating'].mean(),
            'rating_distribution': df['rating'].value_counts().to_dict(),
            'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['product_id'].nunique())),
            'avg_ratings_per_user': df.groupby('user_id').size().mean(),
            'avg_ratings_per_product': df.groupby('product_id').size().mean()
        }
        
        logger.info("FINAL DATASET STATISTICS:")
        logger.info(f"  Total ratings: {self.final_stats['total_ratings']:,}")
        logger.info(f"  Unique users: {self.final_stats['unique_users']:,}")
        logger.info(f"  Unique products: {self.final_stats['unique_products']:,}")
        logger.info(f"  Average rating: {self.final_stats['avg_rating']:.2f}")
        logger.info(f"  Sparsity: {self.final_stats['sparsity']:.4f}")
        logger.info(f"  Avg ratings/user: {self.final_stats['avg_ratings_per_user']:.1f}")
        logger.info(f"  Avg ratings/product: {self.final_stats['avg_ratings_per_product']:.1f}")
        
        return df
    
    def get_consistent_parameters(self):
        """Get parameters to be used consistently across all components"""
        return {
            'max_users': self.final_stats.get('unique_users', self.MAX_USERS),
            'max_products': self.final_stats.get('unique_products', self.MAX_PRODUCTS),
            'min_ratings_per_user': self.MIN_RATINGS_PER_USER,
            'data_file': self.FULL_DATA_PATH,
            'dataset_stats': self.final_stats
        }


# FIXED: Modified OptimizedWorkingEvaluator to use consistent data loading
class FixedOptimizedWorkingEvaluator:
    """
    FIXED version that ensures consistent data loading with Amazon Electronics
    """
    
    def __init__(self, data_config, output_dir='./amazon_optimized'):
        self.data_config = data_config
        self.output_dir = output_dir
        
        # Use parameters from data config
        params = data_config.get_consistent_parameters()
        self.max_users = params['max_users']
        self.max_products = params['max_products']
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        self.ratings_df = None
        self.train_df = None
        self.user_profiles = {}
        self.product_features = {}
        
        self.electronics_categories = [
            'Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras', 
            'Gaming', 'Smart_Home', 'Wearables', 'Audio', 'TV_Video',
            'Computer_Accessories', 'Storage', 'Networking', 'Software'
        ]
        
        logger.info(f"Fixed evaluator initialized with consistent parameters")
    
    def load_data(self, min_ratings_per_user=None):
        """Load data using the consistent data configuration"""
        logger.info("Loading data using consistent configuration...")
        
        # Use the pre-loaded and processed data
        self.ratings_df = self.data_config.loaded_data.copy()
        
        if self.ratings_df is None:
            logger.error("No data available from data config")
            return False
        
        logger.info(f"Loaded consistent dataset: {len(self.ratings_df):,} ratings")
        logger.info(f"Users: {self.ratings_df['user_id'].nunique():,}")
        logger.info(f"Products: {self.ratings_df['product_id'].nunique():,}")
        
        return True
    
    def create_train_test_split(self, test_ratio=0.2):
        """Create train/test split using the loaded data"""
        logger.info("Creating train/test split...")
        
        test_data = {}
        train_data = []
        
        for user_id in self.ratings_df['user_id'].unique():
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id].copy()
            user_ratings = user_ratings.sort_values('timestamp')
            
            if len(user_ratings) >= self.data_config.MIN_RATINGS_PER_USER:
                n_test = max(1, int(len(user_ratings) * test_ratio))
                n_test = min(n_test, len(user_ratings) - 5)
                
                if n_test > 0:
                    train_ratings = user_ratings.iloc[:-n_test]
                    test_ratings = user_ratings.iloc[-n_test:]
                    
                    train_data.append(train_ratings)
                    test_items = set(test_ratings['product_id'].astype(str).tolist())
                    test_data[str(user_id)] = test_items
        
        self.train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        
        logger.info(f"Train/test split completed:")
        logger.info(f"  Training: {len(self.train_df):,} ratings")
        logger.info(f"  Test users: {len(test_data)}")
        
        return test_data
    
    def build_profiles(self):
        """Build user profiles using the consistent data"""
        logger.info("Building user profiles from consistent data...")
        
        if self.train_df is None or len(self.train_df) == 0:
            logger.warning("No training data available for building profiles")
            return False
        
        self.user_profiles = {}
        
        # Build user profiles
        for user_id in self.train_df['user_id'].unique():
            user_ratings = self.train_df[self.train_df['user_id'] == user_id]
            
            # Category preferences
            category_prefs = defaultdict(float)
            for _, row in user_ratings.iterrows():
                category = self._get_category(str(row['product_id']))
                category_prefs[category] += row['rating'] / 5.0
            
            # Normalize
            total = sum(category_prefs.values())
            if total > 0:
                category_prefs = {k: v/total for k, v in category_prefs.items()}
            
            self.user_profiles[str(user_id)] = {
                'avg_rating': float(user_ratings['rating'].mean()),
                'category_preferences': dict(category_prefs),
                'rated_products': set(user_ratings['product_id'].astype(str).tolist()),
                'rating_count': len(user_ratings)
            }
        
        # Build product features
        self._build_product_features()
        
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
        logger.info(f"Built features for {len(self.product_features)} products")
        
        return True
    
    def _build_product_features(self):
        """Build product features from training data"""
        if self.train_df is None:
            return
            
        self.product_features = {}
        
        for product_id in self.train_df['product_id'].unique():
            product_ratings = self.train_df[self.train_df['product_id'] == product_id]
            
            self.product_features[str(product_id)] = {
                'category': self._get_category(str(product_id)),
                'avg_rating': float(product_ratings['rating'].mean()),
                'popularity': len(product_ratings),
            }
    
    def _get_category(self, product_id):
        """Get category for product"""
        return self.electronics_categories[hash(product_id) % len(self.electronics_categories)]
    
    def get_recommendations(self, user_id, method, n=10):
        """Get recommendations using working methods only"""
        user_id = str(user_id)
        
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        rated_products = profile['rated_products']
        candidates = []
        
        # Get candidates (products not rated by user)
        all_products = set(self.product_features.keys())
        candidate_products = all_products - rated_products
        
        for product_id in candidate_products:
            features = self.product_features[product_id]
            
            if method == 'popular':
                score = features['popularity']
                
            elif method == 'content_based':
                category = features['category']
                cat_pref = profile['category_preferences'].get(category, 0.1)
                quality = features['avg_rating'] / 5.0
                popularity = min(1.0, features['popularity'] / 20.0)
                score = 0.5 * cat_pref + 0.3 * quality + 0.2 * popularity
                
            elif method == 'quality_based':
                score = features['avg_rating'] * np.log(features['popularity'] + 1)
                
            else:
                score = features['avg_rating']
            
            candidates.append((product_id, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in candidates[:n]]
    
    def calculate_metrics(self, test_items, recommendations, method_name=""):
        """Calculate metrics for evaluation"""
        
        if not recommendations or not test_items:
            return {
                'hit_rate': 0.0, 'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0,
                'novelty': 0.0, 'diversity': 0.0, 'coverage': 0.0
            }
        
        test_items = set(str(item) for item in test_items)
        recommendations = [str(item) for item in recommendations]
        
        # Basic metrics
        overlap = [item for item in recommendations if item in test_items]
        hit_rate = 1.0 if len(overlap) > 0 else 0.0
        precision = len(overlap) / len(recommendations)
        recall = len(overlap) / len(test_items)
        
        # Enhanced NDCG
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommendations) if item in test_items)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), len(recommendations))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Advanced metrics
        novelty = 1.0 - (len(set(recommendations)) / len(recommendations))
        diversity = min(1.0, len(set(r[:4] for r in recommendations)) / 5.0)
        coverage = len(set(recommendations)) / max(10000, len(recommendations) * 10)
        
        return {
            'hit_rate': min(1.0, hit_rate),
            'precision': min(1.0, precision),
            'recall': min(1.0, recall),
            'ndcg': min(1.0, ndcg),
            'novelty': min(1.0, novelty),
            'diversity': min(1.0, diversity),
            'coverage': min(1.0, coverage)
        }


# FIXED: Main execution function that ensures consistent data loading
def run_with_consistent_amazon_electronics_data():
    """
    Main function that ensures all components use the same Amazon Electronics data
    """
    print("🎯 RUNNING WITH CONSISTENT AMAZON ELECTRONICS DATA")
    print("=" * 60)
    
    # Step 1: Initialize data configuration
    print("\n📊 Step 1: Initializing Amazon Electronics data configuration...")
    data_config = AmazonDataConfig()
    
    # Step 2: Load and process the data once
    print(f"\n📂 Step 2: Loading data from {data_config.FULL_DATA_PATH}...")
    processed_data = data_config.load_and_process_amazon_electronics()
    
    if processed_data is None:
        print("❌ Failed to load Amazon Electronics data")
        print("💡 Please ensure the file 'ratings_Electronics.csv' exists in the './data' folder")
        return None, None
    
    print("✅ Data loaded and processed successfully!")
    
    # Step 3: Initialize evaluator with consistent data
    print("\n🔧 Step 3: Initializing evaluator with consistent data...")
    evaluator = FixedOptimizedWorkingEvaluator(data_config)
    
    success = evaluator.load_data()
    if not success:
        print("❌ Failed to initialize evaluator")
        return None, None
    
    print("✅ Evaluator initialized successfully!")
    
    # Step 4: Build profiles and create train/test split
    print("\n🏗️ Step 4: Building profiles and creating train/test split...")
    test_data = evaluator.create_train_test_split()
    success = evaluator.build_profiles()
    
    if not success:
        print("❌ Failed to build profiles")
        return None, None
    
    print(f"✅ Profiles built successfully!")
    print(f"   Users with profiles: {len(evaluator.user_profiles)}")
    print(f"   Products with features: {len(evaluator.product_features)}")
    print(f"   Test users: {len(test_data)}")
    
    # Step 5: Return everything for use in thesis evaluation
    print("\n🎯 Step 5: Ready for thesis evaluation!")
    
    # Create knowledge graph adapter for advanced mechanisms
    class FixedKGAdapter:
        def __init__(self, evaluator):
            self.categories = evaluator.electronics_categories
            self.user_profiles = evaluator.user_profiles
            self.product_features = evaluator.product_features
            self.ratings_df = evaluator.ratings_df
            
            # Initialize memory structures for advanced mechanisms
            self.memory_strength = {}
            self.last_interaction_time = {}
            
            current_time = datetime.datetime.now().timestamp()
            
            # Initialize memory from ratings data
            for _, row in evaluator.train_df.iterrows():
                user_id = str(row['user_id'])
                product_id = str(row['product_id'])
                rating = row['rating']
                timestamp = row['timestamp']
                
                memory_key = (user_id, product_id)
                self.memory_strength[memory_key] = rating / 5.0  # Normalize to [0,1]
                self.last_interaction_time[memory_key] = timestamp
    
    kg_adapter = FixedKGAdapter(evaluator)
    
    print(f"✅ Knowledge graph adapter ready!")
    print(f"   Memory entries: {len(kg_adapter.memory_strength)}")
    
    # Return consistent data and configurations
    return {
        'data_config': data_config,
        'evaluator': evaluator,
        'kg_adapter': kg_adapter,
        'test_data': test_data,
        'consistent_parameters': data_config.get_consistent_parameters()
    }, True


if __name__ == "__main__":
    print("🧪 TESTING CONSISTENT DATA LOADING")
    
    result, success = run_with_consistent_amazon_electronics_data()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Amazon Electronics data loaded consistently")
        print("✅ All components use the same dataset")
        print("✅ Ready for thesis evaluation")
        
        # Print final statistics
        stats = result['consistent_parameters']['dataset_stats']
        print(f"\n📊 FINAL DATASET STATISTICS:")
        print(f"  📁 File: ./data/ratings_Electronics.csv")
        print(f"  📈 Ratings: {stats['total_ratings']:,}")
        print(f"  👥 Users: {stats['unique_users']:,}")
        print(f"  📦 Products: {stats['unique_products']:,}")
        print(f"  ⭐ Avg Rating: {stats['avg_rating']:.2f}")
        print(f"  🔗 Sparsity: {stats['sparsity']:.4f}")
        
    else:
        print("\n❌ FAILED!")
        print("Please check that ./data/ratings_Electronics.csv exists")
