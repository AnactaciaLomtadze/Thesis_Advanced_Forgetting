# src/amazon_knowledge_graph.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
import datetime
import logging
from tqdm import tqdm
import requests
import time

class AmazonProductKnowledgeGraph:
    """
    A knowledge graph representation of Amazon product review data with recommendations functionality.
    
    This class builds a graph-based representation of user-product interactions,
    product-product relationships based on similarity, and provides methods for
    generating personalized recommendations.
    """
    def __init__(self, data_path='./data/amazon'):
        """
        Initialize the Amazon Product Knowledge Graph.
        
        Args:
            data_path: Path to the Amazon dataset
        """
        self.data_path = data_path
        self.G = nx.Graph()  
        self.user_profiles = {}
        self.product_features = {}
        self.ratings_df = None
        self.products_df = None
        self.similarity_matrix = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('AmazonKG')
        
        # Product categories (Amazon-specific)
        self.categories = [
            'Books', 'Electronics', 'Home', 'Sports', 'Toys', 'Health',
            'Beauty', 'Clothing', 'Tools', 'Automotive', 'Pet', 'Garden',
            'Baby', 'Grocery', 'Office', 'Industrial', 'Jewelry', 'Shoes',
            'Kitchen', 'Computer'
        ]
        
    def load_data(self, ratings_file='ratings.csv', products_file=None):
        """Load the Amazon dataset."""
        try:
            # Load ratings data
            ratings_path = os.path.join(self.data_path, ratings_file)
            
            # Try different formats
            if ratings_file.endswith('.csv'):
                # Assume CSV format: user_id,product_id,rating,timestamp
                self.ratings_df = pd.read_csv(ratings_path, names=['user_id', 'product_id', 'rating', 'timestamp'])
            else:
                # Assume comma-separated format like your example
                with open(ratings_path, 'r') as f:
                    lines = f.readlines()
                
                data = []
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        user_id = parts[0]
                        product_id = parts[1]
                        rating = float(parts[2])
                        timestamp = int(parts[3])
                        data.append([user_id, product_id, rating, timestamp])
                
                self.ratings_df = pd.DataFrame(data, columns=['user_id', 'product_id', 'rating', 'timestamp'])
            
            # Clean and process data
            self.ratings_df = self.ratings_df.dropna()
            self.ratings_df['rating'] = pd.to_numeric(self.ratings_df['rating'], errors='coerce')
            self.ratings_df['timestamp'] = pd.to_numeric(self.ratings_df['timestamp'], errors='coerce')
            self.ratings_df = self.ratings_df.dropna()
            
            # Create products dataframe if not provided
            if products_file and os.path.exists(os.path.join(self.data_path, products_file)):
                self.products_df = pd.read_csv(os.path.join(self.data_path, products_file))
            else:
                # Create basic products dataframe from ratings
                unique_products = self.ratings_df['product_id'].unique()
                self.products_df = pd.DataFrame({
                    'product_id': unique_products,
                    'title': [f'Product_{pid}' for pid in unique_products],
                    'category': np.random.choice(self.categories, len(unique_products))
                })
                
                # Add random features for demonstration
                for i, category in enumerate(self.categories):
                    self.products_df[category] = 0
                
                # Assign categories
                for idx, row in self.products_df.iterrows():
                    category = row['category']
                    if category in self.categories:
                        self.products_df.at[idx, category] = 1
            
            # Create user mapping for consistent IDs
            unique_users = self.ratings_df['user_id'].unique()
            self.user_mapping = {user: i for i, user in enumerate(unique_users)}
            self.ratings_df['user_numeric_id'] = self.ratings_df['user_id'].map(self.user_mapping)
            
            # Create product mapping
            unique_products = self.ratings_df['product_id'].unique()
            self.product_mapping = {product: i for i, product in enumerate(unique_products)}
            self.ratings_df['product_numeric_id'] = self.ratings_df['product_id'].map(self.product_mapping)
            
            self.logger.info(f"Loaded {len(self.ratings_df)} ratings from {self.ratings_df['user_id'].nunique()} users on {self.ratings_df['product_id'].nunique()} products")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
        
    def build_knowledge_graph(self):
        """Build the knowledge graph from the Amazon data."""
        if self.ratings_df is None:
            if not self.load_data():
                self.logger.error("Cannot build knowledge graph without data")
                return False
        
        self.logger.info("Building knowledge graph...")
        
        # Add user nodes
        for user_id in tqdm(self.ratings_df['user_id'].unique(), desc="Adding user nodes"):
            self.G.add_node(
                f"user_{user_id}", 
                type='user'
            )
        
        # Add product nodes
        for _, product in tqdm(self.products_df.iterrows(), total=len(self.products_df), desc="Adding product nodes"):
            product_id = product['product_id']
            
            # Get category features
            category_features = []
            for category in self.categories:
                if category in product:
                    category_features.append(product[category])
                else:
                    category_features.append(0)
            
            self.G.add_node(
                f"product_{product_id}", 
                type='product',
                title=product.get('title', f'Product_{product_id}'),
                category=product.get('category', 'Unknown')
            )
            
            # Store product features for similarity calculation
            self.product_features[product_id] = np.array(category_features)
    
        # Add user-product edges (ratings)
        for _, rating in tqdm(self.ratings_df.iterrows(), total=len(self.ratings_df), desc="Adding rating edges"):
            user_id = rating['user_id']
            product_id = rating['product_id']
            rating_value = rating['rating']
            timestamp = rating['timestamp']
            
            rating_time = datetime.datetime.fromtimestamp(timestamp)

            self.G.add_edge(
                f"user_{user_id}", 
                f"product_{product_id}", 
                weight=rating_value,
                timestamp=timestamp,
                rating_time=rating_time
            )
        
        # Add product-product similarity edges
        self._add_product_similarity_edges()
        
        # Add category nodes and connections
        self._add_category_nodes()
        
        # Build user profiles for recommendations
        self._build_user_profiles()
        
        self.logger.info(f"Knowledge graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return True
    
    def _add_category_nodes(self):
        """Add category nodes and connect them to products."""
        # Add category nodes
        for category in self.categories:
            self.G.add_node(f"category_{category}", type='category', name=category)
        
        # Connect products to categories
        for _, product in self.products_df.iterrows():
            product_id = product['product_id']
            category = product.get('category', 'Unknown')
            
            if category in self.categories:
                self.G.add_edge(
                    f"product_{product_id}",
                    f"category_{category}",
                    relation_type='belongs_to'
                )
        
    def _add_product_similarity_edges(self, threshold=0.3):
        """
        Add product-product edges based on category/feature similarity.
        
        Args:
            threshold: Minimum similarity score to create an edge
        """
        self.logger.info("Adding product similarity edges...")
        
        product_ids = list(self.product_features.keys())
        if not product_ids:
            return
            
        feature_matrix = np.array([self.product_features[pid] for pid in product_ids])
        
        # Calculate cosine similarity matrix between products
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Add edges for sufficiently similar products
        edge_count = 0
        for i in tqdm(range(len(product_ids)), desc="Calculating product similarities"):
            for j in range(i+1, len(product_ids)):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    self.G.add_edge(
                        f"product_{product_ids[i]}", 
                        f"product_{product_ids[j]}", 
                        weight=similarity,
                        relation_type='similar'
                    )
                    edge_count += 1
        
        self.logger.info(f"Added {edge_count} product similarity edges")
    
    def _build_user_profiles(self):
        """Build user profiles based on their ratings."""
        self.logger.info("Building user profiles...")
        
        user_ratings = self.ratings_df.groupby('user_id')
        
        for user_id, ratings in tqdm(user_ratings, total=len(user_ratings), desc="Building user profiles"):
            # Calculate average rating for this user
            avg_rating = ratings['rating'].mean()
            
            # Get all products rated by this user and their ratings
            rated_products = ratings[['product_id', 'rating']].values
            
            # Calculate category preferences
            category_preferences = np.zeros(len(self.categories))
            category_counts = np.zeros(len(self.categories))
            
            for product_id, rating in rated_products:
                if product_id in self.product_features:
                    # Get product categories
                    product_categories = self.product_features[product_id]
                    
                    # Normalize rating relative to user's average
                    normalized_rating = rating - avg_rating
                    
                    # Update category preferences
                    for i, has_category in enumerate(product_categories):
                        if has_category:
                            category_preferences[i] += normalized_rating
                            category_counts[i] += 1
          
            # Avoid division by zero
            category_counts[category_counts == 0] = 1
            
            # Calculate average preference for each category
            category_preferences = category_preferences / category_counts
            
            # Store user profile
            self.user_profiles[user_id] = {
                'avg_rating': avg_rating,
                'category_preferences': category_preferences,
                'rated_products': set(ratings['product_id'].values),
                'rating_count': len(ratings),
                'last_rating_time': ratings['timestamp'].max()
            }
    
    def get_personalized_recommendations(self, user_id, n=10):
        """
        Get personalized product recommendations for a user based on content similarity.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            
        Returns:
            A list of recommended product IDs
        """
        if user_id not in self.user_profiles:
            self.logger.warning(f"User {user_id} not found in profiles")
            return []
        
        user_profile = self.user_profiles[user_id]
        rated_products = user_profile['rated_products']
        category_preferences = user_profile['category_preferences']
        
        product_scores = []
        
        for product_id, features in self.product_features.items():
            if product_id not in rated_products:
                # Calculate content-based score
                content_score = np.dot(category_preferences, features)
                
                # Apply popularity factor
                product_rating_count = len(self.ratings_df[self.ratings_df['product_id'] == product_id])
                popularity_factor = np.log1p(product_rating_count) / 10
                
                # Combine scores
                final_score = content_score + popularity_factor
                
                product_scores.append((product_id, final_score))
        
        # Sort by score and return top n
        recommendations = sorted(product_scores, key=lambda x: x[1], reverse=True)[:n]
        return [product_id for product_id, _ in recommendations]
    
    def get_graph_based_recommendations(self, user_id, n=10, depth=2):
        """
        Get recommendations using graph traversal.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            depth: How many hops to traverse in the graph
            
        Returns:
            A list of recommended product IDs
        """
        if f"user_{user_id}" not in self.G:
            self.logger.warning(f"User {user_id} not found in graph")
            return []
        
        user_node = f"user_{user_id}"
        rated_products = set()
        candidate_scores = defaultdict(float)
        
        # Get products already rated by user
        for neighbor in self.G.neighbors(user_node):
            if neighbor.startswith("product_"):
                product_id = neighbor.split("_", 1)[1]  # Handle complex product IDs
                rated_products.add(product_id)
        
        # Initialize BFS
        paths = [(user_node, [])]
        visited = {user_node}
        
        # Explore graph up to specified depth
        for _ in range(depth):
            new_paths = []
            for node, path in paths:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = path + [(node, neighbor)]
                        new_paths.append((neighbor, new_path))
                        
                        # If we found a product node
                        if neighbor.startswith("product_"):
                            product_id = neighbor.split("_", 1)[1]
                            if product_id not in rated_products:
                                # Calculate path score based on edge weights
                                path_weight = 1.0
                                for i in range(len(new_path)):
                                    n1, n2 = new_path[i]
                                    edge_weight = self.G.edges[n1, n2].get('weight', 1.0)
                                    path_weight *= edge_weight
                                
                                path_score = path_weight / len(new_path)
                                candidate_scores[product_id] += path_score
            
            paths = new_paths
        
        # Sort by score and return top n
        recommendations = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [product_id for product_id, _ in recommendations]
    
    def get_hybrid_recommendations(self, user_id, n=10, alpha=0.5):
        """
        Get hybrid recommendations combining content-based and graph-based approaches.
        
        Args:
            user_id: The user ID to generate recommendations for
            n: Number of recommendations to return
            alpha: Weight for content-based recommendations (1-alpha for graph-based)
            
        Returns:
            A list of recommended product IDs
        """
        # Get content-based recommendations
        content_recs = self.get_personalized_recommendations(user_id, n=n*2)
        content_scores = {product_id: (n*2 - i)/n*2 for i, product_id in enumerate(content_recs)}
        
        # Get graph-based recommendations
        graph_recs = self.get_graph_based_recommendations(user_id, n=n*2)
        graph_scores = {product_id: (n*2 - i)/n*2 for i, product_id in enumerate(graph_recs)}
        
        # Combine scores
        combined_scores = defaultdict(float)
        all_products = set(content_scores.keys()).union(set(graph_scores.keys()))
        
        for product_id in all_products:
            content_score = content_scores.get(product_id, 0)
            graph_score = graph_scores.get(product_id, 0)
            combined_scores[product_id] = alpha * content_score + (1 - alpha) * graph_score
        
        # Sort and return top n
        recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [product_id for product_id, _ in recommendations]
    
    def create_temporal_train_test_split(self, test_days=30):
        """
        Create train-test split based on time (train on earlier data, test on later data).
        
        Args:
            test_days: Number of days at the end to use as test set
            
        Returns:
            Dictionary mapping user_id to set of test product_ids
        """
        if self.ratings_df is None:
            if not self.load_data():
                return {}
                
        # Sort ratings by timestamp
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        
        # Calculate cutoff timestamp
        min_timestamp = self.ratings_df['timestamp'].min()
        max_timestamp = self.ratings_df['timestamp'].max()
        
        # Get approx time range in days
        time_range_seconds = max_timestamp - min_timestamp
        time_range_days = time_range_seconds / (24 * 60 * 60)
        
        # Adjust if requested test period is longer than available data
        test_ratio = min(1.0, test_days / time_range_days)
        cutoff_timestamp = min_timestamp + (1 - test_ratio) * time_range_seconds
        
        # Split data
        train_data = self.ratings_df[self.ratings_df['timestamp'] <= cutoff_timestamp]
        test_data = self.ratings_df[self.ratings_df['timestamp'] > cutoff_timestamp]
        
        self.logger.info(f"Temporal split: {len(train_data)} training and {len(test_data)} testing ratings")
        
        # Create test set mapping
        test_set = {}
        for user_id in test_data['user_id'].unique():
            if user_id in train_data['user_id'].unique():  # Only include users in both sets
                test_set[user_id] = set(test_data[test_data['user_id'] == user_id]['product_id'])
        
        # Update knowledge graph with training data only
        self.ratings_df = train_data
        
        # Rebuild user profiles with training data only
        self._build_user_profiles()
        
        return test_set
    
    def get_recommendations(self, user_id, method='hybrid', n=10):
        """
        Get recommendations using the specified method.
        
        Args:
            user_id: The user ID to generate recommendations for
            method: 'content', 'graph', or 'hybrid'
            n: Number of recommendations to return
            
        Returns:
            A list of recommended product IDs
        """
        if method == 'content':
            return self.get_personalized_recommendations(user_id, n)
        elif method == 'graph':
            return self.get_graph_based_recommendations(user_id, n)
        elif method == 'hybrid':
            return self.get_hybrid_recommendations(user_id, n)
        else:
            self.logger.warning(f"Unknown recommendation method: {method}, using hybrid")
            return self.get_hybrid_recommendations(user_id, n)
    
    def enrich_with_amazon_metadata(self, metadata_file=None):
        """
        Enrich the knowledge graph with Amazon product metadata.
        
        Args:
            metadata_file: Path to metadata file (optional)
        """
        if metadata_file and os.path.exists(metadata_file):
            try:
                # Load metadata
                metadata_df = pd.read_csv(metadata_file)
                
                # Add metadata to products
                for _, product in self.products_df.iterrows():
                    product_id = product['product_id']
                    metadata = metadata_df[metadata_df['product_id'] == product_id]
                    
                    if not metadata.empty:
                        meta_row = metadata.iloc[0]
                        
                        # Add metadata to graph node
                        node_id = f"product_{product_id}"
                        if node_id in self.G:
                            for column in metadata_df.columns:
                                if column != 'product_id':
                                    self.G.nodes[node_id][column] = meta_row[column]
                
                self.logger.info("Successfully enriched graph with metadata")
                
            except Exception as e:
                self.logger.error(f"Error enriching with metadata: {e}")
        else:
            # Create synthetic metadata for demonstration
            self._create_synthetic_metadata()
    
    def _create_synthetic_metadata(self):
        """Create synthetic metadata for products."""
        brands = ['Amazon', 'Apple', 'Samsung', 'Sony', 'Microsoft', 'Google', 'Nike', 'Adidas']
        price_ranges = ['$0-25', '$25-50', '$50-100', '$100-250', '$250+']
        
        for _, product in self.products_df.iterrows():
            product_id = product['product_id']
            node_id = f"product_{product_id}"
            
            if node_id in self.G:
                # Add synthetic metadata
                self.G.nodes[node_id]['brand'] = np.random.choice(brands)
                self.G.nodes[node_id]['price_range'] = np.random.choice(price_ranges)
                self.G.nodes[node_id]['avg_rating'] = np.random.uniform(3.0, 5.0)
                self.G.nodes[node_id]['review_count'] = np.random.randint(10, 1000)
        
        self.logger.info("Created synthetic metadata for products")
    
    def _recommend_popular(self, user_id, n=10):
        """Get popular product recommendations."""
        # Get product popularity counts
        product_counts = self.ratings_df['product_id'].value_counts()
        
        # Get products already rated by the user
        if user_id in self.user_profiles:
            rated_products = self.user_profiles[user_id]['rated_products']
        else:
            rated_products = set()
        
        # Get popular products not rated by the user
        popular_products = []
        for product_id, count in product_counts.items():
            if product_id not in rated_products:
                popular_products.append(product_id)
                if len(popular_products) >= n:
                    break
                
        return popular_products
    
    def visualize_subgraph(self, center_node, depth=1, filename=None):
        """
        Visualize a subgraph around a specific node.
        
        Args:
            center_node: Center node (e.g., "user_ABC123" or "product_XYZ789")
            depth: How many hops to include
            filename: If specified, save figure to this file
        """
        nodes = {center_node}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                if node in self.G:
                    new_nodes.update(self.G.neighbors(node))
            nodes.update(new_nodes)
        
        subgraph = self.G.subgraph(nodes)
        
        node_colors = []
        for node in subgraph.nodes():
            if node.startswith('user'):
                node_colors.append('skyblue')
            elif node.startswith('product'):
                node_colors.append('lightgreen')
            elif node.startswith('category'):
                node_colors.append('orange')
            else:
                node_colors.append('lightgray')
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
        nx.draw_networkx_labels(subgraph, pos, font_size=8)
        plt.title(f"Subgraph around {center_node}")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()