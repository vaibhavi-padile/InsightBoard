"""
Feature Engineering Module
Creates new features from existing data
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering operations
    """
    
    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.features_config = config.get('features', {})
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Starting feature engineering pipeline...")
        
        df_features = df.copy()
        
        # 1. Time-based features
        df_features = self.create_time_features(df_features)
        
        # 2. Price-based features
        df_features = self.create_price_features(df_features)
        
        # 3. Visibility features
        df_features = self.create_visibility_features(df_features)
        
        # 4. Sales performance features
        df_features = self.create_sales_features(df_features)
        
        # 5. Aggregated features
        df_features = self.create_aggregated_features(df_features)
        
        # 6. Categorical encoding
        df_features = self.encode_categories(df_features)
        
        logger.info(f"Feature engineering completed. Total features: {len(df_features.columns)}")
        return df_features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        logger.info("Creating time-based features...")
        
        df_features = df.copy()
        
        # Outlet Age
        current_year = self.features_config.get('current_year', 2026)
        df_features['Outlet_Age'] = current_year - df_features['Outlet_Establishment_Year']
        
        # Outlet Age Groups
        age_bins = self.features_config.get('outlet_age_bins', [0, 10, 20, 40])
        age_labels = self.features_config.get('outlet_age_labels', ['New', 'Mid', 'Old'])
        
        df_features['Outlet_Age_Group'] = pd.cut(
            df_features['Outlet_Age'],
            bins=age_bins,
            labels=age_labels
        )
        
        logger.info(f"  Created: Outlet_Age, Outlet_Age_Group")
        
        return df_features
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with price features
        """
        logger.info("Creating price-based features...")
        
        df_features = df.copy()
        
        # MRP Bands
        mrp_bins = self.features_config.get('mrp_bins', [0, 70, 140, 210, 300])
        mrp_labels = self.features_config.get('mrp_labels', ['Low', 'Medium', 'High', 'Premium'])
        
        df_features['MRP_Band'] = pd.cut(
            df_features['Item_MRP'],
            bins=mrp_bins,
            labels=mrp_labels
        )
        
        # Price per unit weight (if weight is available)
        if 'Item_Weight' in df_features.columns:
            df_features['Price_Per_Weight'] = df_features['Item_MRP'] / df_features['Item_Weight']
        
        logger.info(f"  Created: MRP_Band, Price_Per_Weight")
        
        return df_features
    
    def create_visibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create visibility-based features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with visibility features
        """
        logger.info("Creating visibility features...")
        
        df_features = df.copy()
        
        # Low visibility flag
        visibility_threshold = self.features_config.get('visibility_threshold', 0.05)
        df_features['Low_Visibility'] = (df_features['Item_Visibility'] < visibility_threshold).astype(int)
        
        # Visibility categories
        df_features['Visibility_Category'] = pd.cut(
            df_features['Item_Visibility'],
            bins=[0, 0.05, 0.10, 0.20, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
        
        logger.info(f"  Created: Low_Visibility, Visibility_Category")
        
        return df_features
    
    def create_sales_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sales performance features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sales features
        """
        logger.info("Creating sales performance features...")
        
        df_features = df.copy()
        
        # Sales Performance Categories
        sales_quantiles = self.features_config.get('sales_quantiles', 3)
        df_features['Sales_Performance'] = pd.qcut(
            df_features['Item_Outlet_Sales'],
            q=sales_quantiles,
            labels=['Low', 'Medium', 'High']
        )
        
        # Sales Score (0-4 scale)
        df_features['Sales_Score'] = pd.qcut(
            df_features['Item_Outlet_Sales'],
            q=5,
            labels=False
        )
        
        # Sales Contribution
        total_sales = df_features['Item_Outlet_Sales'].sum()
        df_features['Sales_Contribution_%'] = (df_features['Item_Outlet_Sales'] / total_sales) * 100
        
        logger.info(f"  Created: Sales_Performance, Sales_Score, Sales_Contribution_%")
        
        return df_features
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features based on groupings
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aggregated features
        """
        logger.info("Creating aggregated features...")
        
        df_features = df.copy()
        
        # Average sales by Item
        df_features['Avg_Item_Sales'] = df_features.groupby('Item_Identifier')['Item_Outlet_Sales'].transform('mean')
        
        # Average sales by Outlet
        df_features['Avg_Outlet_Sales'] = df_features.groupby('Outlet_Identifier')['Item_Outlet_Sales'].transform('mean')
        
        # Item count by Type
        df_features['Item_Type_Count'] = df_features.groupby('Item_Type')['Item_Identifier'].transform('count')
        
        # Outlet item count
        df_features['Outlet_Item_Count'] = df_features.groupby('Outlet_Identifier')['Item_Identifier'].transform('count')
        
        # Sales rank within outlet
        df_features['Sales_Rank_In_Outlet'] = df_features.groupby('Outlet_Identifier')['Item_Outlet_Sales'].rank(ascending=False)
        
        logger.info(f"  Created: Avg_Item_Sales, Avg_Outlet_Sales, Item_Type_Count, Outlet_Item_Count, Sales_Rank_In_Outlet")
        
        return df_features
    
    def encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categories
        """
        logger.info("Encoding categorical variables...")
        
        df_features = df.copy()
        
        # Outlet Size Code
        if 'Outlet_Size' in df_features.columns:
            df_features['Outlet_Size_Code'] = df_features['Outlet_Size'].map({
                'Small': 1,
                'Medium': 2,
                'High': 3
            })
        
        # Location Type Code
        if 'Outlet_Location_Type' in df_features.columns:
            df_features['Location_Type_Code'] = df_features['Outlet_Location_Type'].map({
                'Tier 1': 1,
                'Tier 2': 2,
                'Tier 3': 3
            })
        
        # Fat Content Binary
        if 'Item_Fat_Content' in df_features.columns:
            df_features['Is_Low_Fat'] = (df_features['Item_Fat_Content'] == 'Low Fat').astype(int)
        
        logger.info(f"  Created: Outlet_Size_Code, Location_Type_Code, Is_Low_Fat")
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_features = df.copy()
        
        # MRP x Visibility
        df_features['MRP_x_Visibility'] = df_features['Item_MRP'] * df_features['Item_Visibility']
        
        # Outlet Age x Size
        if 'Outlet_Size_Code' in df_features.columns:
            df_features['Age_x_Size'] = df_features['Outlet_Age'] * df_features['Outlet_Size_Code']
        
        logger.info(f"  Created interaction features")
        
        return df_features
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'Item_Outlet_Sales') -> pd.DataFrame:
        """
        Calculate feature importance using correlation
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame with feature correlations
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        feature_importance = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values
        })
        
        logger.info(f"\nTop 10 Most Important Features:")
        logger.info(feature_importance.head(10).to_string(index=False))
        
        return feature_importance
    
    def save_features(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Save feature-engineered data
        
        Args:
            df: DataFrame with features
            output_path: Path to save the file
        """
        path = output_path or self.config.get('data', {}).get('features_path')
        
        df.to_csv(path, index=False)
        logger.info(f"Feature data saved to {path}")


if __name__ == "__main__":
    # Example usage
    import yaml
    from read import DataReader
    from clean import DataCleaner
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and clean data
    reader = DataReader(config)
    df = reader.read_csv()
    
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_data(df)
    
    # Create features
    engineer = FeatureEngineer(config)
    df_features = engineer.create_features(df_clean)
    
    # Get feature importance
    importance = engineer.get_feature_importance(df_features)
    
    # Save features
    engineer.save_features(df_features)