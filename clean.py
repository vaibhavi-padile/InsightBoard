"""
Data Cleaning Module
Handles missing values, duplicates, and data standardization
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles all data cleaning operations
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataCleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cleaning_config = config.get('cleaning', {})
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main cleaning pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        
        df_clean = df.copy()
        
        # 1. Handle missing values
        df_clean = self.handle_missing_values(df_clean)
        
        # 2. Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # 3. Standardize categorical values
        df_clean = self.standardize_categories(df_clean)
        
        # 4. Validate data types
        df_clean = self.validate_dtypes(df_clean)
        
        logger.info("Data cleaning completed successfully")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        df_clean = df.copy()
        
        # Handle Item_Weight (numerical)
        if 'Item_Weight' in df_clean.columns:
            missing_before = df_clean['Item_Weight'].isnull().sum()
            df_clean['Item_Weight'] = df_clean.groupby('Item_Type')['Item_Weight'].transform(
                lambda x: x.fillna(x.median())
            )
            logger.info(f"  Item_Weight: Filled {missing_before} missing values with median by Item_Type")
        
        # Handle Outlet_Size (categorical)
        if 'Outlet_Size' in df_clean.columns:
            missing_before = df_clean['Outlet_Size'].isnull().sum()
            df_clean['Outlet_Size'] = df_clean.groupby('Outlet_Type')['Outlet_Size'].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium')
            )
            logger.info(f"  Outlet_Size: Filled {missing_before} missing values with mode by Outlet_Type")
        
        # Check remaining missing values
        remaining_missing = df_clean.isnull().sum().sum()
        logger.info(f"  Remaining missing values: {remaining_missing}")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        else:
            logger.info("No duplicates found")
        
        return df_clean
    
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized categories
        """
        logger.info("Standardizing categorical values...")
        
        df_clean = df.copy()
        
        # Standardize Item_Fat_Content
        if 'Item_Fat_Content' in df_clean.columns:
            fat_content_map = self.cleaning_config.get('fat_content_mapping', {
                'LF': 'Low Fat',
                'low fat': 'Low Fat',
                'reg': 'Regular'
            })
            
            df_clean['Item_Fat_Content'] = df_clean['Item_Fat_Content'].replace(fat_content_map)
            logger.info(f"  Standardized Item_Fat_Content: {df_clean['Item_Fat_Content'].unique()}")
        
        # Trim whitespace from all string columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_clean[col] = df_clean[col].str.strip()
        
        logger.info(f"  Trimmed whitespace from {len(string_columns)} string columns")
        
        return df_clean
    
    def validate_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        logger.info("Validating data types...")
        
        df_clean = df.copy()
        
        # Ensure numerical columns are numeric
        numeric_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 
                       'Outlet_Establishment_Year', 'Item_Outlet_Sales']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Ensure categorical columns are strings
        categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type',
                           'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
                           'Outlet_Type']
        
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        logger.info("Data type validation completed")
        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, column: str, 
                       method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numerical column
        
        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[column]))
            outliers = z_scores > 3
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Detected {outliers.sum()} outliers in {column} using {method} method")
        return outliers
    
    def get_cleaning_report(self, df_before: pd.DataFrame, 
                           df_after: pd.DataFrame) -> Dict:
        """
        Generate a report of cleaning operations
        
        Args:
            df_before: DataFrame before cleaning
            df_after: DataFrame after cleaning
            
        Returns:
            Dictionary with cleaning statistics
        """
        report = {
            'rows_before': len(df_before),
            'rows_after': len(df_after),
            'rows_removed': len(df_before) - len(df_after),
            'missing_before': df_before.isnull().sum().sum(),
            'missing_after': df_after.isnull().sum().sum(),
            'missing_resolved': df_before.isnull().sum().sum() - df_after.isnull().sum().sum()
        }
        
        logger.info("\n" + "="*50)
        logger.info("CLEANING REPORT")
        logger.info("="*50)
        for key, value in report.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50)
        
        return report
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Save cleaned data to CSV
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the file. If None, uses config path
        """
        path = output_path or self.config.get('data', {}).get('processed_path')
        
        df.to_csv(path, index=False)
        logger.info(f"Cleaned data saved to {path}")


if __name__ == "__main__":
    # Example usage
    import yaml
    from read import DataReader
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    reader = DataReader(config)
    df = reader.read_csv()
    
    # Clean data
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_data(df)
    
    # Get report
    report = cleaner.get_cleaning_report(df, df_clean)
    
    # Save cleaned data
    cleaner.save_cleaned_data(df_clean)