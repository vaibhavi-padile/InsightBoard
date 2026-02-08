"""
Data Reading Module
Handles loading data from various sources
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataReader:
    """
    Handles reading data from various file formats
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataReader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = config.get('data', {}).get('raw_path')
        
    def read_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from CSV file
        
        Args:
            file_path: Path to CSV file. If None, uses config path
            
        Returns:
            DataFrame containing the data
        """
        path = file_path or self.data_path
        
        try:
            logger.info(f"Reading data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise
    
    def read_excel(self, file_path: str, sheet_name: str = 0) -> pd.DataFrame:
        """
        Read data from Excel file
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            
        Returns:
            DataFrame containing the data
        """
        try:
            logger.info(f"Reading Excel data from {file_path}, sheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            logger.info(f"Successfully loaded {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error reading Excel: {str(e)}")
            raise
    
    def read_sql(self, query: str, connection_string: str) -> pd.DataFrame:
        """
        Read data from SQL database
        
        Args:
            query: SQL query string
            connection_string: Database connection string
            
        Returns:
            DataFrame containing query results
        """
        try:
            import sqlalchemy
            logger.info(f"Executing SQL query")
            engine = sqlalchemy.create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Successfully loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Error reading from SQL: {str(e)}")
            raise
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info("Dataset Info:")
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Missing values: {sum(info['missing_values'].values())}")
        logger.info(f"  Duplicates: {info['duplicates']}")
        logger.info(f"  Memory: {info['memory_usage_mb']:.2f} MB")
        
        return info
    
    def validate_columns(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that required columns exist in DataFrame
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            True if all required columns exist, False otherwise
        """
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        logger.info("All required columns present")
        return True


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    reader = DataReader(config)
    df = reader.read_csv()
    info = reader.get_data_info(df)
    
    required_cols = ['Item_Identifier', 'Item_Outlet_Sales', 'Outlet_Type']
    reader.validate_columns(df, required_cols)