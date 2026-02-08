"""
Data Analysis Module
Performs statistical analysis and generates insights
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Performs comprehensive data analysis
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analysis_config = config.get('analysis', {})
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform complete analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive data analysis...")
        
        results = {
            'summary_statistics': self.get_summary_statistics(df),
            'sales_analysis': self.analyze_sales(df),
            'outlet_analysis': self.analyze_outlets(df),
            'item_analysis': self.analyze_items(df),
            'correlations': self.analyze_correlations(df),
            'insights': self.generate_insights(df)
        }
        
        logger.info("Analysis completed successfully")
        return results
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Calculating summary statistics...")
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_sales': float(df['Item_Outlet_Sales'].sum()),
            'avg_sales': float(df['Item_Outlet_Sales'].mean()),
            'median_sales': float(df['Item_Outlet_Sales'].median()),
            'std_sales': float(df['Item_Outlet_Sales'].std()),
            'min_sales': float(df['Item_Outlet_Sales'].min()),
            'max_sales': float(df['Item_Outlet_Sales'].max()),
            'unique_items': int(df['Item_Identifier'].nunique()),
            'unique_outlets': int(df['Outlet_Identifier'].nunique()),
            'unique_item_types': int(df['Item_Type'].nunique()),
            'avg_item_price': float(df['Item_MRP'].mean())
        }
        
        logger.info(f"  Total Sales: ${stats['total_sales']:,.2f}")
        logger.info(f"  Unique Items: {stats['unique_items']}")
        logger.info(f"  Unique Outlets: {stats['unique_outlets']}")
        
        return stats
    
    def analyze_sales(self, df: pd.DataFrame) -> Dict:
        """
        Analyze sales patterns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with sales analysis
        """
        logger.info("Analyzing sales patterns...")
        
        analysis = {
            'by_outlet_type': df.groupby('Outlet_Type')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count']).to_dict(),
            'by_location': df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count']).to_dict(),
            'by_outlet_size': df.groupby('Outlet_Size')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count']).to_dict(),
            'by_fat_content': df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].agg(['sum', 'mean', 'count']).to_dict()
        }
        
        # Top performing categories
        top_n = self.analysis_config.get('top_n_items', 10)
        analysis['top_item_types'] = df.groupby('Item_Type')['Item_Outlet_Sales'].sum().nlargest(top_n).to_dict()
        
        return analysis
    
    def analyze_outlets(self, df: pd.DataFrame) -> Dict:
        """
        Analyze outlet performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with outlet analysis
        """
        logger.info("Analyzing outlet performance...")
        
        outlet_stats = df.groupby('Outlet_Identifier').agg({
            'Item_Outlet_Sales': ['sum', 'mean', 'count'],
            'Outlet_Type': 'first',
            'Outlet_Size': 'first',
            'Outlet_Location_Type': 'first'
        })
        
        outlet_stats.columns = ['Total_Sales', 'Avg_Sales', 'Item_Count', 'Type', 'Size', 'Location']
        outlet_stats = outlet_stats.sort_values('Total_Sales', ascending=False)
        
        top_n = self.analysis_config.get('top_n_outlets', 10)
        
        analysis = {
            'outlet_performance': outlet_stats.to_dict(),
            'top_outlets': outlet_stats.head(top_n)[['Total_Sales', 'Avg_Sales', 'Item_Count']].to_dict(),
            'bottom_outlets': outlet_stats.tail(top_n)[['Total_Sales', 'Avg_Sales', 'Item_Count']].to_dict(),
            'avg_items_per_outlet': float(outlet_stats['Item_Count'].mean()),
            'outlet_sales_variance': float(outlet_stats['Total_Sales'].std())
        }
        
        logger.info(f"  Top performing outlet: {outlet_stats.index[0]} (${outlet_stats.iloc[0]['Total_Sales']:,.2f})")
        
        return analysis
    
    def analyze_items(self, df: pd.DataFrame) -> Dict:
        """
        Analyze item performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with item analysis
        """
        logger.info("Analyzing item performance...")
        
        item_stats = df.groupby('Item_Identifier').agg({
            'Item_Outlet_Sales': ['sum', 'mean', 'count'],
            'Item_Type': 'first',
            'Item_MRP': 'first',
            'Item_Fat_Content': 'first'
        })
        
        item_stats.columns = ['Total_Sales', 'Avg_Sales', 'Outlet_Count', 'Type', 'MRP', 'Fat_Content']
        item_stats = item_stats.sort_values('Total_Sales', ascending=False)
        
        top_n = self.analysis_config.get('top_n_items', 15)
        
        analysis = {
            'top_items': item_stats.head(top_n)[['Total_Sales', 'Type', 'MRP']].to_dict(),
            'items_by_type': df.groupby('Item_Type')['Item_Identifier'].nunique().to_dict(),
            'avg_price_by_type': df.groupby('Item_Type')['Item_MRP'].mean().to_dict(),
            'price_range': {
                'min': float(df['Item_MRP'].min()),
                'max': float(df['Item_MRP'].max()),
                'mean': float(df['Item_MRP'].mean())
            }
        }
        
        return analysis
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with correlation analysis
        """
        logger.info("Analyzing feature correlations...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'Item_Outlet_Sales' in numerical_cols:
            correlations = df[numerical_cols].corr()['Item_Outlet_Sales'].sort_values(ascending=False)
            
            analysis = {
                'top_positive_correlations': correlations.head(10).to_dict(),
                'top_negative_correlations': correlations.tail(10).to_dict(),
                'correlation_matrix': correlations.to_dict()
            }
            
            logger.info(f"  Strongest positive correlation: {correlations.index[1]} ({correlations.iloc[1]:.4f})")
            
            return analysis
        
        return {}
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        Generate business insights
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with key insights
        """
        logger.info("Generating business insights...")
        
        insights = {}
        
        # Best performing outlet type
        outlet_type_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].sum()
        insights['best_outlet_type'] = outlet_type_sales.idxmax()
        insights['best_outlet_type_sales'] = float(outlet_type_sales.max())
        
        # Best performing location
        location_sales = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].sum()
        insights['best_location'] = location_sales.idxmax()
        insights['best_location_sales'] = float(location_sales.max())
        
        # Best performing item type
        item_type_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].sum()
        insights['best_item_type'] = item_type_sales.idxmax()
        insights['best_item_type_sales'] = float(item_type_sales.max())
        
        # Fat content preference
        fat_content_sales = df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum()
        insights['preferred_fat_content'] = fat_content_sales.idxmax()
        insights['low_fat_percentage'] = float((fat_content_sales.get('Low Fat', 0) / fat_content_sales.sum()) * 100)
        
        # Price analysis
        if 'MRP_Band' in df.columns:
            price_band_sales = df.groupby('MRP_Band')['Item_Outlet_Sales'].sum()
            insights['best_price_band'] = price_band_sales.idxmax()
        
        # Outlet age analysis
        if 'Outlet_Age_Group' in df.columns:
            age_group_sales = df.groupby('Outlet_Age_Group')['Item_Outlet_Sales'].mean()
            insights['best_age_group'] = age_group_sales.idxmax()
        
        # Sales concentration
        total_sales = df['Item_Outlet_Sales'].sum()
        top_20_pct_items = int(len(df) * 0.2)
        top_items_sales = df.nlargest(top_20_pct_items, 'Item_Outlet_Sales')['Item_Outlet_Sales'].sum()
        insights['pareto_principle'] = float((top_items_sales / total_sales) * 100)
        
        logger.info("\nKey Insights:")
        logger.info(f"  Best Outlet Type: {insights['best_outlet_type']}")
        logger.info(f"  Best Location: {insights['best_location']}")
        logger.info(f"  Best Item Type: {insights['best_item_type']}")
        logger.info(f"  Top 20% items contribute: {insights['pareto_principle']:.1f}% of sales")
        
        return insights
    
    def save_analysis(self, analysis: Dict, output_path: str):
        """
        Save analysis results to JSON
        
        Args:
            analysis: Analysis results dictionary
            output_path: Path to save the JSON file
        """
        import json
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=4, default=str)
        
        logger.info(f"Analysis results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import yaml
    from read import DataReader
    from clean import DataCleaner
    from features import FeatureEngineer
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load, clean, and create features
    reader = DataReader(config)
    df = reader.read_csv()
    
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_data(df)
    
    engineer = FeatureEngineer(config)
    df_features = engineer.create_features(df_clean)
    
    # Analyze
    analyzer = DataAnalyzer(config)
    results = analyzer.analyze(df_features)
    
    # Save results
    analyzer.save_analysis(results, 'outputs/reports/analysis_results.json')