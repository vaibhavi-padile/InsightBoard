"""
Data Visualization Module
Creates charts and plots for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Handles data visualization
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataVisualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_path = config.get('outputs', {}).get('figures', 'outputs/figures/')
        
        # Set style
        plt.style.use(self.viz_config.get('style', 'seaborn-v0_8-darkgrid'))
        sns.set_palette(self.viz_config.get('color_palette', 'husl'))
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
    
    def plot_sales_distribution(self, df: pd.DataFrame, save: bool = True):
        """
        Plot sales distribution
        
        Args:
            df: Input DataFrame
            save: Whether to save the figure
        """
        logger.info("Creating sales distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(df['Item_Outlet_Sales'], bins=50, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(df['Item_Outlet_Sales'].mean(), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(df['Item_Outlet_Sales'].median(), color='green', 
                       linestyle='--', linewidth=2, label='Median')
        axes[0].set_title('Distribution of Item Sales', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Sales Amount ($)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df['Item_Outlet_Sales'], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
        axes[1].set_title('Sales Distribution (Box Plot)', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Sales Amount ($)', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_path}sales_distribution.png', dpi=100, bbox_inches='tight')
            logger.info(f"  Saved: sales_distribution.png")
        
        plt.show()
        plt.close()
    
    def plot_outlet_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Plot outlet performance analysis
        
        Args:
            df: Input DataFrame
            save: Whether to save the figure
        """
        logger.info("Creating outlet analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Sales by Outlet Type
        outlet_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
        outlet_sales.plot(kind='bar', ax=axes[0, 0], color='coral', edgecolor='black')
        axes[0, 0].set_title('Total Sales by Outlet Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Outlet Type')
        axes[0, 0].set_ylabel('Total Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Sales by Location
        location_sales = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].sum()
        axes[0, 1].pie(location_sales.values, labels=location_sales.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Sales Distribution by Location', fontsize=14, fontweight='bold')
        
        # 3. Sales by Outlet Size
        size_sales = df.groupby('Outlet_Size')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
        size_sales.plot(kind='bar', ax=axes[1, 0], color='green', edgecolor='black')
        axes[1, 0].set_title('Total Sales by Outlet Size', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Outlet Size')
        axes[1, 0].set_ylabel('Total Sales ($)')
        axes[1, 0].tick_params(axis='x', rotation=0)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Average Sales Comparison
        avg_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
        avg_sales.plot(kind='barh', ax=axes[1, 1], color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Average Sales by Outlet Type', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Average Sales ($)')
        axes[1, 1].set_ylabel('Outlet Type')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_path}outlet_analysis.png', dpi=100, bbox_inches='tight')
            logger.info(f"  Saved: outlet_analysis.png")
        
        plt.show()
        plt.close()
    
    def plot_item_analysis(self, df: pd.DataFrame, top_n: int = 15, save: bool = True):
        """
        Plot item performance analysis
        
        Args:
            df: Input DataFrame
            top_n: Number of top items to show
            save: Whether to save the figure
        """
        logger.info("Creating item analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top Item Types
        top_items = df.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(top_n)
        top_items.plot(kind='barh', ax=axes[0, 0], color='teal', edgecolor='black')
        axes[0, 0].set_title(f'Top {top_n} Item Types by Revenue', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Total Sales ($)')
        axes[0, 0].set_ylabel('Item Type')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Sales by Fat Content
        fat_sales = df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum()
        fat_sales.plot(kind='bar', ax=axes[0, 1], color=['lightcoral', 'lightgreen'], 
                      edgecolor='black')
        axes[0, 1].set_title('Sales by Fat Content', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Fat Content')
        axes[0, 1].set_ylabel('Total Sales ($)')
        axes[0, 1].tick_params(axis='x', rotation=0)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Price vs Sales
        axes[1, 0].scatter(df['Item_MRP'], df['Item_Outlet_Sales'], 
                          alpha=0.5, s=30, c='purple', edgecolors='black')
        axes[1, 0].set_title('Item Price vs Sales', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Item MRP ($)')
        axes[1, 0].set_ylabel('Sales ($)')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Sales by MRP Band
        if 'MRP_Band' in df.columns:
            mrp_sales = df.groupby('MRP_Band')['Item_Outlet_Sales'].sum()
            mrp_sales.plot(kind='bar', ax=axes[1, 1], color='orange', edgecolor='black')
            axes[1, 1].set_title('Sales by Price Band', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Price Band')
            axes[1, 1].set_ylabel('Total Sales ($)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_path}item_analysis.png', dpi=100, bbox_inches='tight')
            logger.info(f"  Saved: item_analysis.png")
        
        plt.show()
        plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save: bool = True):
        """
        Plot correlation heatmap
        
        Args:
            df: Input DataFrame
            save: Whether to save the figure
        """
        logger.info("Creating correlation heatmap...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_path}correlation_heatmap.png', dpi=100, bbox_inches='tight')
            logger.info(f"  Saved: correlation_heatmap.png")
        
        plt.show()
        plt.close()
    
    def plot_age_analysis(self, df: pd.DataFrame, save: bool = True):
        """
        Plot outlet age analysis
        
        Args:
            df: Input DataFrame
            save: Whether to save the figure
        """
        if 'Outlet_Age_Group' not in df.columns:
            logger.warning("Outlet_Age_Group column not found, skipping age analysis plot")
            return
        
        logger.info("Creating outlet age analysis plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total sales by age group
        age_sales = df.groupby('Outlet_Age_Group')['Item_Outlet_Sales'].sum()
        age_sales.plot(kind='bar', ax=axes[0], color='brown', edgecolor='black')
        axes[0].set_title('Total Sales by Outlet Age Group', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Age Group')
        axes[0].set_ylabel('Total Sales ($)')
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Average sales by age group
        avg_age_sales = df.groupby('Outlet_Age_Group')['Item_Outlet_Sales'].mean()
        avg_age_sales.plot(kind='bar', ax=axes[1], color='olive', edgecolor='black')
        axes[1].set_title('Average Sales by Outlet Age Group', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Age Group')
        axes[1].set_ylabel('Average Sales ($)')
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_path}age_analysis.png', dpi=100, bbox_inches='tight')
            logger.info(f"  Saved: age_analysis.png")
        
        plt.show()
        plt.close()
    
    def create_dashboard(self, df: pd.DataFrame, save: bool = True):
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            df: Input DataFrame
            save: Whether to save individual figures
        """
        logger.info("Creating comprehensive visualization dashboard...")
        
        self.plot_sales_distribution(df, save)
        self.plot_outlet_analysis(df, save)
        self.plot_item_analysis(df, save)
        self.plot_correlation_heatmap(df, save)
        self.plot_age_analysis(df, save)
        
        logger.info(f"All visualizations saved to {self.output_path}")


if __name__ == "__main__":
    # Example usage
    import yaml
    from read import DataReader
    from clean import DataCleaner
    from features import FeatureEngineer
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and prepare data
    reader = DataReader(config)
    df = reader.read_csv()
    
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean_data(df)
    
    engineer = FeatureEngineer(config)
    df_features = engineer.create_features(df_clean)
    
    # Create visualizations
    visualizer = DataVisualizer(config)
    visualizer.create_dashboard(df_features)