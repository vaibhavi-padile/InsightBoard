"""
InsightBoard - Retail Sales Analytics Pipeline
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .read import DataReader
from .clean import DataCleaner
from .features import FeatureEngineer
from .analyze import DataAnalyzer
from .visualize import DataVisualizer
from .pipeline import Pipeline

__all__ = [
    'DataReader',
    'DataCleaner', 
    'FeatureEngineer',
    'DataAnalyzer',
    'DataVisualizer',
    'Pipeline'
]