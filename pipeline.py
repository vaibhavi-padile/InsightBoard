import logging
import time
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import sys

# Add src directory to path if needed
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Use absolute imports instead of relative imports
from read import DataReader
from clean import DataCleaner
from features import FeatureEngineer
from analyze import DataAnalyzer
from visualize import DataVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main pipeline orchestrator
    """
    
    def __init__(self, config: dict):
        """
        Initialize Pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize all components
        self.reader = DataReader(config)
        self.cleaner = DataCleaner(config)
        self.engineer = FeatureEngineer(config)
        self.analyzer = DataAnalyzer(config)
        self.visualizer = DataVisualizer(config)
        
        # Data storage
        self.raw_data = None
        self.clean_data = None
        self.feature_data = None
        self.analysis_results = None
        
    def run_full_pipeline(self, create_visualizations: bool = True) -> Dict:
        """
        Run the complete data pipeline
        
        Args:
            create_visualizations: Whether to create visualization dashboard
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        logger.info("="*60)
        logger.info("STARTING FULL DATA PIPELINE")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Step 1: Read Data
            self._step_read_data()
            
            # Step 2: Clean Data
            self._step_clean_data()
            
            # Step 3: Engineer Features
            self._step_engineer_features()
            
            # Step 4: Analyze Data
            self._step_analyze_data()
            
            # Step 5: Create Visualizations
            if create_visualizations:
                self._step_create_visualizations()
            
            # Step 6: Save All Outputs
            self._step_save_outputs()
            
            elapsed_time = time.time() - start_time
            
            logger.info("="*60)
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed_time:.2f} seconds")
            logger.info("="*60)
            
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'rows_processed': len(self.feature_data),
                'features_created': len(self.feature_data.columns),
                'analysis_results': self.analysis_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _step_read_data(self):
        """Step 1: Read raw data"""
        logger.info("\n[STEP 1] Reading raw data...")
        
        self.raw_data = self.reader.read_csv()
        info = self.reader.get_data_info(self.raw_data)
        
        # Validate required columns
        required_cols = ['Item_Identifier', 'Item_Outlet_Sales', 'Outlet_Type', 
                        'Item_Type', 'Outlet_Identifier']
        self.reader.validate_columns(self.raw_data, required_cols)
        
        logger.info(f"✓ Data loaded: {info['shape'][0]} rows, {info['shape'][1]} columns")
    
    def _step_clean_data(self):
        """Step 2: Clean data"""
        logger.info("\n[STEP 2] Cleaning data...")
        
        self.clean_data = self.cleaner.clean_data(self.raw_data)
        report = self.cleaner.get_cleaning_report(self.raw_data, self.clean_data)
        
        logger.info(f"✓ Data cleaned: {report['missing_resolved']} missing values resolved")
    
    def _step_engineer_features(self):
        """Step 3: Engineer features"""
        logger.info("\n[STEP 3] Engineering features...")
        
        self.feature_data = self.engineer.create_features(self.clean_data)
        
        new_features = len(self.feature_data.columns) - len(self.clean_data.columns)
        logger.info(f"✓ Features created: {new_features} new features")
    
    def _step_analyze_data(self):
        """Step 4: Analyze data"""
        logger.info("\n[STEP 4] Analyzing data...")
        
        self.analysis_results = self.analyzer.analyze(self.feature_data)
        
        logger.info(f"✓ Analysis completed: {len(self.analysis_results)} analysis sections")
    
    def _step_create_visualizations(self):
        """Step 5: Create visualizations"""
        logger.info("\n[STEP 5] Creating visualizations...")
        
        self.visualizer.create_dashboard(self.feature_data, save=True)
        
        logger.info("✓ Visualizations created and saved")
    
    def _step_save_outputs(self):
        """Step 6: Save all outputs"""
        logger.info("\n[STEP 6] Saving outputs...")
        
        # Save cleaned data
        self.cleaner.save_cleaned_data(self.clean_data)
        
        # Save feature data
        self.engineer.save_features(self.feature_data)
        
        # Save analysis results
        output_path = Path(self.config.get('outputs', {}).get('reports', 'outputs/reports/'))
        output_path.mkdir(parents=True, exist_ok=True)
        self.analyzer.save_analysis(self.analysis_results, 
                                   str(output_path / 'analysis_results.json'))
        
        logger.info("✓ All outputs saved")
    
    def run_partial_pipeline(self, steps: list) -> Dict:
        """
        Run only specific steps of the pipeline
        
        Args:
            steps: List of step names to run
                   Options: ['read', 'clean', 'features', 'analyze', 'visualize']
        
        Returns:
            Dictionary with results
        """
        logger.info(f"Running partial pipeline with steps: {steps}")
        
        step_mapping = {
            'read': self._step_read_data,
            'clean': self._step_clean_data,
            'features': self._step_engineer_features,
            'analyze': self._step_analyze_data,
            'visualize': self._step_create_visualizations
        }
        
        for step in steps:
            if step in step_mapping:
                step_mapping[step]()
            else:
                logger.warning(f"Unknown step: {step}")
        
        return {'status': 'success', 'steps_completed': steps}
    
    def get_data(self, stage: str = 'features') -> pd.DataFrame:
        """
        Get data from a specific pipeline stage
        
        Args:
            stage: Pipeline stage ('raw', 'clean', or 'features')
            
        Returns:
            DataFrame from requested stage
        """
        stage_mapping = {
            'raw': self.raw_data,
            'clean': self.clean_data,
            'features': self.feature_data
        }
        
        if stage not in stage_mapping:
            raise ValueError(f"Unknown stage: {stage}")
        
        data = stage_mapping[stage]
        
        if data is None:
            raise ValueError(f"No data available for stage: {stage}. Run pipeline first.")
        
        return data
    
    def get_analysis(self) -> Dict:
        """
        Get analysis results
        
        Returns:
            Dictionary with analysis results
        """
        if self.analysis_results is None:
            raise ValueError("No analysis results available. Run analysis step first.")
        
        return self.analysis_results
    
    def get_pipeline_summary(self) -> Dict:
        """
        Get summary of pipeline execution
        
        Returns:
            Dictionary with pipeline summary
        """
        summary = {
            'raw_data_loaded': self.raw_data is not None,
            'data_cleaned': self.clean_data is not None,
            'features_created': self.feature_data is not None,
            'analysis_completed': self.analysis_results is not None
        }
        
        if self.feature_data is not None:
            summary['total_rows'] = len(self.feature_data)
            summary['total_features'] = len(self.feature_data.columns)
        
        return summary


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run_full_pipeline(create_visualizations=True)
    
    # Get summary
    summary = pipeline.get_pipeline_summary()
    print("\nPipeline Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get final data
    final_data = pipeline.get_data('features')
    print(f"\nFinal dataset shape: {final_data.shape}")