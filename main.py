"""
InsightBoard - Main Entry Point
Run the complete data analytics pipeline
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='InsightBoard - Retail Sales Analytics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --run full
  
  # Run specific stage
  python main.py --run clean
  
  # Run custom steps
  python main.py --steps read clean features
  
  # Skip visualizations
  python main.py --run full --no-viz
        """
    )
    
    parser.add_argument(
        '--run',
        type=str,
        choices=['full', 'clean', 'features', 'analyze', 'visualize'],
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['read', 'clean', 'features', 'analyze', 'visualize'],
        help='Specific steps to run'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualizations'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    return parser.parse_args()


def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'outputs/figures',
        'outputs/reports',
        'outputs/models'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directory structure created")


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  INSIGHTBOARD - RETAIL SALES ANALYTICS PIPELINE")
    print("="*60 + "\n")
    
    args = parse_args()
    
    # Create directories
    create_directories()
    
    # Load configuration
    try:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✓ Configuration loaded successfully")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default config.yaml...")
        # Could create default config here
        return
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = Pipeline(config)
    
    # Run pipeline based on arguments
    try:
        if args.run == 'full':
            logger.info("Running FULL pipeline...")
            results = pipeline.run_full_pipeline(create_visualizations=not args.no_viz)
            
        elif args.run:
            logger.info(f"Running {args.run.upper()} stage...")
            results = pipeline.run_partial_pipeline([args.run])
            
        elif args.steps:
            logger.info(f"Running custom steps: {[s.upper() for s in args.steps]}")
            results = pipeline.run_partial_pipeline(args.steps)
            
        else:
            logger.info("No arguments provided. Running FULL pipeline...")
            results = pipeline.run_full_pipeline(create_visualizations=not args.no_viz)
        
        # Print summary
        print("\n" + "="*60)
        print("  PIPELINE SUMMARY")
        print("="*60)
        
        summary = pipeline.get_pipeline_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        if 'elapsed_time' in results:
            print(f"\n  Total Time: {results['elapsed_time']:.2f} seconds")
        
        print("="*60)
        print("  STATUS: SUCCESS ✓")
        print("="*60 + "\n")
        
        # Show next steps
        print("Next Steps:")
        print("  1. Run Streamlit dashboard: streamlit run app.py")
        print("  2. Check outputs in: outputs/figures/ and outputs/reports/")
        print("  3. Review processed data in: data/processed/ and data/features/")
        print()
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        print("\n" + "="*60)
        print("  STATUS: FAILED ✗")
        print("="*60 + "\n")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())