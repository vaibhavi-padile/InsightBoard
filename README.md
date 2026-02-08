# 📊 InsightBoard - Retail Sales Analytics Pipeline

A production-grade data analytics pipeline for retail sales data, featuring automated data processing, feature engineering, comprehensive analysis, and interactive dashboards.

## 🎯 Project Overview

InsightBoard consolidates multi-source retail sales data and delivers actionable insights through:
- **Automated ETL pipeline** with data cleaning and validation
- **Advanced feature engineering** for better analytics
- **Comprehensive statistical analysis** with business insights
- **Interactive Streamlit dashboard** for KPI tracking
- **Professional visualizations** for data-driven decisions

## 🏗️ Project Structure

```
InsightBoard/
│
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Cleaned data
│   └── features/               # Feature-engineered data
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── read.py                # Data loading module
│   ├── clean.py               # Data cleaning module
│   ├── features.py            # Feature engineering module
│   ├── analyze.py             # Analysis module
│   ├── visualize.py           # Visualization module
│   └── pipeline.py            # Pipeline orchestrator
│
├── notebooks/
│   └── eda_analysis.ipynb     # Exploratory data analysis
│
├── outputs/
│   ├── figures/               # Generated visualizations
│   └── reports/               # Analysis reports (JSON)
│
├── streamlit_app/
│   ├── app.py                 # Streamlit dashboard
│   ├── utils.py               # Dashboard utilities
│   └── config.py              # Dashboard configuration
│
├── main.py                    # Main entry point
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/InsightBoard.git
cd InsightBoard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Full Pipeline (Recommended for first run):**
```bash
python main.py --run full
```

**Specific Steps:**
```bash
# Data cleaning only
python main.py --run clean

# Feature engineering only
python main.py --run features

# Analysis only
python main.py --run analyze

# Custom steps
python main.py --steps read clean features
```

**Skip Visualizations:**
```bash
python main.py --run full --no-viz
```

### Running the Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

## 📋 Pipeline Stages

### 1. **Data Reading** (`read.py`)
- Load data from CSV, Excel, or SQL databases
- Validate data structure and required columns
- Generate data quality reports

### 2. **Data Cleaning** (`clean.py`)
- Handle missing values intelligently
- Remove duplicates
- Standardize categorical values
- Detect and handle outliers
- Data type validation

### 3. **Feature Engineering** (`features.py`)
- **Time-based features**: Outlet age, age groups
- **Price features**: MRP bands, price per weight
- **Visibility features**: Visibility categories, low visibility flags
- **Sales features**: Performance categories, sales scores
- **Aggregated features**: Average sales by item/outlet
- **Encoded features**: Numerical encoding for categorical variables

### 4. **Data Analysis** (`analyze.py`)
- Summary statistics
- Sales pattern analysis
- Outlet performance metrics
- Item performance analysis
- Correlation analysis
- Business insights generation

### 5. **Visualization** (`visualize.py`)
- Sales distribution plots
- Outlet performance charts
- Item analysis visualizations
- Correlation heatmaps
- Age-based analysis

## 📊 Key Features

### Data Processing
- ✅ Automated missing value imputation
- ✅ Categorical standardization
- ✅ Outlier detection
- ✅ Data validation

### Feature Engineering
- ✅ 15+ engineered features
- ✅ Temporal features
- ✅ Aggregation features
- ✅ Categorical encoding

### Analysis
- ✅ Comprehensive statistical analysis
- ✅ Business insights generation
- ✅ Performance metrics
- ✅ Correlation analysis

### Visualization
- ✅ 10+ professional charts
- ✅ Interactive dashboards
- ✅ Export-ready figures
- ✅ Customizable themes

## 🎨 Configuration

Edit `config.yaml` to customize:

```yaml
# Data paths
data:
  raw_path: "data/raw/blinkit.csv"
  processed_path: "data/processed/blinkit_cleaned.csv"
  features_path: "data/features/blinkit_features.csv"

# Feature engineering settings
features:
  current_year: 2026
  outlet_age_bins: [0, 10, 20, 40]
  mrp_bins: [0, 70, 140, 210, 300]

# Analysis settings
analysis:
  top_n_items: 15
  top_n_outlets: 10
```

## 📈 Output Files

### Processed Data
- `data/processed/blinkit_cleaned.csv` - Cleaned dataset
- `data/features/blinkit_features.csv` - Feature-engineered dataset

### Analysis Reports
- `outputs/reports/analysis_results.json` - Complete analysis results

### Visualizations
- `outputs/figures/sales_distribution.png`
- `outputs/figures/outlet_analysis.png`
- `outputs/figures/item_analysis.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/age_analysis.png`

## 🔧 Advanced Usage

### Using as a Python Module

```python
import yaml
from src.pipeline import Pipeline

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run pipeline
pipeline = Pipeline(config)
results = pipeline.run_full_pipeline()

# Get processed data
data = pipeline.get_data('features')

# Get analysis results
analysis = pipeline.get_analysis()
```

### Custom Analysis

```python
from src.read import DataReader
from src.analyze import DataAnalyzer

reader = DataReader(config)
df = reader.read_csv()

analyzer = DataAnalyzer(config)
insights = analyzer.generate_insights(df)
```

## 📊 Sample Insights

The pipeline automatically generates insights such as:
- Best performing outlet types and locations
- Top revenue-generating item categories
- Sales concentration analysis (Pareto principle)
- Price-performance relationships
- Outlet age impact on sales

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset: Blinkit Sales Data
- Built with: Python, Pandas, Matplotlib, Streamlit
- Inspired by modern MLOps practices

---

**⭐ If you find this project useful, please give it a star!**