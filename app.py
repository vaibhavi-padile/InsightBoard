"""
InsightBoard - Streamlit Dashboard
100% BULLETPROOF VERSION - No UnboundLocalError possible
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="InsightBoard - Blinkit Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
            
            .stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* KPI label */
.stMetric label {
    color: #333333 !important;
}

/* KPI value */
.stMetric div[data-testid="stMetricValue"] {
    color: #111111 !important;
    font-weight: 700;
}

/* KPI delta */
.stMetric div[data-testid="stMetricDelta"] {
    color: #2E7D32 !important;
}


    h1 {color: #FF6B6B; padding-bottom: 10px;}
    h2 {color: #4ECDC4; padding-top: 20px;}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load data with fallback mechanism - Returns df and stage OR raises error"""
    data_paths = [
        ('data/features/blinkit_features.csv', 'features'),
        ('data/processed/blinkit_cleaned.csv', 'cleaned'),
        ('data/raw/blinkit.csv', 'raw')
    ]
    
    for path, stage in data_paths:
        try:
            df = pd.read_csv(path)
            return df, stage
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error reading {path}: {e}")
            continue
    
    # If we get here, no file was found
    raise FileNotFoundError("No data file found in any expected location")


def main():
    """Main application - Fixed to prevent UnboundLocalError"""
    
    # Initialize df as None FIRST
    df = None
    data_stage = None
    
    # Title
    st.title("🛒 InsightBoard - Blinkit Sales Analytics")
    st.markdown("**Comprehensive insights into retail performance**")
    st.markdown("---")
    
    # Try to load data
    try:
        df, data_stage = load_data()
    except FileNotFoundError:
        st.error("❌ No data file found!")
        st.markdown("""
        ### 📋 Please add your data file:
        
        **Option 1: Add raw data**
        ```
        InsightBoard/
        └── data/
            └── raw/
                └── blinkit.csv  ← PUT YOUR CSV HERE
        ```
        
        **Option 2: Run the pipeline**
        ```bash
        python run_pipeline.py
        ```
        
        **Expected locations (checked in order):**
        1. `data/features/blinkit_features.csv` ✨ (best - includes features)
        2. `data/processed/blinkit_cleaned.csv` ✅ (good - cleaned data)
        3. `data/raw/blinkit.csv` 📁 (minimum - raw data)
        """)
        st.stop()  # CRITICAL: Stop execution here
        return  # Extra safety - will never reach here but good practice
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.code(str(e))
        st.stop()  # CRITICAL: Stop execution here
        return  # Extra safety
    
    # AT THIS POINT, df is GUARANTEED to be defined
    # If it wasn't, we would have stopped above
    
    # Show data stage
    stage_colors = {'features': '🟢', 'cleaned': '🟡', 'raw': '🔴'}
    st.sidebar.success(f"{stage_colors.get(data_stage, '⚪')} Data: {data_stage.upper()}")
    
    if data_stage == 'raw':
        st.warning("⚠️ Using raw data. Run `python run_pipeline.py` for better features!")
    
    # Sidebar filters - df is GUARANTEED to exist here
    st.sidebar.header("🔍 Filters")
    
    outlet_types = ['All'] + sorted(df['Outlet_Type'].unique().tolist())
    selected_outlet = st.sidebar.selectbox("🏪 Outlet Type", outlet_types)
    
    location_types = ['All'] + sorted(df['Outlet_Location_Type'].unique().tolist())
    selected_location = st.sidebar.selectbox("📍 Location", location_types)
    
    outlet_sizes = ['All'] + sorted(df['Outlet_Size'].dropna().unique().tolist())
    selected_size = st.sidebar.selectbox("📏 Size", outlet_sizes)
    
    # Filter data
    filtered_df = df.copy()  # df MUST exist here
    
    if selected_outlet != 'All':
        filtered_df = filtered_df[filtered_df['Outlet_Type'] == selected_outlet]
    
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Outlet_Location_Type'] == selected_location]
    
    if selected_size != 'All':
        filtered_df = filtered_df[filtered_df['Outlet_Size'] == selected_size]
    
    # Show counts
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Records", f"{len(filtered_df):,} / {len(df):,}")
    
    # Key Metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = filtered_df['Item_Outlet_Sales'].sum()
    avg_sales = filtered_df['Item_Outlet_Sales'].mean()
    total_outlets = filtered_df['Outlet_Identifier'].nunique()
    total_items = filtered_df['Item_Identifier'].nunique()
    avg_price = filtered_df['Item_MRP'].mean()
    
    with col1:
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    with col2:
        st.metric("📈 Avg Sales", f"${avg_sales:,.2f}")
    with col3:
        st.metric("🏪 Outlets", f"{total_outlets:,}")
    with col4:
        st.metric("🛍️ Items", f"{total_items:,}")
    with col5:
        st.metric("💵 Avg Price", f"${avg_price:,.2f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🏪 Outlets", "🛍️ Products"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Outlet Type")
            outlet_sales = filtered_df.groupby('Outlet_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
            
            try:
                fig = px.bar(
                    x=outlet_sales.index, y=outlet_sales.values,
                    labels={'x': 'Outlet Type', 'y': 'Sales ($)'},
                    color=outlet_sales.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.bar_chart(outlet_sales)
        
        with col2:
            st.subheader("Sales by Location")
            location_sales = filtered_df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].sum()
            
            try:
                fig = px.pie(
                    values=location_sales.values,
                    names=location_sales.index,
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.bar_chart(location_sales)
        
        # Best performers
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            best_outlet = outlet_sales.idxmax()
            st.metric("🏆 Best Outlet Type", best_outlet)
        with col2:
            best_location = location_sales.idxmax()
            st.metric("📍 Best Location", best_location)
    
    with tab2:
        st.subheader("🏆 Top 10 Outlets")
        
        outlet_perf = filtered_df.groupby('Outlet_Identifier').agg({
            'Item_Outlet_Sales': ['sum', 'mean', 'count'],
            'Outlet_Type': 'first',
            'Outlet_Size': 'first'
        }).round(2)
        
        outlet_perf.columns = ['Total_Sales', 'Avg_Sales', 'Items', 'Type', 'Size']
        outlet_perf = outlet_perf.sort_values('Total_Sales', ascending=False).head(10)
        
        st.dataframe(outlet_perf, use_container_width=True)
        
        st.subheader("Sales by Size")
        size_sales = filtered_df.groupby('Outlet_Size')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
        st.bar_chart(size_sales)
    
    with tab3:
        st.subheader("Top 15 Item Types")
        
        item_sales = filtered_df.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(15)
        
        try:
            fig = px.bar(
                x=item_sales.values, y=item_sales.index,
                orientation='h',
                labels={'x': 'Sales ($)', 'y': 'Item Type'},
                color=item_sales.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.bar_chart(item_sales)
        
        st.subheader("Sales by Fat Content")
        fat_sales = filtered_df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(fat_sales)
        with col2:
            for fat_type in fat_sales.index:
                st.metric(f"{fat_type} Sales", f"${fat_sales[fat_type]:,.0f}")
    
    # Raw Data
    st.markdown("---")
    with st.expander("📋 View Data"):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Data",
            csv,
            "blinkit_filtered.csv",
            "text/csv"
        )
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 InsightBoard**")
    with col2:
        st.markdown(f"**📅 {len(df):,} records**")
    with col3:
        st.markdown("**⚡ Streamlit**")


if __name__ == "__main__":
    main()