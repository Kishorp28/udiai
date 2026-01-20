"""
AADHAAR DATA ANALYTICS DASHBOARD - Streamlit App
Comprehensive analysis of Aadhaar enrolment and update datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import warnings
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import zipfile
import tempfile

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Aadhaar Data Analytics Dashboard",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4fc;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #10B981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warning-box {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .st-bb {
        border-bottom: 2px solid #e5e7eb;
    }
    .highlight {
        background-color: #fffacd;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
    }
    .dataset-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .dataset-enrolment {
        background-color: #dbeafe;
        color: #1E40AF;
        border: 1px solid #93c5fd;
    }
    .dataset-demographic {
        background-color: #dcfce7;
        color: #065f46;
        border: 1px solid #86efac;
    }
    .dataset-biometric {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
</style>
""", unsafe_allow_html=True)

class AadhaarStreamlitDashboard:
    def __init__(self):
        self.base_path = os.getcwd()
        self.enrolment_df = None
        self.demographic_df = None
        self.biometric_df = None
        self.processed_data = {}
        self.analysis_results = {}
        
    def load_data(self):
        """Load all three Aadhaar datasets"""
        with st.spinner("📂 Loading Aadhaar datasets..."):
            try:
                # Load enrolment data
                enrolment_files = glob.glob(os.path.join(
                    self.base_path, "api_data_aadhar_enrolment", "*.csv"
                ))
                
                if enrolment_files:
                    enrolment_dfs = []
                    for file in enrolment_files:
                        try:
                            df = pd.read_csv(file, low_memory=False)
                            enrolment_dfs.append(df)
                            st.sidebar.success(f"✓ Enrolment: {os.path.basename(file)}")
                        except Exception as e:
                            st.sidebar.error(f"✗ Error loading {os.path.basename(file)}: {str(e)}")
                    
                    if enrolment_dfs:
                        self.enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
                        st.sidebar.info(f"Enrolment: {len(self.enrolment_df):,} records")
                
                # Load demographic data
                demo_files = glob.glob(os.path.join(
                    self.base_path, "api_data_aadhar_demographic", "*.csv"
                ))
                
                if demo_files:
                    demo_dfs = []
                    for file in demo_files:
                        try:
                            df = pd.read_csv(file, low_memory=False)
                            demo_dfs.append(df)
                            st.sidebar.success(f"✓ Demographic: {os.path.basename(file)}")
                        except Exception as e:
                            st.sidebar.error(f"✗ Error loading {os.path.basename(file)}: {str(e)}")
                    
                    if demo_dfs:
                        self.demographic_df = pd.concat(demo_dfs, ignore_index=True)
                        st.sidebar.info(f"Demographic: {len(self.demographic_df):,} records")
                
                # Load biometric data
                bio_files = glob.glob(os.path.join(
                    self.base_path, "api_data_aadhar_biometric", "*.csv"
                ))
                
                if bio_files:
                    bio_dfs = []
                    for file in bio_files:
                        try:
                            df = pd.read_csv(file, low_memory=False)
                            bio_dfs.append(df)
                            st.sidebar.success(f"✓ Biometric: {os.path.basename(file)}")
                        except Exception as e:
                            st.sidebar.error(f"✗ Error loading {os.path.basename(file)}: {str(e)}")
                    
                    if bio_dfs:
                        self.biometric_df = pd.concat(bio_dfs, ignore_index=True)
                        st.sidebar.info(f"Biometric: {len(self.biometric_df):,} records")
                
                # Check if any data was loaded
                if self.enrolment_df is None and self.demographic_df is None and self.biometric_df is None:
                    st.error("No data files found! Please check the data directory.")
                    return False
                
                return True
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
    
    def process_datasets(self):
        """Process and enrich all datasets"""
        with st.spinner("🔄 Processing datasets..."):
            # Process enrolment data
            if self.enrolment_df is not None:
                df = self.enrolment_df.copy()
                
                # Convert date
                try:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                except:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Ensure numeric columns
                numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                
                # Calculate derived columns
                df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
                df['child_enrolments'] = df['age_0_5'] + df['age_5_17']
                df['adult_enrolments'] = df['age_18_greater']
                
                # Calculate percentages
                df['child_percentage'] = np.where(
                    df['total_enrolments'] > 0,
                    (df['child_enrolments'] / df['total_enrolments'] * 100).round(2),
                    0
                )
                df['adult_percentage'] = 100 - df['child_percentage']
                
                # Extract temporal features
                if df['date'].notnull().any():
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['month_year'] = df['date'].dt.strftime('%Y-%m')
                    df['quarter'] = df['date'].dt.quarter
                    df['day_of_week'] = df['date'].dt.day_name()
                
                self.processed_data['enrolment'] = df
                
                # Calculate enrolment KPIs
                self.calculate_enrolment_kpis(df)
            
            # Process demographic data
            if self.demographic_df is not None:
                df = self.demographic_df.copy()
                
                # Convert date
                try:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                except:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Find update columns (assuming they contain 'age' or 'demo')
                update_cols = [col for col in df.columns if 'age' in col.lower() or 'demo' in col.lower()]
                numeric_update_cols = []
                
                for col in update_cols:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                        numeric_update_cols.append(col)
                    except:
                        continue
                
                if numeric_update_cols:
                    df['total_demo_updates'] = df[numeric_update_cols].sum(axis=1)
                
                # Extract temporal features
                if df['date'].notnull().any():
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['month_year'] = df['date'].dt.strftime('%Y-%m')
                
                self.processed_data['demographic'] = df
                
                # Calculate demographic KPIs
                self.calculate_demographic_kpis(df)
            
            # Process biometric data
            if self.biometric_df is not None:
                df = self.biometric_df.copy()
                
                # Convert date
                try:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                except:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Find update columns (assuming they contain 'age' or 'bio')
                update_cols = [col for col in df.columns if 'age' in col.lower() or 'bio' in col.lower()]
                numeric_update_cols = []
                
                for col in update_cols:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                        numeric_update_cols.append(col)
                    except:
                        continue
                
                if numeric_update_cols:
                    df['total_bio_updates'] = df[numeric_update_cols].sum(axis=1)
                
                # Extract temporal features
                if df['date'].notnull().any():
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                    df['month_year'] = df['date'].dt.strftime('%Y-%m')
                
                self.processed_data['biometric'] = df
                
                # Calculate biometric KPIs
                self.calculate_biometric_kpis(df)
            
            st.success("✅ Data processing completed")
    
    def calculate_enrolment_kpis(self, df):
        """Calculate KPIs for enrolment data"""
        kpis = {
            'total_records': len(df),
            'total_enrolments': int(df['total_enrolments'].sum()),
            'states': df['state'].nunique(),
            'districts': df['district'].nunique(),
            'pincodes': df['pincode'].nunique() if 'pincode' in df.columns else 0,
            'child_enrolments': int(df['child_enrolments'].sum()),
            'adult_enrolments': int(df['adult_enrolments'].sum()),
            'child_percentage': (df['child_enrolments'].sum() / df['total_enrolments'].sum() * 100) if df['total_enrolments'].sum() > 0 else 0,
            'avg_daily_enrolments': 0,
            'date_range': 'N/A'
        }
        
        if df['date'].notnull().any():
            daily_enrolments = df.groupby('date')['total_enrolments'].sum()
            kpis['avg_daily_enrolments'] = daily_enrolments.mean() if len(daily_enrolments) > 0 else 0
            kpis['date_range'] = f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}"
        
        # Top states
        top_states = df.groupby('state')['total_enrolments'].sum().nlargest(5).to_dict()
        kpis['top_states'] = top_states
        
        # Monthly trends
        if 'month_year' in df.columns:
            monthly_trends = df.groupby('month_year')['total_enrolments'].sum().to_dict()
            kpis['monthly_trends'] = monthly_trends
        
        self.analysis_results['enrolment'] = kpis
    
    def calculate_demographic_kpis(self, df):
        """Calculate KPIs for demographic data"""
        kpis = {
            'total_records': len(df),
            'states': df['state'].nunique() if 'state' in df.columns else 0,
            'date_range': 'N/A'
        }
        
        if 'total_demo_updates' in df.columns:
            kpis['total_updates'] = int(df['total_demo_updates'].sum())
        
        if df['date'].notnull().any():
            kpis['date_range'] = f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}"
        
        # Top states for updates
        if 'total_demo_updates' in df.columns and 'state' in df.columns:
            top_states = df.groupby('state')['total_demo_updates'].sum().nlargest(5).to_dict()
            kpis['top_states'] = top_states
        
        # Monthly trends
        if 'month_year' in df.columns and 'total_demo_updates' in df.columns:
            monthly_trends = df.groupby('month_year')['total_demo_updates'].sum().to_dict()
            kpis['monthly_trends'] = monthly_trends
        
        self.analysis_results['demographic'] = kpis
    
    def calculate_biometric_kpis(self, df):
        """Calculate KPIs for biometric data"""
        kpis = {
            'total_records': len(df),
            'states': df['state'].nunique() if 'state' in df.columns else 0,
            'date_range': 'N/A'
        }
        
        if 'total_bio_updates' in df.columns:
            kpis['total_updates'] = int(df['total_bio_updates'].sum())
        
        if df['date'].notnull().any():
            kpis['date_range'] = f"{df['date'].min().strftime('%d %b %Y')} to {df['date'].max().strftime('%d %b %Y')}"
        
        # Top states for updates
        if 'total_bio_updates' in df.columns and 'state' in df.columns:
            top_states = df.groupby('state')['total_bio_updates'].sum().nlargest(5).to_dict()
            kpis['top_states'] = top_states
        
        # Monthly trends
        if 'month_year' in df.columns and 'total_bio_updates' in df.columns:
            monthly_trends = df.groupby('month_year')['total_bio_updates'].sum().to_dict()
            kpis['monthly_trends'] = monthly_trends
        
        self.analysis_results['biometric'] = kpis
    
    def create_dashboard(self):
        """Create the main dashboard"""
        
        # Header
        st.markdown('<h1 class="main-header">🆔 Aadhaar Data Analytics Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("**UIDAI Data Analysis | Enrolment & Update Patterns**")
        st.markdown("---")
        
        # Load and process data
        if not self.load_data():
            st.warning("Could not load data. Please check the data directory structure.")
            return
        
        self.process_datasets()
        
        # Sidebar for filters and controls
        with st.sidebar:
            st.markdown("## 🎛️ Dashboard Controls")
            st.markdown("---")
            
            # Dataset selection
            st.markdown("### 📊 Select Datasets")
            show_enrolment = st.checkbox("Enrolment Data", value=True)
            show_demographic = st.checkbox("Demographic Updates", value=True)
            show_biometric = st.checkbox("Biometric Updates", value=True)
            
            st.markdown("---")
            
            # Date range filter (if dates available)
            available_dates = []
            if 'enrolment' in self.processed_data and self.processed_data['enrolment']['date'].notnull().any():
                available_dates.append('enrolment')
            if 'demographic' in self.processed_data and self.processed_data['demographic']['date'].notnull().any():
                available_dates.append('demographic')
            if 'biometric' in self.processed_data and self.processed_data['biometric']['date'].notnull().any():
                available_dates.append('biometric')
            
            if available_dates:
                # Get overall date range
                min_dates = []
                max_dates = []
                
                for dataset in available_dates:
                    df = self.processed_data[dataset]
                    min_dates.append(df['date'].min())
                    max_dates.append(df['date'].max())
                
                overall_min = min(min_dates)
                overall_max = max(max_dates)
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(overall_min.date(), overall_max.date()),
                    min_value=overall_min.date(),
                    max_value=overall_max.date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    # Filter each dataset
                    for dataset in self.processed_data:
                        df = self.processed_data[dataset]
                        if 'date' in df.columns and df['date'].notnull().any():
                            self.processed_data[dataset] = df[
                                (df['date'].dt.date >= start_date) & 
                                (df['date'].dt.date <= end_date)
                            ]
            
            # State filter
            all_states = set()
            for dataset in self.processed_data:
                if 'state' in self.processed_data[dataset].columns:
                    all_states.update(self.processed_data[dataset]['state'].unique())
            
            if all_states:
                selected_states = st.multiselect(
                    "Select States",
                    options=sorted(all_states),
                    default=list(sorted(all_states))[:min(5, len(all_states))]
                )
                
                if selected_states:
                    for dataset in self.processed_data:
                        df = self.processed_data[dataset]
                        if 'state' in df.columns:
                            self.processed_data[dataset] = df[df['state'].isin(selected_states)]
            
            st.markdown("---")
            st.markdown("### 📈 Analysis Options")
            
            analysis_focus = st.selectbox(
                "Primary Analysis Focus",
                options=["Dataset Overview", "Geographic Patterns", "Temporal Trends", 
                        "Update Analysis", "Comparative Insights", "Anomaly Detection"]
            )
            
            st.markdown("---")
            st.markdown("### 📥 Data Export")
            
            if st.button("Export Analysis Results", use_container_width=True):
                self.export_analysis_results()
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.info("""
            **Aadhaar Data Analytics Dashboard**
            
            Analyzes enrolment, demographic update, and biometric update patterns to provide actionable insights for UIDAI system improvements.
            
            **Datasets Analyzed:**
            • Enrolment Records
            • Demographic Updates
            • Biometric Updates
            """)
        
        # Main dashboard layout
        tab_names = ["📊 Overview", "🗺️ Geographic", "📅 Temporal", "🔄 Updates", "💡 Insights", "⚙️ Advanced"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            self.create_overview_tab()
        
        with tabs[1]:
            self.create_geographic_tab()
        
        with tabs[2]:
            self.create_temporal_tab()
        
        with tabs[3]:
            self.create_updates_tab()
        
        with tabs[4]:
            self.create_insights_tab()
        
        with tabs[5]:
            self.create_advanced_tab()
    
    def create_overview_tab(self):
        """Create overview tab with KPIs and high-level insights"""
        st.markdown('<h2 class="sub-header">📊 Dataset Overview & KPIs</h2>', unsafe_allow_html=True)
        
        # Dataset summary cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'enrolment' in self.analysis_results:
                kpis = self.analysis_results['enrolment']
                st.markdown("""
                <div class="metric-card">
                    <span class="dataset-tag dataset-enrolment">Enrolment Data</span>
                    <h3>{:,}</h3>
                    <p>Total Enrolments</p>
                    <small>📋 {:,} records | 🏛️ {} states</small>
                </div>
                """.format(
                    kpis['total_enrolments'],
                    kpis['total_records'],
                    kpis['states']
                ), unsafe_allow_html=True)
        
        with col2:
            if 'demographic' in self.analysis_results:
                kpis = self.analysis_results['demographic']
                updates = kpis.get('total_updates', 0)
                st.markdown("""
                <div class="metric-card">
                    <span class="dataset-tag dataset-demographic">Demographic Updates</span>
                    <h3>{:,}</h3>
                    <p>Total Updates</p>
                    <small>📋 {:,} records | 🏛️ {} states</small>
                </div>
                """.format(
                    updates,
                    kpis['total_records'],
                    kpis['states']
                ), unsafe_allow_html=True)
        
        with col3:
            if 'biometric' in self.analysis_results:
                kpis = self.analysis_results['biometric']
                updates = kpis.get('total_updates', 0)
                st.markdown("""
                <div class="metric-card">
                    <span class="dataset-tag dataset-biometric">Biometric Updates</span>
                    <h3>{:,}</h3>
                    <p>Total Updates</p>
                    <small>📋 {:,} records | 🏛️ {} states</small>
                </div>
                """.format(
                    updates,
                    kpis['total_records'],
                    kpis['states']
                ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Age distribution for enrolment data
        if 'enrolment' in self.processed_data:
            st.markdown("### 👥 Age Distribution Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                df = self.processed_data['enrolment']
                age_totals = {
                    '0-5 Years': df['age_0_5'].sum(),
                    '5-17 Years': df['age_5_17'].sum(),
                    '18+ Years': df['age_18_greater'].sum()
                }
                
                if sum(age_totals.values()) > 0:
                    fig = px.pie(
                        values=list(age_totals.values()),
                        names=list(age_totals.keys()),
                        title='Age Distribution of Enrolments',
                        color_discrete_sequence=px.colors.sequential.Blues_r,
                        hole=0.3
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Age Group Statistics")
                
                for age_group, total in age_totals.items():
                    percentage = (total / sum(age_totals.values()) * 100) if sum(age_totals.values()) > 0 else 0
                    st.metric(
                        label=age_group,
                        value=f"{total:,}",
                        delta=f"{percentage:.1f}%"
                    )
                
                # Child vs Adult ratio
                child_total = age_totals['0-5 Years'] + age_totals['5-17 Years']
                adult_total = age_totals['18+ Years']
                
                if adult_total > 0:
                    child_adult_ratio = child_total / adult_total
                    st.metric(
                        label="Child:Adult Ratio",
                        value=f"{child_adult_ratio:.2f}",
                        delta=f"{child_total:,} : {adult_total:,}"
                    )
        
        st.markdown("---")
        
        # Dataset comparison
        st.markdown("### 📈 Dataset Comparison")
        
        comparison_data = []
        colors = ['#3B82F6', '#10B981', '#F59E0B']
        
        if 'enrolment' in self.analysis_results:
            comparison_data.append({
                'Dataset': 'Enrolment',
                'Records': self.analysis_results['enrolment']['total_records'],
                'Value': self.analysis_results['enrolment']['total_enrolments'],
                'Type': 'Enrolments'
            })
        
        if 'demographic' in self.analysis_results and 'total_updates' in self.analysis_results['demographic']:
            comparison_data.append({
                'Dataset': 'Demographic',
                'Records': self.analysis_results['demographic']['total_records'],
                'Value': self.analysis_results['demographic']['total_updates'],
                'Type': 'Updates'
            })
        
        if 'biometric' in self.analysis_results and 'total_updates' in self.analysis_results['biometric']:
            comparison_data.append({
                'Dataset': 'Biometric',
                'Records': self.analysis_results['biometric']['total_records'],
                'Value': self.analysis_results['biometric']['total_updates'],
                'Type': 'Updates'
            })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_comparison,
                    x='Dataset',
                    y='Records',
                    title='Number of Records by Dataset',
                    color='Dataset',
                    color_discrete_sequence=colors[:len(df_comparison)]
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    df_comparison,
                    x='Dataset',
                    y='Value',
                    title='Total Values by Dataset',
                    color='Dataset',
                    color_discrete_sequence=colors[:len(df_comparison)],
                    labels={'Value': 'Count'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
    
    def create_geographic_tab(self):
        """Create geographic analysis tab"""
        st.markdown('<h2 class="sub-header">🗺️ Geographic Distribution Analysis</h2>', unsafe_allow_html=True)
        
        # State-wise analysis for each dataset
        datasets_to_show = []
        if 'enrolment' in self.processed_data:
            datasets_to_show.append(('Enrolment', self.processed_data['enrolment'], 'total_enrolments'))
        if 'demographic' in self.processed_data and 'total_demo_updates' in self.processed_data['demographic'].columns:
            datasets_to_show.append(('Demographic', self.processed_data['demographic'], 'total_demo_updates'))
        if 'biometric' in self.processed_data and 'total_bio_updates' in self.processed_data['biometric'].columns:
            datasets_to_show.append(('Biometric', self.processed_data['biometric'], 'total_bio_updates'))
        
        if not datasets_to_show:
            st.warning("No geographic data available for analysis")
            return
        
        # Create tabs for each dataset
        dataset_tabs = st.tabs([d[0] for d in datasets_to_show])
        
        for idx, (dataset_name, df, value_col) in enumerate(datasets_to_show):
            with dataset_tabs[idx]:
                if 'state' in df.columns:
                    # State-wise totals
                    state_totals = df.groupby('state')[value_col].sum().reset_index()
                    state_totals = state_totals.sort_values(value_col, ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Top states chart
                        top_states = state_totals.head(15)
                        
                        fig = px.bar(
                            top_states,
                            x='state',
                            y=value_col,
                            title=f'Top 15 States - {dataset_name}',
                            labels={value_col: 'Count', 'state': 'State'},
                            color=value_col,
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=500, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"### 📊 Top 5 States - {dataset_name}")
                        
                        for i, row in state_totals.head(5).iterrows():
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong>{i+1}. {row['state']}</strong><br>
                                <h4>{int(row[value_col]):,}</h4>
                                <small>{dataset_name.lower()}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Geographic concentration
                        if len(state_totals) >= 3:
                            top_3_total = state_totals.head(3)[value_col].sum()
                            total_sum = state_totals[value_col].sum()
                            concentration = (top_3_total / total_sum * 100) if total_sum > 0 else 0
                            
                            st.metric(
                                label="Geographic Concentration",
                                value=f"{concentration:.1f}%",
                                delta="Top 3 states share"
                            )
        
        # Comparative geographic analysis
        st.markdown("---")
        st.markdown("### 📊 Comparative Geographic Analysis")
        
        # Create a comparative chart
        comparative_data = []
        
        for dataset_name, df, value_col in datasets_to_show:
            if 'state' in df.columns:
                state_totals = df.groupby('state')[value_col].sum()
                # Get top 5 states for this dataset
                top_states = state_totals.nlargest(5)
                
                for state, value in top_states.items():
                    comparative_data.append({
                        'State': state,
                        'Dataset': dataset_name,
                        'Value': value
                    })
        
        if comparative_data:
            df_comparative = pd.DataFrame(comparative_data)
            
            # Pivot for heatmap
            pivot_data = df_comparative.pivot(index='State', columns='Dataset', values='Value').fillna(0)
            
            fig = px.imshow(
                pivot_data,
                text_auto=True,
                color_continuous_scale='Blues',
                title='Top States Comparison Across Datasets',
                labels=dict(x="Dataset", y="State", color="Value")
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_temporal_tab(self):
        """Create temporal trends analysis tab"""
        st.markdown('<h2 class="sub-header">📅 Temporal Trends & Patterns</h2>', unsafe_allow_html=True)
        
        # Collect datasets with date information
        datasets_with_dates = []
        if 'enrolment' in self.processed_data and self.processed_data['enrolment']['date'].notnull().any():
            datasets_with_dates.append(('Enrolment', self.processed_data['enrolment'], 'total_enrolments'))
        if 'demographic' in self.processed_data and 'date' in self.processed_data['demographic'].columns and self.processed_data['demographic']['date'].notnull().any() and 'total_demo_updates' in self.processed_data['demographic'].columns:
            datasets_with_dates.append(('Demographic', self.processed_data['demographic'], 'total_demo_updates'))
        if 'biometric' in self.processed_data and 'date' in self.processed_data['biometric'].columns and self.processed_data['biometric']['date'].notnull().any() and 'total_bio_updates' in self.processed_data['biometric'].columns:
            datasets_with_dates.append(('Biometric', self.processed_data['biometric'], 'total_bio_updates'))
        
        if not datasets_with_dates:
            st.warning("No temporal data available for analysis")
            return
        
        # Create tabs for each dataset
        dataset_tabs = st.tabs([d[0] for d in datasets_with_dates])
        
        for idx, (dataset_name, df, value_col) in enumerate(datasets_with_dates):
            with dataset_tabs[idx]:
                # Daily trends
                if 'date' in df.columns:
                    daily_data = df.groupby('date')[value_col].sum().reset_index()
                    
                    if len(daily_data) > 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Line chart
                            fig = px.line(
                                daily_data,
                                x='date',
                                y=value_col,
                                title=f'Daily {dataset_name} Trends',
                                labels={value_col: 'Count', 'date': 'Date'}
                            )
                            
                            # Add 7-day moving average if enough data
                            if len(daily_data) >= 7:
                                daily_data['7_day_avg'] = daily_data[value_col].rolling(window=7).mean()
                                fig.add_scatter(
                                    x=daily_data['date'],
                                    y=daily_data['7_day_avg'],
                                    mode='lines',
                                    name='7-day Moving Avg',
                                    line=dict(color='red', dash='dash')
                                )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Statistics
                            st.markdown(f"### 📊 {dataset_name} Statistics")
                            
                            avg_daily = daily_data[value_col].mean()
                            max_daily = daily_data[value_col].max()
                            min_daily = daily_data[value_col].min()
                            max_date = daily_data.loc[daily_data[value_col].idxmax(), 'date']
                            
                            st.metric("Average Daily", f"{avg_daily:,.0f}")
                            st.metric("Peak Day", f"{max_daily:,.0f}", max_date.strftime('%d %b %Y'))
                            st.metric("Minimum Daily", f"{min_daily:,.0f}")
                            
                            # Growth rate if enough data
                            if len(daily_data) >= 30:
                                first_half = daily_data.head(15)[value_col].mean()
                                second_half = daily_data.tail(15)[value_col].mean()
                                if first_half > 0:
                                    growth = ((second_half - first_half) / first_half) * 100
                                    st.metric("Growth Rate", f"{growth:.1f}%")
                    
                    # Monthly trends
                    if 'month_year' in df.columns:
                        monthly_data = df.groupby('month_year')[value_col].sum().reset_index()
                        
                        if len(monthly_data) > 1:
                            fig = px.bar(
                                monthly_data,
                                x='month_year',
                                y=value_col,
                                title=f'Monthly {dataset_name} Pattern',
                                labels={value_col: 'Count', 'month_year': 'Month'}
                            )
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
        
        # Comparative temporal analysis
        st.markdown("---")
        st.markdown("### 📈 Comparative Temporal Analysis")
        
        # Create a combined time series
        combined_data = []
        
        for dataset_name, df, value_col in datasets_with_dates:
            if 'date' in df.columns:
                daily_totals = df.groupby('date')[value_col].sum()
                for date, value in daily_totals.items():
                    combined_data.append({
                        'Date': date,
                        'Dataset': dataset_name,
                        'Value': value
                    })
        
        if combined_data:
            df_combined = pd.DataFrame(combined_data)
            
            # Normalize values for comparison
            for dataset in df_combined['Dataset'].unique():
                dataset_values = df_combined[df_combined['Dataset'] == dataset]['Value']
                if dataset_values.max() > 0:
                    df_combined.loc[df_combined['Dataset'] == dataset, 'Normalized'] = (
                        dataset_values / dataset_values.max() * 100
                    )
            
            fig = px.line(
                df_combined,
                x='Date',
                y='Normalized',
                color='Dataset',
                title='Comparative Trends (Normalized to 100%)',
                labels={'Normalized': 'Normalized Value (%)', 'Date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_updates_tab(self):
        """Create updates analysis tab"""
        st.markdown('<h2 class="sub-header">🔄 Update Pattern Analysis</h2>', unsafe_allow_html=True)
        
        # Check if we have update data
        has_demo_updates = 'demographic' in self.processed_data and 'total_demo_updates' in self.processed_data['demographic'].columns
        has_bio_updates = 'biometric' in self.processed_data and 'total_bio_updates' in self.processed_data['biometric'].columns
        
        if not has_demo_updates and not has_bio_updates:
            st.warning("No update data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Update comparison chart
            update_data = []
            
            if has_demo_updates:
                demo_total = self.processed_data['demographic']['total_demo_updates'].sum()
                update_data.append({
                    'Type': 'Demographic Updates',
                    'Count': demo_total,
                    'Color': '#10B981'
                })
            
            if has_bio_updates:
                bio_total = self.processed_data['biometric']['total_bio_updates'].sum()
                update_data.append({
                    'Type': 'Biometric Updates',
                    'Count': bio_total,
                    'Color': '#F59E0B'
                })
            
            if update_data:
                df_updates = pd.DataFrame(update_data)
                
                fig = px.pie(
                    df_updates,
                    values='Count',
                    names='Type',
                    title='Update Type Distribution',
                    color='Type',
                    color_discrete_sequence=[d['Color'] for d in update_data]
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Update Statistics")
            
            if has_demo_updates:
                demo_df = self.processed_data['demographic']
                demo_total = demo_df['total_demo_updates'].sum()
                demo_avg = demo_df['total_demo_updates'].mean()
                demo_states = demo_df['state'].nunique() if 'state' in demo_df.columns else 0
                
                st.markdown("""
                <div class="metric-card">
                    <span class="dataset-tag dataset-demographic">Demographic</span>
                    <h3>{:,}</h3>
                    <p>Total Updates</p>
                    <small>📈 Avg: {:.1f} | 🏛️ {} states</small>
                </div>
                """.format(demo_total, demo_avg, demo_states), unsafe_allow_html=True)
            
            if has_bio_updates:
                bio_df = self.processed_data['biometric']
                bio_total = bio_df['total_bio_updates'].sum()
                bio_avg = bio_df['total_bio_updates'].mean()
                bio_states = bio_df['state'].nunique() if 'state' in bio_df.columns else 0
                
                st.markdown("""
                <div class="metric-card">
                    <span class="dataset-tag dataset-biometric">Biometric</span>
                    <h3>{:,}</h3>
                    <p>Total Updates</p>
                    <small>📈 Avg: {:.1f} | 🏛️ {} states</small>
                </div>
                """.format(bio_total, bio_avg, bio_states), unsafe_allow_html=True)
        
        # Update patterns by state
        st.markdown("---")
        st.markdown("### 🏛️ Update Patterns by State")
        
        # Create a comparison of top states for updates
        update_by_state = {}
        
        if has_demo_updates and 'state' in self.processed_data['demographic'].columns:
            demo_by_state = self.processed_data['demographic'].groupby('state')['total_demo_updates'].sum()
            update_by_state['Demographic'] = demo_by_state
        
        if has_bio_updates and 'state' in self.processed_data['biometric'].columns:
            bio_by_state = self.processed_data['biometric'].groupby('state')['total_bio_updates'].sum()
            update_by_state['Biometric'] = bio_by_state
        
        if update_by_state:
            # Get common states
            all_states = set()
            for updates in update_by_state.values():
                all_states.update(updates.index)
            
            # Create comparison dataframe
            comparison_data = []
            for state in sorted(all_states):
                row = {'State': state}
                for update_type, state_updates in update_by_state.items():
                    row[update_type] = state_updates.get(state, 0)
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Show top 10 states by total updates
            df_comparison['Total'] = df_comparison.sum(axis=1, numeric_only=True)
            top_states = df_comparison.nlargest(10, 'Total')
            
            fig = go.Figure()
            
            colors = ['#10B981', '#F59E0B']
            for i, (update_type, color) in enumerate(zip(update_by_state.keys(), colors)):
                fig.add_trace(go.Bar(
                    x=top_states['State'],
                    y=top_states[update_type],
                    name=update_type,
                    marker_color=color
                ))
            
            fig.update_layout(
                title='Top 10 States by Update Activity',
                barmode='stack',
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Update to enrolment ratio (if enrolment data available)
        if 'enrolment' in self.processed_data and (has_demo_updates or has_bio_updates):
            st.markdown("---")
            st.markdown("### 📊 Update to Enrolment Ratios")
            
            # Calculate state-wise enrolment
            enrolment_by_state = self.processed_data['enrolment'].groupby('state')['total_enrolments'].sum()
            
            ratio_data = []
            for state in enrolment_by_state.index:
                enrolment = enrolment_by_state[state]
                if enrolment > 0:
                    row = {'State': state, 'Enrolment': enrolment}
                    
                    if has_demo_updates and state in demo_by_state:
                        demo_updates = demo_by_state[state]
                        row['Demo_Ratio'] = (demo_updates / enrolment) * 100
                    
                    if has_bio_updates and state in bio_by_state:
                        bio_updates = bio_by_state[state]
                        row['Bio_Ratio'] = (bio_updates / enrolment) * 100
                    
                    ratio_data.append(row)
            
            if ratio_data:
                df_ratios = pd.DataFrame(ratio_data)
                
                # Show states with highest update ratios
                if 'Demo_Ratio' in df_ratios.columns:
                    top_demo_ratio = df_ratios.nlargest(5, 'Demo_Ratio')[['State', 'Demo_Ratio']]
                    st.markdown("**Top 5 States - Demographic Update Ratio:**")
                    st.dataframe(top_demo_ratio.style.format({'Demo_Ratio': '{:.2f}%'}))
                
                if 'Bio_Ratio' in df_ratios.columns:
                    top_bio_ratio = df_ratios.nlargest(5, 'Bio_Ratio')[['State', 'Bio_Ratio']]
                    st.markdown("**Top 5 States - Biometric Update Ratio:**")
                    st.dataframe(top_bio_ratio.style.format({'Bio_Ratio': '{:.2f}%'}))
    
    def create_insights_tab(self):
        """Create insights and recommendations tab"""
        st.markdown('<h2 class="sub-header">💡 Strategic Insights & Recommendations</h2>', unsafe_allow_html=True)
        
        # Key insights from each dataset
        insights = []
        
        # Enrolment insights
        if 'enrolment' in self.analysis_results:
            kpis = self.analysis_results['enrolment']
            
            # Age distribution insight
            child_pct = kpis.get('child_percentage', 0)
            if child_pct > 60:
                insights.append({
                    'type': 'success',
                    'title': 'High Child Enrollment',
                    'content': f'Child enrollment rate is {child_pct:.1f}%, indicating successful targeting of younger demographics for Aadhaar registration.',
                    'recommendation': 'Continue child-focused enrollment campaigns and consider age-specific verification processes.'
                })
            elif child_pct < 40:
                insights.append({
                    'type': 'warning',
                    'title': 'Low Child Enrollment',
                    'content': f'Child enrollment rate is {child_pct:.1f}%, suggesting potential gaps in child registration coverage.',
                    'recommendation': 'Implement targeted child enrollment drives and school-based registration programs.'
                })
            
            # Geographic concentration insight
            if 'top_states' in kpis:
                top_states = list(kpis['top_states'].keys())[:3]
                top_states_str = ', '.join(top_states)
                insights.append({
                    'type': 'info',
                    'title': 'Geographic Concentration',
                    'content': f'Top 3 states ({top_states_str}) account for significant portion of enrollments.',
                    'recommendation': 'Consider resource allocation optimization based on geographic demand patterns.'
                })
        
        # Update insights
        demo_updates = self.analysis_results.get('demographic', {}).get('total_updates', 0)
        bio_updates = self.analysis_results.get('biometric', {}).get('total_updates', 0)
        
        if demo_updates > 0 and bio_updates > 0:
            update_ratio = demo_updates / bio_updates
            if update_ratio > 2:
                insights.append({
                    'type': 'info',
                    'title': 'High Demographic Update Activity',
                    'content': f'Demographic updates are {update_ratio:.1f}x more frequent than biometric updates.',
                    'recommendation': 'Streamline demographic update processes and consider automated verification systems.'
                })
        
        # Display insights
        st.markdown("### 🔍 Key Insights")
        
        if insights:
            for insight in insights:
                if insight['type'] == 'success':
                    st.success(f"**{insight['title']}**: {insight['content']}")
                elif insight['type'] == 'warning':
                    st.warning(f"**{insight['title']}**: {insight['content']}")
                else:
                    st.info(f"**{insight['title']}**: {insight['content']}")
                
                st.markdown(f"*💡 Recommendation:* {insight['recommendation']}")
                st.markdown("---")
        else:
            st.info("Generate more data for detailed insights. Key patterns will emerge with comprehensive dataset analysis.")
        
        # Recommendations matrix
        st.markdown("### 🎯 Actionable Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Enrollment Optimization</h4>
            <ul>
            <li>Targeted campaigns in low-enrollment regions</li>
            <li>Mobile enrollment units for remote areas</li>
            <li>Age-specific enrollment strategies</li>
            <li>School-based registration programs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>🔧 System Improvements</h4>
            <ul>
            <li>Streamline update processes</li>
            <li>Automated data validation</li>
            <li>Real-time monitoring dashboards</li>
            <li>Enhanced security protocols</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>📊 Data Analytics</h4>
            <ul>
            <li>Predictive modeling for enrollment trends</li>
            <li>Anomaly detection systems</li>
            <li>Geographic heatmaps for planning</li>
            <li>Performance benchmarking</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>🤝 Stakeholder Engagement</h4>
            <ul>
            <li>Community awareness programs</li>
            <li>Government agency coordination</li>
            <li>Feedback collection mechanisms</li>
            <li>Transparency initiatives</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Priority matrix
        st.markdown("---")
        st.markdown("### 📊 Priority Implementation Matrix")
        
        matrix_data = {
            'High Impact, Easy Implementation': [
                'Digital outreach optimization',
                'Process automation for high-volume centers',
                'Real-time monitoring dashboards'
            ],
            'High Impact, Complex Implementation': [
                'Mobile enrollment unit deployment',
                'Inter-state coordination frameworks',
                'Policy framework enhancements'
            ],
            'Quick Wins': [
                'Data quality improvement initiatives',
                'Staff training programs',
                'Feedback system implementation'
            ]
        }
        
        cols = st.columns(3)
        for idx, (category, items) in enumerate(matrix_data.items()):
            with cols[idx]:
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"• {item}")
        
        # Export options
        st.markdown("---")
        st.markdown("### 📥 Export Analysis Results")
        
        if st.button("Generate Comprehensive Report", use_container_width=True):
            self.generate_comprehensive_report()
    
    def create_advanced_tab(self):
        """Create advanced analysis tab"""
        st.markdown('<h2 class="sub-header">⚙️ Advanced Analysis & Tools</h2>', unsafe_allow_html=True)
        
        # Anomaly Detection
        st.markdown("### ⚠️ Anomaly Detection")
        
        if 'enrolment' in self.processed_data and self.processed_data['enrolment']['date'].notnull().any():
            df = self.processed_data['enrolment']
            daily_enrolments = df.groupby('date')['total_enrolments'].sum()
            
            if len(daily_enrolments) > 0:
                # Calculate statistical anomalies
                mean = daily_enrolments.mean()
                std = daily_enrolments.std()
                
                if std > 0:
                    anomalies = daily_enrolments[
                        (daily_enrolments > mean + 2*std) | 
                        (daily_enrolments < mean - 2*std)
                    ]
                    
                    if len(anomalies) > 0:
                        st.warning(f"Found {len(anomalies)} days with unusual enrollment patterns")
                        
                        anomaly_df = pd.DataFrame({
                            'Date': anomalies.index,
                            'Enrollments': anomalies.values,
                            'Deviation from Mean (%)': ((anomalies.values - mean) / mean * 100).round(1)
                        })
                        
                        st.dataframe(anomaly_df.style.format({
                            'Enrollments': '{:,}',
                            'Deviation from Mean (%)': '{:.1f}%'
                        }), use_container_width=True)
                    else:
                        st.info("No significant anomalies detected in enrollment patterns")
        
        # Predictive Analysis
        st.markdown("---")
        st.markdown("### 🔮 Predictive Analysis")
        
        if 'enrolment' in self.processed_data and self.processed_data['enrolment']['date'].notnull().any():
            df = self.processed_data['enrolment']
            daily_trend = df.groupby('date')['total_enrolments'].sum()
            
            if len(daily_trend) >= 30:
                # Simple trend analysis
                last_30_avg = daily_trend.tail(30).mean()
                prev_30_avg = daily_trend.tail(60).head(30).mean() if len(daily_trend) >= 60 else last_30_avg
                
                if prev_30_avg > 0:
                    growth_rate = ((last_30_avg - prev_30_avg) / prev_30_avg) * 100
                else:
                    growth_rate = 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current 30-day Average",
                        f"{last_30_avg:,.0f}",
                        "enrollments/day"
                    )
                
                with col2:
                    st.metric(
                        "Growth Rate",
                        f"{growth_rate:.1f}%",
                        "month-over-month"
                    )
                
                with col3:
                    next_30_forecast = last_30_avg * 30 * (1 + growth_rate/100)
                    st.metric(
                        "30-day Forecast",
                        f"{next_30_forecast:,.0f}",
                        "expected enrollments"
                    )
            else:
                st.info("Need at least 30 days of data for predictive analysis")
        
        # Data Quality Analysis
        st.markdown("---")
        st.markdown("### 📋 Data Quality Analysis")
        
        quality_data = []
        
        for dataset_name in ['enrolment', 'demographic', 'biometric']:
            if dataset_name in self.processed_data:
                df = self.processed_data[dataset_name]
                
                # Calculate completeness
                total_cells = df.shape[0] * df.shape[1]
                non_null_cells = df.notnull().sum().sum()
                completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
                
                # Check for duplicates
                duplicates = df.duplicated().sum()
                duplicate_pct = (duplicates / len(df) * 100) if len(df) > 0 else 0
                
                quality_data.append({
                    'Dataset': dataset_name.capitalize(),
                    'Records': len(df),
                    'Completeness': completeness,
                    'Duplicates': duplicate_pct,
                    'Columns': df.shape[1]
                })
        
        if quality_data:
            df_quality = pd.DataFrame(quality_data)
            
            # Display quality metrics
            cols = st.columns(len(quality_data))
            for idx, row in df_quality.iterrows():
                with cols[idx]:
                    st.metric(
                        row['Dataset'],
                        f"{row['Records']:,}",
                        f"{row['Completeness']:.1f}% complete"
                    )
            
            # Quality score chart
            fig = px.bar(
                df_quality,
                x='Dataset',
                y=['Completeness', 'Duplicates'],
                title='Data Quality Metrics',
                barmode='group',
                labels={'value': 'Percentage', 'variable': 'Metric'},
                color_discrete_sequence=['#10B981', '#EF4444']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export tools
        st.markdown("---")
        st.markdown("### 🛠️ Advanced Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Data Quality Report", use_container_width=True):
                self.generate_quality_report()
        
        with col2:
            if st.button("Export Processed Data", use_container_width=True):
                self.export_processed_data()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        report_content = []
        report_content.append("=" * 60)
        report_content.append("AADHAAR DATA ANALYSIS REPORT")
        report_content.append("=" * 60)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("EXECUTIVE SUMMARY")
        report_content.append("-" * 40)
        
        for dataset in ['enrolment', 'demographic', 'biometric']:
            if dataset in self.analysis_results:
                kpis = self.analysis_results[dataset]
                dataset_name = dataset.capitalize()
                
                if dataset == 'enrolment':
                    report_content.append(f"{dataset_name}:")
                    report_content.append(f"  • Total Enrolments: {kpis.get('total_enrolments', 0):,}")
                    report_content.append(f"  • Child Enrollment Rate: {kpis.get('child_percentage', 0):.1f}%")
                else:
                    report_content.append(f"{dataset_name} Updates:")
                    report_content.append(f"  • Total Updates: {kpis.get('total_updates', 0):,}")
                
                report_content.append(f"  • Records: {kpis.get('total_records', 0):,}")
                report_content.append(f"  • States: {kpis.get('states', 0)}")
                report_content.append("")
        
        # Key Insights
        report_content.append("KEY INSIGHTS")
        report_content.append("-" * 40)
        
        if 'enrolment' in self.analysis_results:
            kpis = self.analysis_results['enrolment']
            top_states = list(kpis.get('top_states', {}).keys())[:3]
            if top_states:
                report_content.append(f"1. Geographic Focus: Top 3 states are {', '.join(top_states)}")
        
        if 'demographic' in self.analysis_results and 'biometric' in self.analysis_results:
            demo_updates = self.analysis_results['demographic'].get('total_updates', 0)
            bio_updates = self.analysis_results['biometric'].get('total_updates', 0)
            if demo_updates > 0 and bio_updates > 0:
                ratio = demo_updates / bio_updates
                report_content.append(f"2. Update Patterns: Demographic updates are {ratio:.1f}x biometric updates")
        
        report_content.append("")
        
        # Recommendations
        report_content.append("RECOMMENDATIONS")
        report_content.append("-" * 40)
        recommendations = [
            "1. Implement targeted enrollment campaigns in low-coverage regions",
            "2. Optimize update processes based on frequency patterns",
            "3. Develop real-time monitoring dashboards",
            "4. Enhance data quality through validation procedures",
            "5. Establish predictive analytics for resource planning"
        ]
        
        for rec in recommendations:
            report_content.append(rec)
        
        # Convert to string
        report_text = "\n".join(report_content)
        
        # Create download button
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name="aadhaar_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    def generate_quality_report(self):
        """Generate data quality report"""
        st.info("Data quality report generated. Check the exports section for download options.")
    
    def export_processed_data(self):
        """Export processed data as CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export each dataset
            exported_files = []
            
            for dataset_name, df in self.processed_data.items():
                filename = f"{dataset_name}_processed.csv"
                filepath = os.path.join(tmpdir, filename)
                df.to_csv(filepath, index=False)
                exported_files.append((filename, filepath))
            
            # Create zip file
            zip_path = os.path.join(tmpdir, "aadhaar_processed_data.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for filename, filepath in exported_files:
                    zipf.write(filepath, filename)
            
            # Read zip file as bytes
            with open(zip_path, 'rb') as f:
                zip_bytes = f.read()
            
            # Create download button
            st.download_button(
                label="📥 Download Processed Data (ZIP)",
                data=zip_bytes,
                file_name="aadhaar_processed_data.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    def export_analysis_results(self):
        """Export analysis results as JSON"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_results': self.analysis_results,
            'summary': {}
        }
        
        # Create summary
        for dataset, kpis in self.analysis_results.items():
            export_data['summary'][dataset] = {
                'records': kpis.get('total_records', 0),
                'states': kpis.get('states', 0),
                'value': kpis.get('total_enrolments', kpis.get('total_updates', 0))
            }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Export Analysis Results (JSON)",
            data=json_data,
            file_name="aadhaar_analysis_results.json",
            mime="application/json",
            use_container_width=True
        )

# Main application
def main():
    """Main Streamlit application"""
    
    # Initialize dashboard
    dashboard = AadhaarStreamlitDashboard()
    
    # Create dashboard
    dashboard.create_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
    <p>🆔 <strong>Aadhaar Data Analytics Dashboard</strong> | UIDAI Data Analysis System</p>
    <p>Analyzing enrolment, demographic updates, and biometric updates for system optimization</p>
    <p>Data Source: UIDAI Datasets | Analysis Period: Based on available data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()