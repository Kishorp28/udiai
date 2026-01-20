"""
AADHAAR SOCIETAL TRENDS ANALYZER
Innovative Solution for Uncovering Patterns in Aadhaar Enrolment and Updates

Problem Statement: Unlocking Societal Trends in Aadhaar Enrolment and Updates
Identify meaningful patterns, trends, anomalies, or predictive indicators and translate them into 
clear insights or solution frameworks that can support informed decision-making and system improvements.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from prophet import Prophet
from sklearn.metrics import silhouette_score

# Statistical libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

class AadhaarSocietalTrendsAnalyzer:
    """
    Comprehensive analyzer for identifying societal trends, patterns, anomalies,
    and predictive indicators from Aadhaar enrolment and update data.
    """
    
    def __init__(self, base_path="E:/UDAI/hack"):
        self.base_path = base_path
        self.enrolment_df = None
        self.demographic_df = None
        self.biometric_df = None
        self.merged_df = None
        
        # Setup visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_and_preprocess_data(self):
        """
        Load all datasets with robust preprocessing
        """
        print("="*80)
        print("DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        # Load enrolment data
        print("\n1. Loading Enrolment Data...")
        enrolment_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_enrolment", "*.csv"))
        enrolment_dfs = []
        
        for file in enrolment_files:
            df = pd.read_csv(file)
            enrolment_dfs.append(df)
        
        self.enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
        
        # Parse dates with Indian format (DD-MM-YYYY)
        self.enrolment_df['date'] = pd.to_datetime(self.enrolment_df['date'], dayfirst=True, errors='coerce')
        
        # Create additional features
        self.enrolment_df['year'] = self.enrolment_df['date'].dt.year
        self.enrolment_df['month'] = self.enrolment_df['date'].dt.month
        self.enrolment_df['quarter'] = self.enrolment_df['date'].dt.quarter
        self.enrolment_df['day_of_week'] = self.enrolment_df['date'].dt.dayofweek
        
        # Calculate derived metrics
        self.enrolment_df['total_enrolments'] = (
            self.enrolment_df['age_0_5'] + 
            self.enrolment_df['age_5_17'] + 
            self.enrolment_df['age_18_greater']
        )
        
        # Age distribution percentages
        self.enrolment_df['child_percentage'] = (
            (self.enrolment_df['age_0_5'] + self.enrolment_df['age_5_17']) / 
            self.enrolment_df['total_enrolments'].replace(0, np.nan)
        )
        
        print(f"   Loaded {len(self.enrolment_df):,} enrolment records")
        print(f"   Date range: {self.enrolment_df['date'].min()} to {self.enrolment_df['date'].max()}")
        
        # Load demographic update data
        print("\n2. Loading Demographic Update Data...")
        demo_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_demographic", "*.csv"))
        demo_dfs = []
        
        for file in demo_files:
            df = pd.read_csv(file)
            demo_dfs.append(df)
        
        if demo_dfs:
            self.demographic_df = pd.concat(demo_dfs, ignore_index=True)
            self.demographic_df['date'] = pd.to_datetime(self.demographic_df['date'], dayfirst=True, errors='coerce')
            self.demographic_df['total_demo_updates'] = self.demographic_df.iloc[:, 4:].sum(axis=1)
            print(f"   Loaded {len(self.demographic_df):,} demographic update records")
        
        # Load biometric update data
        print("\n3. Loading Biometric Update Data...")
        bio_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_biometric", "*.csv"))
        bio_dfs = []
        
        for file in bio_files:
            df = pd.read_csv(file)
            bio_dfs.append(df)
        
        if bio_dfs:
            self.biometric_df = pd.concat(bio_dfs, ignore_index=True)
            self.biometric_df['date'] = pd.to_datetime(self.biometric_df['date'], dayfirst=True, errors='coerce')
            self.biometric_df['total_bio_updates'] = self.biometric_df.iloc[:, 4:].sum(axis=1)
            print(f"   Loaded {len(self.biometric_df):,} biometric update records")
        
        return self
    
    def perform_eda(self):
        """
        Comprehensive Exploratory Data Analysis
        """
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        eda_results = {}
        
        # 1. Temporal Analysis
        print("\n1. Temporal Analysis of Enrolments:")
        monthly_enrolment = self.enrolment_df.groupby(
            pd.Grouper(key='date', freq='M')
        )['total_enrolments'].sum().reset_index()
        
        fig = plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(monthly_enrolment['date'], monthly_enrolment['total_enrolments'], 
                marker='o', linewidth=2)
        plt.title('Monthly Enrolment Trends', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Total Enrolments')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Seasonal decomposition
        plt.subplot(2, 2, 2)
        ts_data = monthly_enrolment.set_index('date')['total_enrolments']
        decomposition = seasonal_decompose(ts_data, model='additive', period=12)
        plt.plot(decomposition.trend, label='Trend', linewidth=2)
        plt.plot(decomposition.seasonal, label='Seasonal', alpha=0.7)
        plt.title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Geographic Analysis
        print("\n2. Geographic Distribution Analysis:")
        state_analysis = self.enrolment_df.groupby('state').agg({
            'total_enrolments': 'sum',
            'child_percentage': 'mean'
        }).sort_values('total_enrolments', ascending=False)
        
        plt.subplot(2, 2, 3)
        top_states = state_analysis.head(10)
        plt.barh(range(len(top_states)), top_states['total_enrolments'])
        plt.yticks(range(len(top_states)), top_states.index)
        plt.title('Top 10 States by Enrolments', fontsize=14, fontweight='bold')
        plt.xlabel('Total Enrolments')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 3. Age Distribution Analysis
        print("\n3. Age Distribution Analysis:")
        age_totals = {
            '0-5 years': self.enrolment_df['age_0_5'].sum(),
            '5-17 years': self.enrolment_df['age_5_17'].sum(),
            '18+ years': self.enrolment_df['age_18_greater'].sum()
        }
        
        plt.subplot(2, 2, 4)
        plt.pie(age_totals.values(), labels=age_totals.keys(), autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0.05, 0.05))
        plt.title('Age Distribution of Enrolments', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        eda_results['monthly_trends'] = monthly_enrolment
        eda_results['state_analysis'] = state_analysis
        eda_results['age_distribution'] = age_totals
        
        return eda_results
    
    def detect_societal_patterns(self):
        """
        Detect meaningful societal patterns and trends
        """
        print("\n" + "="*80)
        print("SOCIETAL PATTERN DETECTION")
        print("="*80)
        
        patterns = {}
        
        # 1. Urbanization Patterns
        print("\n1. Urbanization Pattern Analysis:")
        # Using pincode as proxy for urbanization (assuming 6-digit pincodes)
        self.enrolment_df['pincode_length'] = self.enrolment_df['pincode'].astype(str).str.len()
        
        # Analyze enrolment density by pincode
        pincode_analysis = self.enrolment_df.groupby('pincode').agg({
            'total_enrolments': 'sum',
            'child_percentage': 'mean'
        }).reset_index()
        
        # Classify as urban (high density) vs rural (low density)
        pincode_analysis['enrolment_density'] = pd.qcut(
            pincode_analysis['total_enrolments'], 
            q=4, 
            labels=['Very Low', 'Low', 'High', 'Very High']
        )
        
        patterns['urbanization_patterns'] = pincode_analysis
        
        # 2. Digital Literacy Indicators
        print("\n2. Digital Literacy Indicators:")
        # Assuming update frequency indicates digital awareness
        if self.demographic_df is not None:
            state_digital_literacy = self.demographic_df.groupby('state').agg({
                'total_demo_updates': 'sum'
            }).reset_index()
            
            # Merge with enrolment data to calculate update rates
            enrolment_by_state = self.enrolment_df.groupby('state')['total_enrolments'].sum().reset_index()
            digital_metrics = pd.merge(state_digital_literacy, enrolment_by_state, on='state')
            digital_metrics['update_rate'] = (
                digital_metrics['total_demo_updates'] / 
                digital_metrics['total_enrolments'].replace(0, np.nan)
            )
            
            patterns['digital_literacy_indicators'] = digital_metrics.sort_values('update_rate', ascending=False)
        
        # 3. Migration Patterns
        print("\n3. Migration Pattern Detection:")
        # Analyze temporal patterns that might indicate migration
        monthly_state_enrolment = self.enrolment_df.groupby(['state', pd.Grouper(key='date', freq='M')])[
            'total_enrolments'
        ].sum().reset_index()
        
        # Calculate month-over-month growth
        monthly_state_enrolment['monthly_growth'] = monthly_state_enrolment.groupby('state')[
            'total_enrolments'
        ].pct_change()
        
        # Detect sudden spikes indicating possible migration
        migration_indicators = monthly_state_enrolment[
            monthly_state_enrolment['monthly_growth'] > 0.5  # 50% growth threshold
        ]
        
        patterns['migration_indicators'] = migration_indicators
        
        # 4. Education Access Patterns
        print("\n4. Education Access Pattern Analysis:")
        # Child enrolment patterns might indicate education access
        child_enrolment_by_state = self.enrolment_df.groupby('state').agg({
            'age_5_17': 'sum',
            'total_enrolments': 'sum'
        }).reset_index()
        
        child_enrolment_by_state['child_enrolment_rate'] = (
            child_enrolment_by_state['age_5_17'] / 
            child_enrolment_by_state['total_enrolments'].replace(0, np.nan)
        )
        
        patterns['education_access_patterns'] = child_enrolment_by_state.sort_values(
            'child_enrolment_rate', ascending=False
        )
        
        return patterns
    
    def perform_cluster_analysis(self):
        """
        Perform clustering to identify similar regions
        """
        print("\n" + "="*80)
        print("REGIONAL CLUSTER ANALYSIS")
        print("="*80)
        
        # Prepare data for clustering
        state_features = self.enrolment_df.groupby('state').agg({
            'total_enrolments': 'sum',
            'child_percentage': 'mean',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        # Add update metrics if available
        if self.demographic_df is not None:
            demo_by_state = self.demographic_df.groupby('state')['total_demo_updates'].sum().reset_index()
            state_features = pd.merge(state_features, demo_by_state, on='state', how='left')
        
        # Normalize features
        numeric_cols = state_features.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(state_features[numeric_cols])
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        state_features['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette score plot
        axes[0].plot(k_range, silhouette_scores, marker='o', linewidth=2)
        axes[0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('Optimal Cluster Determination', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True, alpha=0.3)
        
        # PCA visualization
        scatter = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=state_features['cluster'], cmap='viridis', 
                                 s=100, alpha=0.7)
        
        # Annotate states
        for i, state in enumerate(state_features['state']):
            axes[1].annotate(state, (pca_result[i, 0], pca_result[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        axes[1].set_title('State Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        axes[1].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1])
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze cluster characteristics
        cluster_analysis = state_features.groupby('cluster').agg({
            'total_enrolments': 'mean',
            'child_percentage': 'mean',
            'total_demo_updates': 'mean' if 'total_demo_updates' in state_features.columns else None
        }).round(2)
        
        print(f"\nIdentified {optimal_k} distinct regional clusters:")
        print(cluster_analysis)
        
        return {
            'state_clusters': state_features,
            'optimal_k': optimal_k,
            'cluster_characteristics': cluster_analysis
        }
    
    def detect_anomalies_and_outliers(self):
        """
        Detect anomalies and outliers in the data
        """
        print("\n" + "="*80)
        print("ANOMALY AND OUTLIER DETECTION")
        print("="*80)
        
        anomalies = {}
        
        # 1. Statistical Anomaly Detection
        print("\n1. Statistical Anomaly Detection:")
        
        # Analyze enrolment patterns by state
        state_enrolment_stats = self.enrolment_df.groupby('state').agg({
            'total_enrolments': ['mean', 'std', 'count']
        }).round(2)
        
        state_enrolment_stats.columns = ['mean_enrolments', 'std_enrolments', 'count']
        
        # Identify states with unusual patterns
        state_enrolment_stats['z_score'] = (
            (state_enrolment_stats['mean_enrolments'] - state_enrolment_stats['mean_enrolments'].mean()) /
            state_enrolment_stats['mean_enrolments'].std()
        )
        
        # Flag anomalies (beyond 2 standard deviations)
        state_enrolment_stats['is_anomaly'] = abs(state_enrolment_stats['z_score']) > 2
        
        anomalies['statistical_anomalies'] = state_enrolment_stats[
            state_enrolment_stats['is_anomaly']
        ]
        
        # 2. Temporal Anomalies
        print("\n2. Temporal Anomaly Detection:")
        daily_enrolment = self.enrolment_df.groupby('date')['total_enrolments'].sum().reset_index()
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        daily_enrolment['anomaly_score'] = iso_forest.fit_predict(
            daily_enrolment['total_enrolments'].values.reshape(-1, 1)
        )
        
        temporal_anomalies = daily_enrolment[daily_enrolment['anomaly_score'] == -1]
        anomalies['temporal_anomalies'] = temporal_anomalies
        
        # 3. Geographic Anomalies
        print("\n3. Geographic Anomaly Detection:")
        
        # Analyze pincode-level anomalies
        pincode_stats = self.enrolment_df.groupby('pincode').agg({
            'total_enrolments': ['sum', 'count']
        }).round(2)
        
        pincode_stats.columns = ['total_enrolments', 'record_count']
        
        # Flag pincodes with very high or low enrolment densities
        pincode_stats['enrolment_per_record'] = (
            pincode_stats['total_enrolments'] / pincode_stats['record_count']
        )
        
        # Identify outliers using IQR method
        Q1 = pincode_stats['enrolment_per_record'].quantile(0.25)
        Q3 = pincode_stats['enrolment_per_record'].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = (
            (pincode_stats['enrolment_per_record'] < (Q1 - 1.5 * IQR)) |
            (pincode_stats['enrolment_per_record'] > (Q3 + 1.5 * IQR))
        )
        
        anomalies['geographic_anomalies'] = pincode_stats[outlier_mask]
        
        # Visualize anomalies
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Statistical anomalies
        anomaly_states = state_enrolment_stats[state_enrolment_stats['is_anomaly']]
        axes[0, 0].bar(range(len(anomaly_states)), anomaly_states['z_score'].abs())
        axes[0, 0].set_xticks(range(len(anomaly_states)))
        axes[0, 0].set_xticklabels(anomaly_states.index, rotation=45)
        axes[0, 0].set_title('Statistical Anomalies by State', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Z-score (absolute)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temporal anomalies
        axes[0, 1].plot(daily_enrolment['date'], daily_enrolment['total_enrolments'], 
                       label='Normal', alpha=0.5)
        axes[0, 1].scatter(temporal_anomalies['date'], temporal_anomalies['total_enrolments'],
                          color='red', label='Anomaly', s=50)
        axes[0, 1].set_title('Temporal Anomalies in Daily Enrolments', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Geographic anomalies
        axes[1, 0].hist(pincode_stats['enrolment_per_record'].dropna(), bins=50, alpha=0.7)
        axes[1, 0].axvline(Q1 - 1.5 * IQR, color='r', linestyle='--', label='Lower Bound')
        axes[1, 0].axvline(Q3 + 1.5 * IQR, color='r', linestyle='--', label='Upper Bound')
        axes[1, 0].set_title('Enrolment Density Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Enrolments per Record')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Anomaly summary
        axes[1, 1].axis('off')
        summary_text = f"""
        Anomaly Detection Summary:
        
        Statistical Anomalies: {len(anomalies['statistical_anomalies'])} states
        Temporal Anomalies: {len(anomalies['temporal_anomalies'])} days
        Geographic Anomalies: {len(anomalies['geographic_anomalies'])} pincodes
        
        Key Insights:
        1. States with unusual enrolment patterns may need investigation
        2. Temporal anomalies could indicate system issues or special events
        3. Geographic anomalies might show data quality issues or special regions
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return anomalies
    
    def build_predictive_models(self):
        """
        Build predictive models for future trends
        """
        print("\n" + "="*80)
        print("PREDICTIVE MODELING")
        print("="*80)
        
        predictions = {}
        
        # 1. Time Series Forecasting
        print("\n1. Time Series Forecasting:")
        
        # Prepare time series data
        monthly_enrolment = self.enrolment_df.groupby(
            pd.Grouper(key='date', freq='M')
        )['total_enrolments'].sum().reset_index()
        
        # Prepare data for Prophet
        prophet_df = monthly_enrolment[['date', 'total_enrolments']].rename(
            columns={'date': 'ds', 'total_enrolments': 'y'}
        )
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        
        # Visualize forecast
        fig = model.plot(forecast)
        plt.title('Aadhaar Enrolment Forecast (12 Months)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Enrolments')
        plt.grid(True, alpha=0.3)
        plt.savefig('enrolment_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. State-wise Growth Prediction
        print("\n2. State-wise Growth Prediction:")
        
        # Calculate growth rates by state
        state_monthly = self.enrolment_df.groupby(['state', pd.Grouper(key='date', freq='M')])[
            'total_enrolments'
        ].sum().reset_index()
        
        # Calculate rolling growth rates
        state_monthly['growth_rate'] = state_monthly.groupby('state')[
            'total_enrolments'
        ].pct_change(periods=3)  # 3-month growth rate
        
        # Predict future growth based on historical patterns
        state_growth_predictions = state_monthly.groupby('state').agg({
            'growth_rate': ['mean', 'std', 'count']
        }).round(4)
        
        state_growth_predictions.columns = ['avg_growth_rate', 'growth_volatility', 'data_points']
        
        # Classify states by growth potential
        state_growth_predictions['growth_potential'] = pd.qcut(
            state_growth_predictions['avg_growth_rate'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        predictions['state_growth_predictions'] = state_growth_predictions
        
        # 3. Update Pattern Prediction
        print("\n3. Update Pattern Prediction:")
        
        if self.demographic_df is not None:
            # Analyze update frequency patterns
            demo_patterns = self.demographic_df.groupby('state').agg({
                'total_demo_updates': ['sum', 'mean', 'std']
            }).round(2)
            
            demo_patterns.columns = ['total_updates', 'avg_updates', 'update_volatility']
            
            # Predict future update needs
            demo_patterns['update_intensity'] = pd.qcut(
                demo_patterns['avg_updates'], 
                q=4, 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            predictions['update_pattern_predictions'] = demo_patterns
        
        return {
            'time_series_forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            'state_predictions': predictions
        }
    
    def generate_insights_and_recommendations(self):
        """
        Generate actionable insights and recommendations
        """
        print("\n" + "="*80)
        print("INSIGHTS AND RECOMMENDATIONS")
        print("="*80)
        
        insights = []
        
        # 1. Demographic Insights
        print("\n1. Demographic Insights:")
        
        # Calculate national averages
        national_child_percentage = self.enrolment_df['child_percentage'].mean()
        national_adult_percentage = 1 - national_child_percentage
        
        insights.append(f"• National child enrolment rate: {national_child_percentage:.1%}")
        insights.append(f"• National adult enrolment rate: {national_adult_percentage:.1%}")
        
        # Identify states with unusual patterns
        state_child_rates = self.enrolment_df.groupby('state')['child_percentage'].mean().sort_values()
        
        lowest_child_states = state_child_rates.head(3)
        highest_child_states = state_child_rates.tail(3)
        
        insights.append(f"\n• States with lowest child enrolment rates:")
        for state, rate in lowest_child_states.items():
            insights.append(f"  - {state}: {rate:.1%}")
        
        insights.append(f"\n• States with highest child enrolment rates:")
        for state, rate in highest_child_states.items():
            insights.append(f"  - {state}: {rate:.1%}")
        
        # 2. Temporal Insights
        print("\n2. Temporal Insights:")
        
        # Seasonal patterns
        monthly_patterns = self.enrolment_df.groupby('month')['total_enrolments'].sum()
        peak_month = monthly_patterns.idxmax()
        low_month = monthly_patterns.idxmin()
        
        insights.append(f"\n• Peak enrolment month: {peak_month} ({monthly_patterns[peak_month]:,} enrolments)")
        insights.append(f"• Lowest enrolment month: {low_month} ({monthly_patterns[low_month]:,} enrolments)")
        
        # Growth trends
        monthly_growth = monthly_patterns.pct_change().mean()
        insights.append(f"• Average monthly growth rate: {monthly_growth:.2%}")
        
        # 3. Geographic Insights
        print("\n3. Geographic Insights:")
        
        # Regional disparities
        region_enrolment = self.enrolment_df.groupby('state')['total_enrolments'].sum()
        top_region = region_enrolment.idxmax()
        bottom_region = region_enrolment.idxmin()
        
        insights.append(f"\n• Highest enrolment state: {top_region} ({region_enrolment[top_region]:,})")
        insights.append(f"• Lowest enrolment state: {bottom_region} ({region_enrolment[bottom_region]:,})")
        
        # Calculate enrolment density
        states_count = self.enrolment_df['state'].nunique()
        avg_enrolment_per_state = region_enrolment.mean()
        insights.append(f"• Average enrolments per state: {avg_enrolment_per_state:,.0f}")
        
        # 4. System Improvement Recommendations
        print("\n4. System Improvement Recommendations:")
        
        recommendations = [
            "\nA. ENROLMENT SYSTEM IMPROVEMENTS:",
            "1. Targeted campaigns in low-enrolment regions",
            "2. Mobile enrolment units for remote areas",
            "3. Simplified process for child enrolments",
            "4. Integration with other government databases",
            
            "\nB. UPDATE PROCESS OPTIMIZATION:",
            "1. Proactive reminder system for updates",
            "2. Streamlined biometric update process",
            "3. Mobile app for demographic updates",
            "4. Automated age-based update triggers",
            
            "\nC. DATA QUALITY ENHANCEMENTS:",
            "1. Regular data validation checks",
            "2. Anomaly detection system",
            "3. Geographic data standardization",
            "4. Temporal pattern monitoring",
            
            "\nD. PREDICTIVE ANALYTICS DEPLOYMENT:",
            "1. Real-time enrolment forecasting",
            "2. Resource allocation optimization",
            "3. Fraud detection algorithms",
            "4. Policy impact simulation"
        ]
        
        # Combine insights and recommendations
        full_report = insights + recommendations
        
        # Save report to file
        with open('aadhaar_insights_report.txt', 'w') as f:
            for line in full_report:
                f.write(line + '\n')
                print(line)
        
        print(f"\n✓ Full report saved to 'aadhaar_insights_report.txt'")
        
        return full_report
    
    def create_dashboard_visualizations(self):
        """
        Create comprehensive dashboard visualizations
        """
        print("\n" + "="*80)
        print("INTERACTIVE DASHBOARD VISUALIZATIONS")
        print("="*80)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Monthly Enrolment Trends', 'State-wise Distribution',
                'Age Group Analysis', 'Temporal Patterns',
                'Geographic Heatmap', 'Update Frequency',
                'Growth Rates', 'Cluster Analysis', 'Anomaly Detection'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Monthly Enrolment Trends
        monthly_data = self.enrolment_df.groupby(
            pd.Grouper(key='date', freq='M')
        )['total_enrolments'].sum().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['total_enrolments'],
                mode='lines+markers',
                name='Monthly Enrolments',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. State-wise Distribution
        state_data = self.enrolment_df.groupby('state')['total_enrolments'].sum().nlargest(10)
        
        fig.add_trace(
            go.Bar(
                x=state_data.index,
                y=state_data.values,
                name='Top 10 States',
                marker_color='coral'
            ),
            row=1, col=2
        )
        
        # 3. Age Group Analysis
        age_totals = [
            self.enrolment_df['age_0_5'].sum(),
            self.enrolment_df['age_5_17'].sum(),
            self.enrolment_df['age_18_greater'].sum()
        ]
        
        fig.add_trace(
            go.Pie(
                labels=['0-5 years', '5-17 years', '18+ years'],
                values=age_totals,
                hole=0.3,
                name='Age Distribution'
            ),
            row=1, col=3
        )
        
        # 4. Temporal Patterns (Heatmap)
        heatmap_data = self.enrolment_df.groupby(['year', 'month'])['total_enrolments'].sum().unstack()
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                name='Monthly Heatmap'
            ),
            row=2, col=1
        )
        
        # 5. Geographic Analysis
        if 'pincode' in self.enrolment_df.columns:
            geo_data = self.enrolment_df.groupby('state').agg({
                'total_enrolments': 'sum',
                'child_percentage': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=geo_data['total_enrolments'],
                    y=geo_data['child_percentage'],
                    mode='markers',
                    text=geo_data['state'],
                    marker=dict(
                        size=geo_data['total_enrolments'] / geo_data['total_enrolments'].max() * 50,
                        color=geo_data['child_percentage'],
                        colorscale='Rainbow',
                        showscale=True
                    ),
                    name='State Analysis'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1400,
            title_text="Aadhaar Data Analytics Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html("aadhaar_dashboard.html")
        print("✓ Interactive dashboard saved to 'aadhaar_dashboard.html'")
        
        return fig
    
    def run_comprehensive_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("\n" + "="*100)
        print("AADHAAR SOCIETAL TRENDS ANALYSIS - COMPREHENSIVE PIPELINE")
        print("="*100)
        
        try:
            # Step 1: Data Loading
            self.load_and_preprocess_data()
            
            # Step 2: EDA
            eda_results = self.perform_eda()
            
            # Step 3: Pattern Detection
            patterns = self.detect_societal_patterns()
            
            # Step 4: Cluster Analysis
            clusters = self.perform_cluster_analysis()
            
            # Step 5: Anomaly Detection
            anomalies = self.detect_anomalies_and_outliers()
            
            # Step 6: Predictive Modeling
            predictions = self.build_predictive_models()
            
            # Step 7: Generate Insights
            insights = self.generate_insights_and_recommendations()
            
            # Step 8: Create Dashboard
            dashboard = self.create_dashboard_visualizations()
            
            print("\n" + "="*100)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*100)
            
            # Summary
            print("\nGENERATED OUTPUT FILES:")
            print("1. eda_analysis.png - Exploratory Data Analysis")
            print("2. cluster_analysis.png - Cluster Analysis")
            print("3. anomaly_detection.png - Anomaly Detection")
            print("4. enrolment_forecast.png - Time Series Forecast")
            print("5. aadhaar_insights_report.txt - Insights & Recommendations")
            print("6. aadhaar_dashboard.html - Interactive Dashboard")
            
            return {
                'eda': eda_results,
                'patterns': patterns,
                'clusters': clusters,
                'anomalies': anomalies,
                'predictions': predictions,
                'insights': insights
            }
            
        except Exception as e:
            print(f"\nError in analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AadhaarSocietalTrendsAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # If analysis successful, print key findings
    if results:
        print("\n" + "="*80)
        print("KEY FINDINGS SUMMARY")
        print("="*80)
        
        # Extract key metrics
        if 'eda' in results and 'age_distribution' in results['eda']:
            age_data = results['eda']['age_distribution']
            print(f"\n1. Age Distribution:")
            for age_group, count in age_data.items():
                print(f"   • {age_group}: {count:,}")
        
        if 'patterns' in results and 'digital_literacy_indicators' in results['patterns']:
            digital_data = results['patterns']['digital_literacy_indicators']
            if not digital_data.empty:
                print(f"\n2. Digital Literacy Indicators:")
                print(f"   • Top state for updates: {digital_data.iloc[0]['state']}")
                print(f"   • Average update rate: {digital_data['update_rate'].mean():.2%}")
        
        if 'anomalies' in results:
            print(f"\n3. Anomaly Detection Summary:")
            print(f"   • Statistical anomalies: {len(results['anomalies']['statistical_anomalies'])} states")
            print(f"   • Temporal anomalies: {len(results['anomalies']['temporal_anomalies'])} days")
            print(f"   • Geographic anomalies: {len(results['anomalies']['geographic_anomalies'])} pincodes")