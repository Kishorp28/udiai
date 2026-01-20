"""
AADHAAR SOCIETAL TRENDS ANALYZER - ROBUST VERSION
Fixed data type issues and added robust error handling
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class RobustAadhaarAnalyzer:
    """
    Robust analyzer for Aadhaar data with proper error handling
    """
    
    def __init__(self, base_path="E:/UDAI/hack"):
        self.base_path = base_path
        self.enrolment_df = None
        self.demographic_df = None
        self.biometric_df = None
        
        # Setup visualization
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def safe_load_csv(self, file_path):
        """Safely load CSV with various encodings and error handling"""
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"  ✓ Loaded with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  ✗ Failed with {encoding}: {e}")
                
        # If all encodings fail, try with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
            print(f"  ✓ Loaded with error replacement")
            return df
        except Exception as e:
            print(f"  ✗ Completely failed to load: {e}")
            return None
    
    def convert_numeric_columns(self, df, exclude_cols=None):
        """Convert object columns to numeric where possible"""
        if exclude_cols is None:
            exclude_cols = ['date', 'state', 'district', 'pincode']
            
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype == 'object':
                try:
                    # Remove any commas and convert to numeric
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                    print(f"    Converted {col} to numeric")
                except:
                    # If conversion fails, keep as is
                    pass
        return df
    
    def load_and_preprocess_data(self):
        """
        Load all datasets with robust preprocessing
        """
        print("="*80)
        print("DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        # 1. Load enrolment data
        print("\n1. Loading Enrolment Data...")
        enrolment_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_enrolment", "*.csv"))
        
        enrolment_dfs = []
        for file in enrolment_files:
            print(f"  Processing {os.path.basename(file)}")
            df = self.safe_load_csv(file)
            if df is not None:
                enrolment_dfs.append(df)
        
        if enrolment_dfs:
            self.enrolment_df = pd.concat(enrolment_dfs, ignore_index=True)
            
            # Convert numeric columns
            self.enrolment_df = self.convert_numeric_columns(self.enrolment_df)
            
            # Parse dates
            self.enrolment_df['date'] = pd.to_datetime(self.enrolment_df['date'], dayfirst=True, errors='coerce')
            
            # Create derived features
            self.enrolment_df['year'] = self.enrolment_df['date'].dt.year
            self.enrolment_df['month'] = self.enrolment_df['date'].dt.month
            self.enrolment_df['quarter'] = self.enrolment_df['date'].dt.quarter
            
            # Calculate total enrolments safely
            age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
            for col in age_cols:
                if col in self.enrolment_df.columns:
                    self.enrolment_df[col] = pd.to_numeric(self.enrolment_df[col], errors='coerce').fillna(0)
            
            if all(col in self.enrolment_df.columns for col in age_cols):
                self.enrolment_df['total_enrolments'] = (
                    self.enrolment_df['age_0_5'] + 
                    self.enrolment_df['age_5_17'] + 
                    self.enrolment_df['age_18_greater']
                )
            
            print(f"   ✓ Loaded {len(self.enrolment_df):,} enrolment records")
            print(f"   Date range: {self.enrolment_df['date'].min()} to {self.enrolment_df['date'].max()}")
        
        # 2. Load demographic data
        print("\n2. Loading Demographic Update Data...")
        demo_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_demographic", "*.csv"))
        
        demo_dfs = []
        for file in demo_files:
            print(f"  Processing {os.path.basename(file)}")
            df = self.safe_load_csv(file)
            if df is not None:
                demo_dfs.append(df)
        
        if demo_dfs:
            self.demographic_df = pd.concat(demo_dfs, ignore_index=True)
            
            # Convert numeric columns
            self.demographic_df = self.convert_numeric_columns(self.demographic_df)
            
            # Parse dates
            self.demographic_df['date'] = pd.to_datetime(self.demographic_df['date'], dayfirst=True, errors='coerce')
            
            # Identify numeric columns for total calculation
            numeric_cols = self.demographic_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude date-related columns if they were converted
            numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day']]
            
            if numeric_cols:
                # Sum only numeric columns
                self.demographic_df['total_demo_updates'] = self.demographic_df[numeric_cols].sum(axis=1, min_count=1)
                print(f"   ✓ Loaded {len(self.demographic_df):,} demographic records")
                print(f"   Total update columns found: {len(numeric_cols)}")
        
        # 3. Load biometric data
        print("\n3. Loading Biometric Update Data...")
        bio_files = glob.glob(os.path.join(self.base_path, "api_data_aadhar_biometric", "*.csv"))
        
        bio_dfs = []
        for file in bio_files:
            print(f"  Processing {os.path.basename(file)}")
            df = self.safe_load_csv(file)
            if df is not None:
                bio_dfs.append(df)
        
        if bio_dfs:
            self.biometric_df = pd.concat(bio_dfs, ignore_index=True)
            
            # Convert numeric columns
            self.biometric_df = self.convert_numeric_columns(self.biometric_df)
            
            # Parse dates
            self.biometric_df['date'] = pd.to_datetime(self.biometric_df['date'], dayfirst=True, errors='coerce')
            
            # Identify numeric columns for total calculation
            numeric_cols = self.biometric_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day']]
            
            if numeric_cols:
                # Sum only numeric columns
                self.biometric_df['total_bio_updates'] = self.biometric_df[numeric_cols].sum(axis=1, min_count=1)
                print(f"   ✓ Loaded {len(self.biometric_df):,} biometric records")
                print(f"   Total update columns found: {len(numeric_cols)}")
        
        print("\n" + "-"*80)
        print("DATA SUMMARY:")
        print(f"Enrolment records: {len(self.enrolment_df) if self.enrolment_df is not None else 0:,}")
        print(f"Demographic records: {len(self.demographic_df) if self.demographic_df is not None else 0:,}")
        print(f"Biometric records: {len(self.biometric_df) if self.biometric_df is not None else 0:,}")
        print("-"*80)
        
        return self
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis
        """
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        if self.enrolment_df is None:
            print("No enrolment data available!")
            return None
        
        results = {}
        
        # 1. Basic Statistics
        print("\n1. BASIC STATISTICS:")
        print(f"   Total records: {len(self.enrolment_df):,}")
        print(f"   States: {self.enrolment_df['state'].nunique()}")
        print(f"   Districts: {self.enrolment_df['district'].nunique()}")
        
        if 'total_enrolments' in self.enrolment_df.columns:
            total_enrolments = self.enrolment_df['total_enrolments'].sum()
            print(f"   Total enrolments: {total_enrolments:,}")
            
            # Age distribution
            age_totals = {
                '0-5 years': self.enrolment_df['age_0_5'].sum(),
                '5-17 years': self.enrolment_df['age_5_17'].sum(),
                '18+ years': self.enrolment_df['age_18_greater'].sum()
            }
            
            print("\n   Age Distribution:")
            for age_group, total in age_totals.items():
                percentage = (total / total_enrolments * 100) if total_enrolments > 0 else 0
                print(f"     {age_group}: {total:,} ({percentage:.1f}%)")
            
            results['age_distribution'] = age_totals
        
        # 2. Temporal Analysis
        print("\n2. TEMPORAL ANALYSIS:")
        
        if 'date' in self.enrolment_df.columns:
            # Monthly trends
            self.enrolment_df['month_year'] = self.enrolment_df['date'].dt.to_period('M')
            monthly_trends = self.enrolment_df.groupby('month_year')['total_enrolments'].sum()
            
            print(f"   Date range: {self.enrolment_df['date'].min()} to {self.enrolment_df['date'].max()}")
            print(f"   Peak month: {monthly_trends.idxmax()} ({monthly_trends.max():,} enrolments)")
            print(f"   Average monthly: {monthly_trends.mean():,.0f} enrolments")
            
            results['monthly_trends'] = monthly_trends
        
        # 3. Geographic Analysis
        print("\n3. GEOGRAPHIC ANALYSIS:")
        
        state_analysis = self.enrolment_df.groupby('state').agg({
            'total_enrolments': 'sum'
        }).sort_values('total_enrolments', ascending=False)
        
        print(f"\n   Top 5 States:")
        for i, (state, total) in enumerate(state_analysis.head(5).iterrows(), 1):
            print(f"     {i}. {state}: {total['total_enrolments']:,}")
        
        print(f"\n   Bottom 5 States:")
        for i, (state, total) in enumerate(state_analysis.tail(5).iterrows(), 1):
            print(f"     {i}. {state}: {total['total_enrolments']:,}")
        
        results['state_analysis'] = state_analysis
        
        # 4. Create Visualizations
        self.create_eda_visualizations()
        
        return results
    
    def create_eda_visualizations(self):
        """Create EDA visualizations"""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Monthly Trends
        if 'month_year' in self.enrolment_df.columns:
            monthly_data = self.enrolment_df.groupby('month_year')['total_enrolments'].sum()
            axes[0, 0].plot(monthly_data.index.astype(str), monthly_data.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Monthly Enrolment Trends', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Enrolments')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top States
        state_totals = self.enrolment_df.groupby('state')['total_enrolments'].sum().nlargest(10)
        axes[0, 1].barh(range(len(state_totals)), state_totals.values)
        axes[0, 1].set_yticks(range(len(state_totals)))
        axes[0, 1].set_yticklabels(state_totals.index)
        axes[0, 1].set_title('Top 10 States by Enrolments', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Total Enrolments')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Age Distribution
        age_data = [
            self.enrolment_df['age_0_5'].sum(),
            self.enrolment_df['age_5_17'].sum(),
            self.enrolment_df['age_18_greater'].sum()
        ]
        age_labels = ['0-5 years', '5-17 years', '18+ years']
        
        axes[0, 2].pie(age_data, labels=age_labels, autopct='%1.1f%%', 
                      startangle=90, explode=(0.05, 0.05, 0.05))
        axes[0, 2].set_title('Age Distribution', fontsize=12, fontweight='bold')
        
        # 4. District Analysis
        if 'district' in self.enrolment_df.columns:
            district_counts = self.enrolment_df['district'].value_counts().head(10)
            axes[1, 0].bar(range(len(district_counts)), district_counts.values)
            axes[1, 0].set_xticks(range(len(district_counts)))
            axes[1, 0].set_xticklabels(district_counts.index, rotation=45, ha='right')
            axes[1, 0].set_title('Top 10 Districts by Records', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Number of Records')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Temporal Distribution (Box plot by month)
        if 'month' in self.enrolment_df.columns:
            monthly_stats = self.enrolment_df.groupby('month')['total_enrolments'].sum()
            axes[1, 1].bar(range(1, 13), [monthly_stats.get(i, 0) for i in range(1, 13)])
            axes[1, 1].set_title('Monthly Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Enrolments')
            axes[1, 1].set_xticks(range(1, 13))
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Update Patterns (if available)
        if self.demographic_df is not None and 'total_demo_updates' in self.demographic_df.columns:
            demo_by_state = self.demographic_df.groupby('state')['total_demo_updates'].sum().nlargest(10)
            axes[1, 2].bar(range(len(demo_by_state)), demo_by_state.values, color='green', alpha=0.7)
            axes[1, 2].set_xticks(range(len(demo_by_state)))
            axes[1, 2].set_xticklabels(demo_by_state.index, rotation=45, ha='right')
            axes[1, 2].set_title('Top 10 States - Demographic Updates', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Total Updates')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Alternative: Enrolment density by state
            state_density = self.enrolment_df['state'].value_counts().head(10)
            axes[1, 2].bar(range(len(state_density)), state_density.values, color='purple', alpha=0.7)
            axes[1, 2].set_xticks(range(len(state_density)))
            axes[1, 2].set_xticklabels(state_density.index, rotation=45, ha='right')
            axes[1, 2].set_title('State Record Density', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Number of Records')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ EDA visualizations saved as 'eda_visualizations.png'")
        plt.show()
    
    def detect_anomalies(self):
        """
        Detect anomalies in the data
        """
        print("\n" + "="*80)
        print("ANOMALY DETECTION")
        print("="*80)
        
        if self.enrolment_df is None:
            print("No data available for anomaly detection!")
            return None
        
        anomalies = {}
        
        # 1. Statistical Anomalies by State
        print("\n1. Statistical Anomalies by State:")
        
        state_stats = self.enrolment_df.groupby('state').agg({
            'total_enrolments': ['mean', 'std', 'count']
        })
        state_stats.columns = ['mean', 'std', 'count']
        
        # Calculate z-scores
        state_stats['z_score'] = (state_stats['mean'] - state_stats['mean'].mean()) / state_stats['mean'].std()
        
        # Flag anomalies (beyond 2 standard deviations)
        state_stats['is_anomaly'] = abs(state_stats['z_score']) > 2
        
        anomaly_states = state_stats[state_stats['is_anomaly']]
        
        if not anomaly_states.empty:
            print(f"   Found {len(anomaly_states)} anomalous states:")
            for state, row in anomaly_states.iterrows():
                print(f"     • {state}: z-score = {row['z_score']:.2f}")
            
            anomalies['state_anomalies'] = anomaly_states
        else:
            print("   No statistical anomalies found by state")
        
        # 2. Temporal Anomalies
        print("\n2. Temporal Anomalies:")
        
        daily_enrolment = self.enrolment_df.groupby('date')['total_enrolments'].sum().reset_index()
        
        # Use simple statistical method for anomaly detection
        Q1 = daily_enrolment['total_enrolments'].quantile(0.25)
        Q3 = daily_enrolment['total_enrolments'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        temporal_anomalies = daily_enrolment[
            (daily_enrolment['total_enrolments'] < lower_bound) | 
            (daily_enrolment['total_enrolments'] > upper_bound)
        ]
        
        if not temporal_anomalies.empty:
            print(f"   Found {len(temporal_anomalies)} anomalous days")
            anomalies['temporal_anomalies'] = temporal_anomalies
        else:
            print("   No temporal anomalies found")
        
        # 3. Geographic Anomalies (Pincode level)
        print("\n3. Geographic Anomalies:")
        
        if 'pincode' in self.enrolment_df.columns:
            pincode_stats = self.enrolment_df.groupby('pincode').agg({
                'total_enrolments': 'sum'
            })
            
            # Calculate enrolment density (enrolments per pincode)
            pincode_stats['enrolment_density'] = pd.qcut(
                pincode_stats['total_enrolments'], 
                q=4, 
                labels=['Very Low', 'Low', 'High', 'Very High']
            )
            
            # Find pincodes with extreme values
            extreme_pincodes = pincode_stats[
                pincode_stats['total_enrolments'] > pincode_stats['total_enrolments'].quantile(0.99)
            ]
            
            if not extreme_pincodes.empty:
                print(f"   Found {len(extreme_pincodes)} pincodes with extreme enrolment values")
                anomalies['geographic_anomalies'] = extreme_pincodes
            else:
                print("   No geographic anomalies found")
        
        # Visualize anomalies
        self.create_anomaly_visualizations(anomalies)
        
        return anomalies
    
    def create_anomaly_visualizations(self, anomalies):
        """Create anomaly detection visualizations"""
        print("\nCreating anomaly visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. State Anomalies
        if 'state_anomalies' in anomalies:
            anomaly_data = anomalies['state_anomalies']
            axes[0, 0].barh(range(len(anomaly_data)), anomaly_data['z_score'].abs())
            axes[0, 0].set_yticks(range(len(anomaly_data)))
            axes[0, 0].set_yticklabels(anomaly_data.index)
            axes[0, 0].set_title('State Anomalies (Z-score)', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Absolute Z-score')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No State Anomalies', 
                           ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('State Anomalies', fontsize=12, fontweight='bold')
        
        # 2. Temporal Anomalies
        daily_enrolment = self.enrolment_df.groupby('date')['total_enrolments'].sum().reset_index()
        
        axes[0, 1].plot(daily_enrolment['date'], daily_enrolment['total_enrolments'], 
                       alpha=0.7, label='Daily Enrolments')
        
        if 'temporal_anomalies' in anomalies:
            anomaly_dates = anomalies['temporal_anomalies']
            axes[0, 1].scatter(anomaly_dates['date'], anomaly_dates['total_enrolments'],
                             color='red', s=50, label='Anomalies', zorder=5)
        
        axes[0, 1].set_title('Temporal Anomalies', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Enrolments')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Geographic Distribution
        if 'geographic_anomalies' in anomalies:
            geo_data = anomalies['geographic_anomalies']
            axes[1, 0].hist(geo_data['total_enrolments'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Extreme Pincode Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Enrolments per Pincode')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Geographic Anomalies', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Geographic Anomalies', fontsize=12, fontweight='bold')
        
        # 4. Anomaly Summary
        axes[1, 1].axis('off')
        
        summary_text = "ANOMALY DETECTION SUMMARY\n\n"
        
        if 'state_anomalies' in anomalies:
            summary_text += f"State Anomalies: {len(anomalies['state_anomalies'])}\n"
        else:
            summary_text += "State Anomalies: 0\n"
        
        if 'temporal_anomalies' in anomalies:
            summary_text += f"Temporal Anomalies: {len(anomalies['temporal_anomalies'])}\n"
        else:
            summary_text += "Temporal Anomalies: 0\n"
        
        if 'geographic_anomalies' in anomalies:
            summary_text += f"Geographic Anomalies: {len(anomalies['geographic_anomalies'])}\n"
        else:
            summary_text += "Geographic Anomalies: 0\n"
        
        summary_text += "\nRECOMMENDATIONS:\n"
        summary_text += "1. Investigate anomalous states\n"
        summary_text += "2. Check temporal anomalies for data issues\n"
        summary_text += "3. Validate extreme pincode data\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
        print("✓ Anomaly visualizations saved as 'anomaly_detection.png'")
        plt.show()
    
    def perform_cluster_analysis(self):
        """
        Perform cluster analysis to identify similar regions
        """
        print("\n" + "="*80)
        print("CLUSTER ANALYSIS")
        print("="*80)
        
        if self.enrolment_df is None:
            print("No data available for cluster analysis!")
            return None
        
        # Prepare state-level features
        state_features = self.enrolment_df.groupby('state').agg({
            'total_enrolments': 'sum',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        # Calculate percentages
        state_features['child_percentage'] = (
            (state_features['age_0_5'] + state_features['age_5_17']) / 
            state_features['total_enrolments'].replace(0, np.nan)
        )
        
        # Add demographic update data if available
        if self.demographic_df is not None and 'total_demo_updates' in self.demographic_df.columns:
            demo_by_state = self.demographic_df.groupby('state')['total_demo_updates'].sum().reset_index()
            state_features = pd.merge(state_features, demo_by_state, on='state', how='left')
        
        # Prepare data for clustering
        numeric_cols = ['total_enrolments', 'child_percentage']
        if 'total_demo_updates' in state_features.columns:
            numeric_cols.append('total_demo_updates')
        
        clustering_data = state_features[numeric_cols].fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(clustering_data)
        
        # Determine optimal number of clusters
        print("\nDetermining optimal number of clusters...")
        
        silhouette_scores = []
        k_range = range(2, min(8, len(state_features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(scaled_features, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {optimal_k}")
            
            # Perform clustering with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            state_features['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Visualize clusters
            self.create_cluster_visualizations(state_features, scaled_features, optimal_k, silhouette_scores)
            
            # Analyze cluster characteristics
            print(f"\nCluster Characteristics:")
            cluster_analysis = state_features.groupby('cluster').agg({
                'total_enrolments': ['mean', 'std'],
                'child_percentage': 'mean',
                'state': 'count'
            }).round(2)
            
            print(cluster_analysis)
            
            return {
                'state_clusters': state_features,
                'optimal_k': optimal_k,
                'cluster_analysis': cluster_analysis
            }
        else:
            print("Cannot perform clustering with available data")
            return None
    
    def create_cluster_visualizations(self, state_features, scaled_features, optimal_k, silhouette_scores):
        """Create cluster analysis visualizations"""
        print("\nCreating cluster visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Silhouette Score Plot
        axes[0].plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o', linewidth=2)
        axes[0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('Optimal Cluster Determination', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].grid(True, alpha=0.3)
        
        # 2. PCA Visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        scatter = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=state_features['cluster'], cmap='viridis', 
                                 s=100, alpha=0.7, edgecolor='black')
        
        # Annotate some states
        for i, state in enumerate(state_features['state']):
            if i % 3 == 0:  # Annotate every 3rd state to avoid clutter
                axes[1].annotate(state, (pca_result[i, 0], pca_result[i, 1]), 
                               fontsize=8, alpha=0.7)
        
        axes[1].set_title('State Clusters (PCA Visualization)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Cluster Characteristics
        cluster_stats = state_features.groupby('cluster').agg({
            'total_enrolments': 'mean',
            'child_percentage': 'mean'
        }).round(2)
        
        x = np.arange(len(cluster_stats))
        width = 0.35
        
        axes[2].bar(x - width/2, cluster_stats['total_enrolments'], width, label='Avg Enrolments', alpha=0.7)
        axes[2].bar(x + width/2, cluster_stats['child_percentage'] * 10000, width, 
                   label='Child % (x100)', alpha=0.7)  # Scaled for visualization
        
        axes[2].set_title('Cluster Characteristics', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Cluster')
        axes[2].set_ylabel('Values')
        axes[2].set_xticks(x)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[1])
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Cluster analysis saved as 'cluster_analysis.png'")
        plt.show()
    
    def generate_insights_report(self):
        """
        Generate comprehensive insights report
        """
        print("\n" + "="*80)
        print("GENERATING INSIGHTS REPORT")
        print("="*80)
        
        if self.enrolment_df is None:
            print("No data available for insights generation!")
            return None
        
        insights = []
        
        # 1. Key Statistics
        total_enrolments = self.enrolment_df['total_enrolments'].sum()
        total_states = self.enrolment_df['state'].nunique()
        total_districts = self.enrolment_df['district'].nunique()
        
        insights.append("="*60)
        insights.append("AADHAAR DATA ANALYSIS - KEY INSIGHTS")
        insights.append("="*60)
        insights.append("")
        insights.append("1. OVERALL STATISTICS:")
        insights.append(f"   • Total records analyzed: {len(self.enrolment_df):,}")
        insights.append(f"   • Total enrolments counted: {total_enrolments:,}")
        insights.append(f"   • States covered: {total_states}")
        insights.append(f"   • Districts covered: {total_districts}")
        insights.append("")
        
        # 2. Demographic Insights
        child_total = self.enrolment_df['age_0_5'].sum() + self.enrolment_df['age_5_17'].sum()
        adult_total = self.enrolment_df['age_18_greater'].sum()
        child_percentage = (child_total / total_enrolments * 100) if total_enrolments > 0 else 0
        
        insights.append("2. DEMOGRAPHIC INSIGHTS:")
        insights.append(f"   • Child enrolments (0-17 years): {child_total:,} ({child_percentage:.1f}%)")
        insights.append(f"   • Adult enrolments (18+ years): {adult_total:,} ({100-child_percentage:.1f}%)")
        insights.append("")
        
        # 3. Temporal Insights
        if 'date' in self.enrolment_df.columns:
            monthly_trends = self.enrolment_df.groupby(
                self.enrolment_df['date'].dt.to_period('M')
            )['total_enrolments'].sum()
            
            peak_month = monthly_trends.idxmax()
            peak_value = monthly_trends.max()
            avg_monthly = monthly_trends.mean()
            
            insights.append("3. TEMPORAL INSIGHTS:")
            insights.append(f"   • Peak enrolment month: {peak_month} ({peak_value:,} enrolments)")
            insights.append(f"   • Average monthly enrolments: {avg_monthly:,.0f}")
            insights.append("")
        
        # 4. Geographic Insights
        state_totals = self.enrolment_df.groupby('state')['total_enrolments'].sum()
        top_state = state_totals.idxmax()
        top_state_value = state_totals.max()
        
        insights.append("4. GEOGRAPHIC INSIGHTS:")
        insights.append(f"   • Top state: {top_state} ({top_state_value:,} enrolments)")
        
        # Calculate concentration
        top_5_total = state_totals.nlargest(5).sum()
        concentration = (top_5_total / total_enrolments * 100) if total_enrolments > 0 else 0
        
        insights.append(f"   • Top 5 states concentration: {concentration:.1f}% of total enrolments")
        insights.append("")
        
        # 5. System Performance Indicators
        insights.append("5. SYSTEM PERFORMANCE INDICATORS:")
        
        if self.demographic_df is not None:
            total_updates = self.demographic_df['total_demo_updates'].sum() if 'total_demo_updates' in self.demographic_df.columns else 0
            update_rate = (total_updates / total_enrolments * 100) if total_enrolments > 0 else 0
            
            insights.append(f"   • Total demographic updates: {total_updates:,}")
            insights.append(f"   • Update rate: {update_rate:.2f}%")
        
        if self.biometric_df is not None:
            total_bio_updates = self.biometric_df['total_bio_updates'].sum() if 'total_bio_updates' in self.biometric_df.columns else 0
            bio_update_rate = (total_bio_updates / total_enrolments * 100) if total_enrolments > 0 else 0
            
            insights.append(f"   • Total biometric updates: {total_bio_updates:,}")
            insights.append(f"   • Biometric update rate: {bio_update_rate:.2f}%")
        
        insights.append("")
        
        # 6. Recommendations
        insights.append("6. RECOMMENDATIONS FOR SYSTEM IMPROVEMENT:")
        insights.append("")
        insights.append("   A. TARGETED ENROLMENT DRIVES:")
        insights.append("      1. Focus on states with low enrolment rates")
        insights.append("      2. Implement mobile enrolment units in remote areas")
        insights.append("      3. Simplify child enrolment process")
        insights.append("")
        insights.append("   B. UPDATE PROCESS OPTIMIZATION:")
        insights.append("      1. Proactive reminders for demographic updates")
        insights.append("      2. Streamlined biometric update centers")
        insights.append("      3. Age-based automatic update triggers")
        insights.append("")
        insights.append("   C. DATA QUALITY ENHANCEMENT:")
        insights.append("      1. Regular data validation checks")
        insights.append("      2. Automated anomaly detection system")
        insights.append("      3. Geographic data standardization")
        insights.append("")
        insights.append("   D. ANALYTICS AND MONITORING:")
        insights.append("      1. Real-time enrolment dashboard")
        insights.append("      2. Predictive analytics for resource allocation")
        insights.append("      3. Fraud detection algorithms")
        
        # Save report to file
        with open('aadhaar_insights_report.txt', 'w', encoding='utf-8') as f:
            for line in insights:
                f.write(line + '\n')
                print(line)
        
        print("\n" + "="*60)
        print("✓ Insights report saved to 'aadhaar_insights_report.txt'")
        print("="*60)
        
        return insights
    
    def export_data(self):
        """Export processed data for further analysis"""
        print("\n" + "="*80)
        print("EXPORTING PROCESSED DATA")
        print("="*80)
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Export enrolment data
        if self.enrolment_df is not None:
            self.enrolment_df.to_csv('output/enrolment_processed.csv', index=False)
            print("✓ Enrolment data exported to 'output/enrolment_processed.csv'")
        
        # Export demographic data
        if self.demographic_df is not None:
            self.demographic_df.to_csv('output/demographic_processed.csv', index=False)
            print("✓ Demographic data exported to 'output/demographic_processed.csv'")
        
        # Export biometric data
        if self.biometric_df is not None:
            self.biometric_df.to_csv('output/biometric_processed.csv', index=False)
            print("✓ Biometric data exported to 'output/biometric_processed.csv'")
        
        # Create summary statistics
        summary_data = []
        
        if self.enrolment_df is not None:
            summary_data.append({
                'dataset': 'enrolment',
                'records': len(self.enrolment_df),
                'states': self.enrolment_df['state'].nunique(),
                'total_enrolments': self.enrolment_df['total_enrolments'].sum() if 'total_enrolments' in self.enrolment_df.columns else 0
            })
        
        if self.demographic_df is not None:
            summary_data.append({
                'dataset': 'demographic',
                'records': len(self.demographic_df),
                'states': self.demographic_df['state'].nunique() if 'state' in self.demographic_df.columns else 0,
                'total_updates': self.demographic_df['total_demo_updates'].sum() if 'total_demo_updates' in self.demographic_df.columns else 0
            })
        
        if self.biometric_df is not None:
            summary_data.append({
                'dataset': 'biometric',
                'records': len(self.biometric_df),
                'states': self.biometric_df['state'].nunique() if 'state' in self.biometric_df.columns else 0,
                'total_updates': self.biometric_df['total_bio_updates'].sum() if 'total_bio_updates' in self.biometric_df.columns else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('output/dataset_summary.csv', index=False)
        print("✓ Dataset summary exported to 'output/dataset_summary.csv'")
        
        return summary_df
    
    def run_complete_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("\n" + "="*80)
        print("AADHAAR SOCIETAL TRENDS ANALYSIS")
        print("="*80)
        
        try:
            # Step 1: Load data
            self.load_and_preprocess_data()
            
            # Step 2: Exploratory Data Analysis
            eda_results = self.exploratory_data_analysis()
            
            # Step 3: Anomaly Detection
            anomalies = self.detect_anomalies()
            
            # Step 4: Cluster Analysis
            clusters = self.perform_cluster_analysis()
            
            # Step 5: Generate Insights
            insights = self.generate_insights_report()
            
            # Step 6: Export Data
            exports = self.export_data()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            print("\nGENERATED FILES:")
            print("1. eda_visualizations.png - Exploratory Data Analysis")
            print("2. anomaly_detection.png - Anomaly Detection Results")
            print("3. cluster_analysis.png - Cluster Analysis")
            print("4. aadhaar_insights_report.txt - Comprehensive Insights")
            print("5. output/ - Processed data files")
            
            return {
                'success': True,
                'eda': eda_results,
                'anomalies': anomalies,
                'clusters': clusters,
                'insights': insights
            }
            
        except Exception as e:
            print(f"\nError in analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RobustAadhaarAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary if successful
    if results.get('success', False):
        print("\n" + "="*80)
        print("SUMMARY OF KEY FINDINGS")
        print("="*80)
        
        if 'eda' in results and 'age_distribution' in results['eda']:
            age_data = results['eda']['age_distribution']
            print("\nAge Distribution:")
            for age_group, total in age_data.items():
                print(f"  • {age_group}: {total:,}")
        
        if 'anomalies' in results:
            anomaly_counts = {}
            for key, value in results['anomalies'].items():
                if hasattr(value, '__len__'):
                    anomaly_counts[key] = len(value)
                else:
                    anomaly_counts[key] = 0
            
            print("\nAnomalies Detected:")
            for anomaly_type, count in anomaly_counts.items():
                print(f"  • {anomaly_type}: {count}")
        
        print("\n" + "="*80)
        print("Check the generated files for complete analysis results!")
        print("="*80)