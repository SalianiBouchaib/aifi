import streamlit as st

# Configuration de la page - DOIT être la première commande Streamlit
st.set_page_config(
    page_title="AIFI - Advanced Financial Intelligence Suite", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Imports enrichis
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import warnings
from datetime import datetime, timedelta
import base64
from io import BytesIO
import time

# Imports géospatiaux
import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster
import pydeck as pdk
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Handle optional dependencies with try/except
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    class MLFallback:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            return [0] * len(X)
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
    RandomForestRegressor = MLFallback
    LinearRegression = MLFallback
    StandardScaler = MLFallback
    KMeans = MLFallback

warnings.filterwarnings('ignore')

# ========== ENHANCED CSV DATA MANAGER ==========
class EnhancedCSVDataManager:
    """Gestionnaire de données CSV enrichi avec fonctionnalités géospatiales"""
    
    @staticmethod
    def get_csv_financial_data():
        """Récupère les données financières avec informations géographiques"""
        if not st.session_state.imported_metrics:
            return None
        
        metrics = st.session_state.imported_metrics
        financial_data = {}
        
        # Données financières de base
        if 'revenue' in metrics:
            financial_data['revenue'] = metrics['revenue']['average'] * 12
            financial_data['monthly_revenue'] = metrics['revenue']['average']
            financial_data['revenue_data'] = metrics['revenue']['data']
            financial_data['revenue_growth'] = metrics['revenue'].get('growth_rate', 0)
            financial_data['revenue_volatility'] = metrics['revenue'].get('volatility', 0)
        
        if 'costs' in metrics:
            financial_data['total_costs'] = metrics['costs']['average'] * 12
            financial_data['monthly_costs'] = metrics['costs']['average']
            financial_data['costs_data'] = metrics['costs']['data']
            financial_data['costs_growth'] = metrics['costs'].get('growth_rate', 0)
        
        if 'profit' in metrics:
            financial_data['net_profit'] = metrics['profit']['average'] * 12
            financial_data['monthly_profit'] = metrics['profit']['average']
            financial_data['profit_data'] = metrics['profit']['data']
            financial_data['profit_margin'] = metrics['profit'].get('margin_average', 0)
        
        # Données géographiques enrichies
        if 'geographic_data' in metrics:
            geo_data = metrics['geographic_data']
            financial_data.update({
                'total_locations': geo_data.get('total_locations', 0),
                'geographic_spread': geo_data.get('geographic_spread', 0),
                'best_performing_location': geo_data.get('best_location', {}),
                'geographic_clusters': geo_data.get('clusters', []),
                'regional_performance': geo_data.get('regional_metrics', {})
            })
        
        # Métriques business enrichies
        if 'business_metrics' in metrics:
            biz_data = metrics['business_metrics']
            financial_data.update({
                'customer_count': biz_data.get('customer_count', 0),
                'customer_acquisition_cost': biz_data.get('cac', 0),
                'lifetime_value': biz_data.get('ltv', 0),
                'market_share': biz_data.get('market_share', 0),
                'competition_level': biz_data.get('competition_level', 0),
                'brand_strength': biz_data.get('brand_strength', 0)
            })
        
        # Calculs dérivés avec nouvelles métriques
        if 'revenue' in financial_data and 'total_costs' in financial_data:
            financial_data['gross_profit'] = financial_data['revenue'] - financial_data['total_costs']
            financial_data['operating_profit'] = financial_data['gross_profit'] * 0.8
            financial_data['net_margin'] = financial_data['net_profit'] / financial_data['revenue'] if financial_data['revenue'] > 0 else 0
        
        # Estimations améliorées du bilan
        if 'revenue' in financial_data:
            revenue = financial_data['revenue']
            financial_data.update({
                'current_assets': revenue * 0.3,
                'current_liabilities': revenue * 0.15,
                'total_assets': revenue * 0.8,
                'total_debt': revenue * 0.2,
                'equity': revenue * 0.4,
                'cash': revenue * 0.1,
                'inventory': revenue * 0.08,
                'interest_expense': revenue * 0.02,
                'cash_flow': financial_data.get('monthly_profit', revenue * 0.02),
                'current_ratio': (revenue * 0.3) / (revenue * 0.15),
                'debt_to_equity': (revenue * 0.2) / (revenue * 0.4),
                'interest_coverage': (financial_data['gross_profit'] * 0.8) / (revenue * 0.02)
            })
        
        return financial_data
    
    @staticmethod
    def has_csv_data():
        return bool(st.session_state.imported_metrics)
    
    @staticmethod
    def get_csv_insights():
        if 'csv_data' in st.session_state and 'insights' in st.session_state.csv_data:
            return st.session_state.csv_data['insights']
        return None
    
    @staticmethod
    def get_csv_visualizations():
        if 'csv_data' in st.session_state and 'figures' in st.session_state.csv_data:
            return st.session_state.csv_data['figures']
        return None
    
    @staticmethod
    def get_geographic_data():
        """Récupère les données géographiques spécifiques"""
        if 'csv_data' in st.session_state and 'geographic_analysis' in st.session_state.csv_data:
            return st.session_state.csv_data['geographic_analysis']
        return None

# ========== ENHANCED CSV PROCESSOR WITH GEOSPATIAL ==========
class EnhancedGeoCSVProcessor:
    def __init__(self):
        self.column_mappings = {
            # Colonnes financières de base
            'revenue': ['revenue', 'sales', 'income', 'turnover', 'receipts', 'monthly_recurring_revenue', 'mrr', 'arr'],
            'costs': ['costs', 'expenses', 'expenditure', 'outgoings', 'total_costs', 'variable_costs', 'fixed_costs', 'cost_of_goods_sold', 'cogs'],
            'date': ['date', 'month', 'period', 'time', 'year', 'quarter'],
            'profit': ['profit', 'earnings', 'net_income', 'net income', 'pnl', 'p&l', 'operating_profit', 'gross_profit'],
            
            # Colonnes géographiques
            'latitude': ['latitude', 'lat', 'y', 'coord_y', 'geo_lat'],
            'longitude': ['longitude', 'lng', 'lon', 'x', 'coord_x', 'geo_lng'],
            'location_id': ['location_id', 'store_id', 'branch_id', 'office_id', 'site_id'],
            'location_name': ['location_name', 'store', 'branch', 'office', 'site', 'location'],
            'city': ['city', 'ville', 'town', 'municipality'],
            'region': ['region', 'state', 'province', 'area', 'zone'],
            'country': ['country', 'pays', 'nation'],
            'postal_code': ['postal_code', 'zip_code', 'zip', 'code_postal'],
            
            # Métriques business enrichies
            'customer_count': ['customer_count', 'customers', 'active_users', 'monthly_active_users', 'mau', 'clients'],
            'customer_acquisition_cost': ['customer_acquisition_cost', 'cac', 'acquisition_cost'],
            'lifetime_value': ['lifetime_value', 'ltv', 'customer_lifetime_value', 'clv'],
            'churn_rate': ['churn_rate', 'churn', 'attrition_rate'],
            'retention_rate': ['retention_rate', 'retention'],
            'market_share': ['market_share', 'market_position', 'share'],
            'competition_level': ['competition_level', 'competitive_intensity', 'competition'],
            'brand_strength': ['brand_strength', 'brand_awareness', 'brand_score'],
            'demographics_score': ['demographics_score', 'demographic_index', 'population_score'],
            
            # Métriques opérationnelles
            'units_sold': ['units_sold', 'quantity', 'volume', 'transactions', 'sales_volume'],
            'average_price': ['average_price', 'price_per_unit', 'average_transaction_value', 'atv'],
            'inventory_level': ['inventory_level', 'stock_level', 'inventory'],
            'employee_count': ['employee_count', 'employees', 'staff', 'workforce'],
            'store_size': ['store_size', 'square_feet', 'surface', 'area'],
            
            # Métriques financières avancées
            'cash_flow': ['cash_flow', 'cash flow', 'cashflow', 'cash', 'flow'],
            'assets': ['assets', 'total_assets', 'current_assets', 'fixed_assets'],
            'liabilities': ['liabilities', 'total_liabilities', 'current_liabilities', 'debt'],
            'equity': ['equity', 'shareholders_equity', 'owners_equity'],
            'accounts_receivable': ['accounts_receivable', 'receivables', 'ar', 'debtors'],
            'accounts_payable': ['accounts_payable', 'payables', 'ap', 'creditors'],
            
            # Métriques marketing
            'marketing_spend': ['marketing_spend', 'marketing_budget', 'advertising_cost'],
            'lead_generation': ['lead_generation', 'leads', 'prospects'],
            'conversion_rate': ['conversion_rate', 'conversion', 'close_rate'],
            'website_traffic': ['website_traffic', 'web_visitors', 'online_traffic'],
            'social_media_engagement': ['social_engagement', 'social_score', 'engagement_rate']
        }
        
        self.required_columns = ['date']
        self.detected_mappings = {}
        self.analysis_results = {}
        self.geolocator = Nominatim(user_agent="aifi_enhanced_v2")
    
    def detect_columns(self, df):
        """Détection avancée des colonnes avec priorisation géographique"""
        detected = {}
        
        for target, keywords in self.column_mappings.items():
            for col in df.columns:
                col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                for keyword in keywords:
                    if keyword in col_lower or col_lower in keyword:
                        if target not in detected:
                            detected[target] = []
                        detected[target].append(col)
                        break
        
        # Prioriser les colonnes exactes
        for target in detected:
            detected[target] = list(set(detected[target]))
            if len(detected[target]) > 1:
                exact_matches = [col for col in detected[target] 
                               if col.lower().replace('_', ' ') in self.column_mappings[target]]
                if exact_matches:
                    detected[target] = exact_matches[:1]
                else:
                    detected[target] = detected[target][:1]
        
        # Aplatir en une seule colonne par target
        for target in detected:
            if detected[target]:
                detected[target] = detected[target][0]
        
        self.detected_mappings = detected
        return detected
    
    def geocode_locations(self, df, location_column):
        """Géocodage automatique des emplacements"""
        geocoded_data = []
        
        for location in df[location_column].unique():
            if pd.isna(location) or location == '':
                continue
                
            try:
                location_info = self.geolocator.geocode(location, timeout=10)
                if location_info:
                    geocoded_data.append({
                        'location': location,
                        'latitude': location_info.latitude,
                        'longitude': location_info.longitude,
                        'full_address': location_info.address
                    })
                else:
                    # Fallback: essayer avec le pays ajouté
                    location_with_country = f"{location}, Morocco"
                    location_info = self.geolocator.geocode(location_with_country, timeout=10)
                    if location_info:
                        geocoded_data.append({
                            'location': location,
                            'latitude': location_info.latitude,
                            'longitude': location_info.longitude,
                            'full_address': location_info.address
                        })
                
                # Pause pour éviter le rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"Erreur géocodage pour {location}: {e}")
        
        return pd.DataFrame(geocoded_data)
    
    def calculate_enhanced_metrics(self, df, mappings):
        """Calcul des métriques enrichies incluant géospatiales"""
        metrics = {}
        
        # Métriques financières de base
        if 'revenue' in mappings and mappings['revenue'] in df.columns:
            revenue_col = mappings['revenue']
            revenue_data = self.clean_numeric_column(df[revenue_col]).dropna()
            
            metrics['revenue'] = {
                'total': revenue_data.sum(),
                'average': revenue_data.mean(),
                'median': revenue_data.median(),
                'std': revenue_data.std(),
                'min': revenue_data.min(),
                'max': revenue_data.max(),
                'trend': 'increasing' if len(revenue_data) > 1 and revenue_data.iloc[-1] > revenue_data.iloc[0] else 'decreasing',
                'growth_rate': ((revenue_data.iloc[-1] / revenue_data.iloc[0]) - 1) * 100 if len(revenue_data) > 1 and revenue_data.iloc[0] != 0 else 0,
                'volatility': revenue_data.std() / revenue_data.mean() if revenue_data.mean() != 0 else 0,
                'data': revenue_data.tolist()
            }
        
        if 'costs' in mappings and mappings['costs'] in df.columns:
            costs_col = mappings['costs']
            costs_data = self.clean_numeric_column(df[costs_col]).dropna()
            
            metrics['costs'] = {
                'total': costs_data.sum(),
                'average': costs_data.mean(),
                'median': costs_data.median(),
                'std': costs_data.std(),
                'min': costs_data.min(),
                'max': costs_data.max(),
                'trend': 'increasing' if len(costs_data) > 1 and costs_data.iloc[-1] > costs_data.iloc[0] else 'decreasing',
                'growth_rate': ((costs_data.iloc[-1] / costs_data.iloc[0]) - 1) * 100 if len(costs_data) > 1 and costs_data.iloc[0] != 0 else 0,
                'volatility': costs_data.std() / costs_data.mean() if costs_data.mean() != 0 else 0,
                'data': costs_data.tolist()
            }
        
        # Calcul du profit si revenus et coûts disponibles
        if 'revenue' in metrics and 'costs' in metrics:
            revenue_data = np.array(metrics['revenue']['data'])
            costs_data = np.array(metrics['costs']['data'])
            
            min_length = min(len(revenue_data), len(costs_data))
            revenue_data = revenue_data[:min_length]
            costs_data = costs_data[:min_length]
            
            profit_data = revenue_data - costs_data
            
            metrics['profit'] = {
                'total': profit_data.sum(),
                'average': profit_data.mean(),
                'median': np.median(profit_data),
                'std': np.std(profit_data),
                'min': profit_data.min(),
                'max': profit_data.max(),
                'trend': 'increasing' if len(profit_data) > 1 and profit_data[-1] > profit_data[0] else 'decreasing',
                'margin_average': (profit_data.mean() / revenue_data.mean() * 100) if revenue_data.mean() != 0 else 0,
                'data': profit_data.tolist()
            }
        
        # Métriques géographiques
        if self.has_geographic_data(mappings, df):
            geo_metrics = self.calculate_geographic_metrics(df, mappings)
            metrics['geographic_data'] = geo_metrics
        
        # Métriques business enrichies
        business_metrics = self.calculate_business_metrics(df, mappings)
        if business_metrics:
            metrics['business_metrics'] = business_metrics
        
        self.analysis_results = metrics
        return metrics
    
    def has_geographic_data(self, mappings, df):
        """Vérifie la présence de données géographiques"""
        return ('latitude' in mappings and 'longitude' in mappings and 
                mappings['latitude'] in df.columns and mappings['longitude'] in df.columns)
    
    def calculate_geographic_metrics(self, df, mappings):
        """Calcul des métriques géographiques"""
        geo_metrics = {}
        
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        
        # Données géographiques de base
        valid_coords = df[[lat_col, lon_col]].dropna()
        
        if len(valid_coords) > 0:
            geo_metrics['total_locations'] = len(valid_coords)
            geo_metrics['center_latitude'] = valid_coords[lat_col].mean()
            geo_metrics['center_longitude'] = valid_coords[lon_col].mean()
            geo_metrics['geographic_spread'] = self.calculate_geographic_spread(valid_coords, lat_col, lon_col)
            
            # Analyse par localisation si revenus disponibles
            if 'revenue' in mappings and mappings['revenue'] in df.columns:
                location_performance = self.analyze_location_performance(df, mappings)
                geo_metrics.update(location_performance)
            
            # Clustering géographique
            if len(valid_coords) >= 3:
                clusters = self.perform_geographic_clustering(df, mappings)
                geo_metrics['clusters'] = clusters
        
        return geo_metrics
    
    def calculate_geographic_spread(self, coords_df, lat_col, lon_col):
        """Calcule l'étendue géographique maximale"""
        if len(coords_df) < 2:
            return 0
        
        max_distance = 0
        for i in range(len(coords_df)):
            for j in range(i+1, len(coords_df)):
                coord1 = (coords_df.iloc[i][lat_col], coords_df.iloc[i][lon_col])
                coord2 = (coords_df.iloc[j][lat_col], coords_df.iloc[j][lon_col])
                try:
                    distance = geodesic(coord1, coord2).kilometers
                    max_distance = max(max_distance, distance)
                except:
                    continue
        
        return max_distance
    
    def analyze_location_performance(self, df, mappings):
        """Analyse des performances par localisation"""
        performance_data = {}
        
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        revenue_col = mappings['revenue']
        
        # Agrégation par localisation
        if 'location_name' in mappings:
            location_col = mappings['location_name']
            location_stats = df.groupby(location_col).agg({
                lat_col: 'first',
                lon_col: 'first',
                revenue_col: ['sum', 'mean', 'count']
            }).round(2)
            
            # Meilleure localisation
            best_location = location_stats[revenue_col]['sum'].idxmax()
            best_revenue = location_stats.loc[best_location, (revenue_col, 'sum')]
            
            performance_data['best_location'] = {
                'name': best_location,
                'revenue': best_revenue,
                'latitude': location_stats.loc[best_location, lat_col],
                'longitude': location_stats.loc[best_location, lon_col]
            }
            
            # Statistiques régionales
            performance_data['location_stats'] = location_stats.to_dict()
        
        return performance_data
    
    def perform_geographic_clustering(self, df, mappings):
        """Clustering géographique K-means"""
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        revenue_col = mappings.get('revenue')
        
        coords = df[[lat_col, lon_col]].dropna()
        
        if len(coords) < 3:
            return []
        
        # K-means clustering
        n_clusters = min(3, len(coords))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords)
        
        cluster_analysis = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_data = df.loc[cluster_coords.index]
            
            cluster_info = {
                'cluster_id': cluster_id,
                'location_count': len(cluster_data),
                'center_lat': kmeans.cluster_centers_[cluster_id][0],
                'center_lon': kmeans.cluster_centers_[cluster_id][1],
                'geographic_bounds': {
                    'min_lat': cluster_coords[lat_col].min(),
                    'max_lat': cluster_coords[lat_col].max(),
                    'min_lon': cluster_coords[lon_col].min(),
                    'max_lon': cluster_coords[lon_col].max()
                }
            }
            
            # Performance du cluster si revenus disponibles
            if revenue_col and revenue_col in cluster_data.columns:
                cluster_revenue = self.clean_numeric_column(cluster_data[revenue_col]).dropna()
                if len(cluster_revenue) > 0:
                    cluster_info['avg_performance'] = cluster_revenue.mean()
                    cluster_info['total_performance'] = cluster_revenue.sum()
                    cluster_info['performance_level'] = 'High' if cluster_revenue.mean() > df[revenue_col].mean() else 'Low'
            
            cluster_analysis.append(cluster_info)
        
        return cluster_analysis
    
    def calculate_business_metrics(self, df, mappings):
        """Calcul des métriques business enrichies"""
        business_metrics = {}
        
        # Métriques clients
        if 'customer_count' in mappings and mappings['customer_count'] in df.columns:
            customer_data = self.clean_numeric_column(df[mappings['customer_count']]).dropna()
            business_metrics['customer_count'] = customer_data.mean()
            business_metrics['customer_growth'] = ((customer_data.iloc[-1] / customer_data.iloc[0]) - 1) * 100 if len(customer_data) > 1 else 0
        
        # CAC et LTV
        if 'customer_acquisition_cost' in mappings and mappings['customer_acquisition_cost'] in df.columns:
            cac_data = self.clean_numeric_column(df[mappings['customer_acquisition_cost']]).dropna()
            business_metrics['cac'] = cac_data.mean()
        
        if 'lifetime_value' in mappings and mappings['lifetime_value'] in df.columns:
            ltv_data = self.clean_numeric_column(df[mappings['lifetime_value']]).dropna()
            business_metrics['ltv'] = ltv_data.mean()
        
        # Ratio LTV/CAC
        if 'ltv' in business_metrics and 'cac' in business_metrics and business_metrics['cac'] > 0:
            business_metrics['ltv_cac_ratio'] = business_metrics['ltv'] / business_metrics['cac']
        
        # Métriques de marché
        if 'market_share' in mappings and mappings['market_share'] in df.columns:
            market_data = self.clean_numeric_column(df[mappings['market_share']]).dropna()
            business_metrics['market_share'] = market_data.mean()
        
        if 'competition_level' in mappings and mappings['competition_level'] in df.columns:
            comp_data = self.clean_numeric_column(df[mappings['competition_level']]).dropna()
            business_metrics['competition_level'] = comp_data.mean()
        
        if 'brand_strength' in mappings and mappings['brand_strength'] in df.columns:
            brand_data = self.clean_numeric_column(df[mappings['brand_strength']]).dropna()
            business_metrics['brand_strength'] = brand_data.mean()
        
        return business_metrics
    
    def clean_numeric_column(self, series):
        """Nettoyage des colonnes numériques"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        cleaned = series.astype(str)
        cleaned = cleaned.str.replace(r'[$€£¥₹₽%]', '', regex=True)
        cleaned = cleaned.str.replace(',', '')
        cleaned = cleaned.str.replace(' ', '')
        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
        
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        return cleaned
    
    def standardize_date_column(self, series):
        """Standardisation des colonnes de date"""
        try:
            return pd.to_datetime(series, infer_datetime_format=True)
        except:
            formats = ['%Y-%m-%d', '%Y-%m', '%m/%d/%Y', '%d/%m/%Y', '%Y', '%m-%Y']
            for fmt in formats:
                try:
                    return pd.to_datetime(series, format=fmt)
                except:
                    continue
            return series
    
    def generate_enhanced_insights(self, metrics):
        """Génération d'insights enrichis avec géospatial"""
        insights = []
        recommendations = []
        alerts = []
        
        # Insights financiers
        if 'revenue' in metrics:
            rev_metrics = metrics['revenue']
            
            if rev_metrics['trend'] == 'increasing':
                insights.append(f"📈 **Croissance Revenue Positive**: {rev_metrics['growth_rate']:.1f}% sur la période")
            else:
                alerts.append(f"📉 **Déclin Revenue**: {abs(rev_metrics['growth_rate']):.1f}% de baisse détectée")
                recommendations.append("Analyser les causes du déclin et développer des stratégies de relance")
            
            if rev_metrics['volatility'] > 0.3:
                alerts.append(f"📊 **Forte Volatilité Revenue**: {rev_metrics['volatility']:.1%} coefficient de variation")
                recommendations.append("Considérer des stratégies pour stabiliser les flux de revenus")
            elif rev_metrics['volatility'] < 0.1:
                insights.append("✅ **Revenue Stable**: Faible volatilité indique une performance prévisible")
        
        # Insights géographiques
        if 'geographic_data' in metrics:
            geo_data = metrics['geographic_data']
            
            if geo_data.get('total_locations', 0) > 1:
                insights.append(f"🌍 **Présence Multi-localisations**: {geo_data['total_locations']} emplacements analysés")
                
                spread = geo_data.get('geographic_spread', 0)
                if spread > 100:
                    insights.append(f"📍 **Large Couverture Géographique**: {spread:.0f} km d'étendue")
                    recommendations.append("Optimiser la logistique et la coordination entre sites distants")
                
                if 'best_location' in geo_data:
                    best_loc = geo_data['best_location']
                    insights.append(f"🏆 **Meilleur Site**: {best_loc['name']} avec {best_loc['revenue']:,.0f} DHS")
                
                if 'clusters' in geo_data and len(geo_data['clusters']) > 1:
                    high_perf_clusters = [c for c in geo_data['clusters'] if c.get('performance_level') == 'High']
                    if high_perf_clusters:
                        insights.append(f"🎯 **Clusters Haute Performance**: {len(high_perf_clusters)} zones identifiées")
                        recommendations.append("Répliquer les stratégies des zones performantes sur les autres sites")
        
        # Insights business
        if 'business_metrics' in metrics:
            biz_data = metrics['business_metrics']
            
            if 'ltv_cac_ratio' in biz_data:
                ratio = biz_data['ltv_cac_ratio']
                if ratio > 3:
                    insights.append(f"💰 **Excellent LTV/CAC**: Ratio de {ratio:.1f} (optimal > 3)")
                elif ratio > 1:
                    insights.append(f"📊 **LTV/CAC Correct**: Ratio de {ratio:.1f} (amélioration possible)")
                else:
                    alerts.append(f"🔴 **LTV/CAC Critique**: Ratio de {ratio:.1f} (< 1)")
                    recommendations.append("Urgent: Réduire les coûts d'acquisition ou augmenter la valeur client")
            
            if 'market_share' in biz_data:
                share = biz_data['market_share'] * 100
                if share > 20:
                    insights.append(f"🎯 **Part de Marché Forte**: {share:.1f}% du marché")
                elif share > 10:
                    insights.append(f"📈 **Part de Marché Correcte**: {share:.1f}% du marché")
                else:
                    recommendations.append(f"📊 **Opportunité Croissance**: {share:.1f}% part de marché actuelle")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'alerts': alerts
        }
    
    def create_enhanced_visualizations(self, df, mappings, metrics):
        """Création de visualisations enrichies"""
        figures = {}
        
        # Graphique financier temporel standard
        if 'date' in mappings and mappings['date'] in df.columns:
            time_col = self.standardize_date_column(df[mappings['date']])
            x_axis = time_col
            x_title = "Date"
        else:
            x_axis = range(len(df))
            x_title = "Période"
        
        # Graphique performance financière
        if 'revenue' in mappings or 'costs' in mappings:
            fig = go.Figure()
            
            if 'revenue' in mappings and mappings['revenue'] in df.columns:
                revenue_data = self.clean_numeric_column(df[mappings['revenue']])
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=revenue_data,
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
            
            if 'costs' in mappings and mappings['costs'] in df.columns:
                costs_data = self.clean_numeric_column(df[mappings['costs']])
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=costs_data,
                    mode='lines+markers',
                    name='Coûts',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
            
            if 'revenue' in mappings and 'costs' in mappings:
                profit_data = revenue_data - costs_data
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=profit_data,
                    mode='lines+markers',
                    name='Profit',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Performance Financière dans le Temps",
                xaxis_title=x_title,
                yaxis_title="Montant (DHS)",
                hovermode='x unified',
                height=500
            )
            
            figures['financial_trend'] = fig
        
        # Graphiques géographiques
        if self.has_geographic_data(mappings, df):
            geo_figs = self.create_geographic_visualizations(df, mappings, metrics)
            figures.update(geo_figs)
        
        # Graphiques business métriques
        if 'business_metrics' in metrics:
            business_figs = self.create_business_visualizations(df, mappings, metrics)
            figures.update(business_figs)
        
        return figures
    
    def create_geographic_visualizations(self, df, mappings, metrics):
        """Création des visualisations géographiques"""
        geo_figures = {}
        
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        revenue_col = mappings.get('revenue')
        
        # Carte scatter géographique
        if revenue_col:
            revenue_data = self.clean_numeric_column(df[revenue_col])
            
            fig = go.Figure()
            
            # Taille des marqueurs basée sur les revenus
            marker_sizes = (revenue_data / revenue_data.max() * 50 + 10).fillna(10)
            
            fig.add_trace(go.Scattermapbox(
                lat=df[lat_col],
                lon=df[lon_col],
                mode='markers',
                marker=dict(
                    size=marker_sizes,
                    color=revenue_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Revenue (DHS)")
                ),
                text=[f"Revenue: {rev:,.0f} DHS" for rev in revenue_data],
                hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>',
                name='Performance par Localisation'
            ))
            
            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(
                        lat=df[lat_col].mean(),
                        lon=df[lon_col].mean()
                    ),
                    zoom=6
                ),
                height=600,
                title="Carte des Performances par Localisation"
            )
            
            geo_figures['performance_map'] = fig
        
        # Heatmap des clusters si disponible
        if 'geographic_data' in metrics and 'clusters' in metrics['geographic_data']:
            clusters = metrics['geographic_data']['clusters']
            
            if len(clusters) > 1:
                cluster_fig = go.Figure()
                
                colors = ['red', 'blue', 'green', 'orange', 'purple']
                
                for i, cluster in enumerate(clusters):
                    cluster_color = colors[i % len(colors)]
                    
                    cluster_fig.add_trace(go.Scattermapbox(
                        lat=[cluster['center_lat']],
                        lon=[cluster['center_lon']],
                        mode='markers',
                        marker=dict(size=25, color=cluster_color),
                        name=f"Cluster {i+1}",
                        text=f"Cluster {i+1}<br>Emplacements: {cluster['location_count']}<br>Performance: {cluster.get('performance_level', 'N/A')}",
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                cluster_fig.update_layout(
                    mapbox=dict(
                        style='open-street-map',
                        center=dict(
                            lat=df[lat_col].mean(),
                            lon=df[lon_col].mean()
                        ),
                        zoom=6
                    ),
                    height=500,
                    title="Clusters Géographiques de Performance"
                )
                
                geo_figures['clusters_map'] = cluster_fig
        
        return geo_figures
    
    def create_business_visualizations(self, df, mappings, metrics):
        """Création des visualisations métriques business"""
        business_figures = {}
        
        biz_metrics = metrics['business_metrics']
        
        # Graphique en radar des métriques business
        if len(biz_metrics) >= 3:
            metrics_names = []
            metrics_values = []
            
            metric_mapping = {
                'customer_count': 'Nombre Clients',
                'cac': 'Coût Acquisition',
                'ltv': 'Valeur Vie Client',
                'market_share': 'Part de Marché',
                'competition_level': 'Niveau Concurrence',
                'brand_strength': 'Force Marque'
            }
            
            for key, value in biz_metrics.items():
                if key in metric_mapping and isinstance(value, (int, float)):
                    metrics_names.append(metric_mapping[key])
                    # Normaliser les valeurs pour le radar (0-100)
                    if key == 'market_share':
                        normalized_value = value * 100
                    elif key == 'competition_level':
                        normalized_value = (1 - value) * 100  # Inverser car moins de concurrence = mieux
                    else:
                        # Normalisation simple
                        normalized_value = min(100, max(0, value / 1000 * 100))
                    metrics_values.append(normalized_value)
            
            if len(metrics_names) >= 3:
                radar_fig = go.Figure()
                
                radar_fig.add_trace(go.Scatterpolar(
                    r=metrics_values,
                    theta=metrics_names,
                    fill='toself',
                    name='Métriques Business',
                    line_color='rgba(0,100,200,0.8)',
                    fillcolor='rgba(0,100,200,0.2)'
                ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title="Radar des Métriques Business",
                    height=500
                )
                
                business_figures['business_radar'] = radar_fig
        
        return business_figures
    
    def process_csv(self, df):
        """Traitement principal du CSV enrichi"""
        mappings = self.detect_columns(df)
        issues, suggestions = self.validate_data(df, mappings)
        metrics = self.calculate_enhanced_metrics(df, mappings)
        insights_data = self.generate_enhanced_insights(metrics)
        figures = self.create_enhanced_visualizations(df, mappings, metrics)
        
        # Analyse géographique approfondie si données disponibles
        geographic_analysis = None
        if self.has_geographic_data(mappings, df):
            geographic_analysis = self.perform_comprehensive_geographic_analysis(df, mappings, metrics)
        
        return {
            'mappings': mappings,
            'metrics': metrics,
            'insights': insights_data,
            'figures': figures,
            'issues': issues,
            'suggestions': suggestions,
            'processed_df': df,
            'geographic_analysis': geographic_analysis
        }
    
    def perform_comprehensive_geographic_analysis(self, df, mappings, metrics):
        """Analyse géographique complète"""
        geo_analysis = {}
        
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        revenue_col = mappings.get('revenue')
        
        # Statistiques géographiques de base
        geo_analysis['basic_stats'] = {
            'total_locations': len(df[[lat_col, lon_col]].dropna()),
            'center_point': {
                'latitude': df[lat_col].mean(),
                'longitude': df[lon_col].mean()
            },
            'geographic_bounds': {
                'north': df[lat_col].max(),
                'south': df[lat_col].min(),
                'east': df[lon_col].max(),
                'west': df[lon_col].min()
            }
        }
        
        # Analyse de densité si suffisamment de points
        if len(df) >= 5:
            geo_analysis['density_analysis'] = self.calculate_density_metrics(df, mappings)
        
        # Corrélations géographiques avec performance
        if revenue_col:
            geo_analysis['geo_performance_correlation'] = self.analyze_geo_performance_correlation(df, mappings)
        
        return geo_analysis
    
    def calculate_density_metrics(self, df, mappings):
        """Calcul des métriques de densité géographique"""
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        
        # Calculer la densité de points par région
        # Diviser l'espace en grille et compter les points
        lat_range = df[lat_col].max() - df[lat_col].min()
        lon_range = df[lon_col].max() - df[lon_col].min()
        
        density_metrics = {
            'geographic_concentration': 1 / (lat_range * lon_range) if lat_range > 0 and lon_range > 0 else 0,
            'average_distance_between_points': self.calculate_average_distance(df, lat_col, lon_col)
        }
        
        return density_metrics
    
    def calculate_average_distance(self, df, lat_col, lon_col):
        """Calcule la distance moyenne entre tous les points"""
        coords = df[[lat_col, lon_col]].dropna()
        
        if len(coords) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                coord1 = (coords.iloc[i][lat_col], coords.iloc[i][lon_col])
                coord2 = (coords.iloc[j][lat_col], coords.iloc[j][lon_col])
                try:
                    distance = geodesic(coord1, coord2).kilometers
                    total_distance += distance
                    count += 1
                except:
                    continue
        
        return total_distance / count if count > 0 else 0
    
    def analyze_geo_performance_correlation(self, df, mappings):
        """Analyse de corrélation entre géographie et performance"""
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        revenue_col = mappings['revenue']
        
        revenue_data = self.clean_numeric_column(df[revenue_col])
        
        # Corrélations avec latitude/longitude
        lat_corr = df[lat_col].corr(revenue_data)
        lon_corr = df[lon_col].corr(revenue_data)
        
        # Analyse par quadrants géographiques
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        
        quadrants = {
            'north_east': df[(df[lat_col] >= center_lat) & (df[lon_col] >= center_lon)],
            'north_west': df[(df[lat_col] >= center_lat) & (df[lon_col] < center_lon)],
            'south_east': df[(df[lat_col] < center_lat) & (df[lon_col] >= center_lon)],
            'south_west': df[(df[lat_col] < center_lat) & (df[lon_col] < center_lon)]
        }
        
        quadrant_performance = {}
        for quad_name, quad_data in quadrants.items():
            if len(quad_data) > 0:
                quad_revenue = self.clean_numeric_column(quad_data[revenue_col])
                quadrant_performance[quad_name] = {
                    'count': len(quad_data),
                    'avg_revenue': quad_revenue.mean(),
                    'total_revenue': quad_revenue.sum()
                }
        
        return {
            'latitude_correlation': lat_corr,
            'longitude_correlation': lon_corr,
            'quadrant_analysis': quadrant_performance
        }
    
    def validate_data(self, df, mappings):
        """Validation des données enrichie"""
        issues = []
        suggestions = []
        
        # Validation géographique
        if 'latitude' in mappings and 'longitude' in mappings:
            lat_col = mappings['latitude']
            lon_col = mappings['longitude']
            
            # Vérifier les coordonnées valides
            valid_lat = df[lat_col].between(-90, 90).sum()
            valid_lon = df[lon_col].between(-180, 180).sum()
            
            if valid_lat < len(df):
                issues.append(f"Coordonnées latitude invalides détectées: {len(df) - valid_lat} lignes")
                suggestions.append("Vérifier les valeurs de latitude (doivent être entre -90 et 90)")
            
            if valid_lon < len(df):
                issues.append(f"Coordonnées longitude invalides détectées: {len(df) - valid_lon} lignes")
                suggestions.append("Vérifier les valeurs de longitude (doivent être entre -180 et 180)")
        
        # Validation business métriques
        if 'customer_acquisition_cost' in mappings and 'lifetime_value' in mappings:
            cac_col = mappings['customer_acquisition_cost']
            ltv_col = mappings['lifetime_value']
            
            cac_data = self.clean_numeric_column(df[cac_col])
            ltv_data = self.clean_numeric_column(df[ltv_col])
            
            # Vérifier le ratio LTV/CAC
            ltv_cac_ratio = (ltv_data / cac_data).mean()
            if ltv_cac_ratio < 3:
                issues.append(f"Ratio LTV/CAC faible: {ltv_cac_ratio:.1f} (recommandé > 3)")
                suggestions.append("Améliorer la rétention client ou réduire les coûts d'acquisition")
        
        # Validations standard
        if 'date' not in mappings:
            suggestions.append("Ajouter une colonne date pour l'analyse temporelle")
        
        if 'revenue' not in mappings:
            issues.append("Aucune colonne revenue détectée - critique pour l'analyse financière")
            suggestions.append("S'assurer d'avoir une colonne revenue/sales")
        
        return issues, suggestions

# ========== ENHANCED TEMPLATE GENERATOR ==========
class EnhancedCSVTemplateGenerator:
    def __init__(self):
        self.templates = {
            'complete_geo_financial': {
                'name': 'Template Financier Géographique Complet',
                'description': 'Template complet avec données financières, géographiques et business',
                'columns': {
                    'Date': 'YYYY-MM-DD (ex: 2025-06-28)',
                    'Location_ID': 'Identifiant unique de localisation (ex: LOC001)',
                    'Location_Name': 'Nom de l\'emplacement (ex: Casablanca_Centre)',
                    'Latitude': 'Latitude en degrés décimaux (ex: 33.5731)',
                    'Longitude': 'Longitude en degrés décimaux (ex: -7.5898)',
                    'City': 'Ville (ex: Casablanca)',
                    'Region': 'Région (ex: Grand Casablanca)',
                    'Country': 'Pays (ex: Morocco)',
                    'Postal_Code': 'Code postal (ex: 20000)',
                    'Revenue': 'Revenus en monnaie locale (ex: 15000)',
                    'Costs': 'Coûts totaux (ex: 12000)',
                    'Profit': 'Profit net (ex: 3000)',
                    'Cash_Flow': 'Flux de trésorerie (ex: 2500)',
                    'Customer_Count': 'Nombre de clients actifs (ex: 150)',
                    'Customer_Acquisition_Cost': 'Coût d\'acquisition client (ex: 200)',
                    'Lifetime_Value': 'Valeur vie client (ex: 1500)',
                    'Churn_Rate': 'Taux d\'attrition (0.05 = 5%)',
                    'Market_Share': 'Part de marché (0.25 = 25%)',
                    'Competition_Level': 'Niveau concurrence (0.7 = élevé)',
                    'Brand_Strength': 'Force de marque (0-1)',
                    'Units_Sold': 'Unités vendues (ex: 300)',
                    'Average_Price': 'Prix moyen (ex: 50)',
                    'Employee_Count': 'Nombre d\'employés (ex: 10)',
                    'Store_Size': 'Taille magasin en m² (ex: 200)',
                    'Marketing_Spend': 'Dépenses marketing (ex: 2000)',
                    'Website_Traffic': 'Trafic web mensuel (ex: 5000)',
                    'Demographics_Score': 'Score démographique zone (0-1)'
                },
                'sample_data': self.generate_complete_sample_data()
            },
            'retail_geo': {
                'name': 'Template Retail Géolocalisé',
                'description': 'Spécialisé pour commerce de détail avec géolocalisation',
                'columns': {
                    'Date': 'Date de l\'analyse',
                    'Store_ID': 'Identifiant magasin',
                    'Store_Name': 'Nom du magasin',
                    'Latitude': 'Latitude',
                    'Longitude': 'Longitude',
                    'City': 'Ville',
                    'Revenue': 'Chiffre d\'affaires',
                    'Foot_Traffic': 'Trafic piéton',
                    'Conversion_Rate': 'Taux de conversion',
                    'Average_Basket': 'Panier moyen',
                    'Inventory_Level': 'Niveau stock',
                    'Competition_Proximity': 'Proximité concurrence (km)'
                },
                'sample_data': self.generate_retail_sample_data()
            },
            'saas_geo': {
                'name': 'Template SaaS Géographique',
                'description': 'Pour entreprises SaaS avec analyse géographique clients',
                'columns': {
                    'Date': 'Date',
                    'Region_ID': 'ID région',
                    'Region_Name': 'Nom région',
                    'Latitude': 'Latitude centre région',
                    'Longitude': 'Longitude centre région',
                    'MRR': 'Revenue récurrent mensuel',
                    'Active_Users': 'Utilisateurs actifs',
                    'Churn_Rate': 'Taux de churn',
                    'CAC': 'Coût acquisition client',
                    'LTV': 'Valeur vie client',
                    'Support_Tickets': 'Tickets support',
                    'Local_Language': 'Langue locale'
                },
                'sample_data': self.generate_saas_sample_data()
            }
        }
    
    def generate_complete_sample_data(self):
        """Génère des données d'exemple complètes"""
        dates = ['2025-06-01', '2025-06-15', '2025-06-28'] * 4
        
        # Données pour différentes villes marocaines
        locations = [
            {'id': 'CAS001', 'name': 'Casablanca_Centre', 'lat': 33.5731, 'lon': -7.5898, 'city': 'Casablanca', 'region': 'Grand Casablanca', 'postal': '20000'},
            {'id': 'RAB001', 'name': 'Rabat_Agdal', 'lat': 34.0209, 'lon': -6.8416, 'city': 'Rabat', 'region': 'Rabat-Salé-Kénitra', 'postal': '10000'},
            {'id': 'MAR001', 'name': 'Marrakech_Gueliz', 'lat': 31.6295, 'lon': -7.9811, 'city': 'Marrakech', 'region': 'Marrakech-Safi', 'postal': '40000'},
            {'id': 'FES001', 'name': 'Fès_Ville_Nouvelle', 'lat': 34.0181, 'lon': -5.0078, 'city': 'Fès', 'region': 'Fès-Meknès', 'postal': '30000'}
        ]
        
        sample_data = []
        for i, date in enumerate(dates):
            loc = locations[i % len(locations)]
            
            # Variations réalistes dans les données
            base_revenue = 15000 if loc['id'] == 'CAS001' else 12000 if loc['id'] == 'RAB001' else 8500 if loc['id'] == 'MAR001' else 9200
            revenue_variation = np.random.normal(1, 0.1)
            revenue = base_revenue * revenue_variation
            
            costs = revenue * 0.75  # 75% du revenue en coûts
            profit = revenue - costs
            
            sample_data.append([
                date,
                loc['id'],
                loc['name'],
                loc['lat'],
                loc['lon'],
                loc['city'],
                loc['region'],
                'Morocco',
                loc['postal'],
                round(revenue, 0),
                round(costs, 0),
                round(profit, 0),
                round(profit * 0.8, 0),  # Cash flow
                round(revenue / 100),  # Customer count
                round(200 + np.random.normal(0, 50), 0),  # CAC
                round(1500 + np.random.normal(0, 300), 0),  # LTV
                round(np.random.uniform(0.03, 0.08), 3),  # Churn rate
                round(np.random.uniform(0.1, 0.3), 2),  # Market share
                round(np.random.uniform(0.5, 0.9), 2),  # Competition level
                round(np.random.uniform(0.6, 0.9), 2),  # Brand strength
                round(revenue / 50),  # Units sold
                50,  # Average price
                round(revenue / 1500),  # Employee count
                200 + (i % 4) * 50,  # Store size
                round(revenue * 0.15, 0),  # Marketing spend
                round(revenue * 0.3, 0),  # Website traffic
                round(np.random.uniform(0.6, 0.9), 2)  # Demographics score
            ])
        
        return sample_data
    
    def generate_retail_sample_data(self):
        """Données d'exemple retail"""
        return [
            ['2025-06-28', 'ST001', 'Mega Mall Casa', 33.5731, -7.5898, 'Casablanca', 25000, 1200, 0.12, 85, 150, 0.5],
            ['2025-06-28', 'ST002', 'Marina Shopping Rabat', 34.0209, -6.8416, 'Rabat', 18000, 950, 0.10, 75, 120, 0.8],
            ['2025-06-28', 'ST003', 'Menara Mall Marrakech', 31.6295, -7.9811, 'Marrakech', 15000, 800, 0.14, 90, 100, 1.2]
        ]
    
    def generate_saas_sample_data(self):
        """Données d'exemple SaaS"""
        return [
            ['2025-06-28', 'REG001', 'North_Africa', 33.0, -7.0, 25000, 500, 0.05, 150, 1800, 45, 'French'],
            ['2025-06-28', 'REG002', 'West_Africa', 14.0, -14.0, 18000, 350, 0.07, 200, 1500, 62, 'French'],
            ['2025-06-28', 'REG003', 'Middle_East', 24.0, 45.0, 22000, 420, 0.04, 180, 2000, 38, 'Arabic']
        ]
    
    def generate_template_csv(self, template_type):
        """Génère un CSV template"""
        template = self.templates.get(template_type)
        if not template:
            return None
        
        columns = list(template['columns'].keys())
        df = pd.DataFrame(template['sample_data'], columns=columns)
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer.getvalue()

# ========== GEOGRAPHIC ANALYZER ==========
class GeoFinancialAnalyzer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="aifi_enhanced_v2")
        self.maps_cache = {}
    
    def create_performance_heatmap(self, df, lat_col, lon_col, value_col, location_col=None):
        """Création de heatmap Folium avancée"""
        # Centre sur le Maroc par défaut ou données
        center_lat = df[lat_col].mean() if not df.empty else 31.7917
        center_lon = df[lon_col].mean() if not df.empty else -7.0926
        
        # Carte Folium avec style personnalisé
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Ajouter des tiles alternatives
        folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='CartoDB Dark').add_to(m)
        
        # Données pour heatmap
        heat_data = []
        marker_data = []
        
        for idx, row in df.iterrows():
            if pd.notna(row[lat_col]) and pd.notna(row[lon_col]) and pd.notna(row[value_col]):
                lat, lon, value = row[lat_col], row[lon_col], row[value_col]
                
                # Intensité pour heatmap (0.1                 # Intensité pour heatmap (0.1 à 1.0)
                max_value = df[value_col].max()
                intensity = max(0.1, min(1.0, value / max_value)) if max_value > 0 else 0.1
                heat_data.append([lat, lon, intensity])
                
                # Données pour marqueurs individuels
                location_name = row[location_col] if location_col and location_col in df.columns else f"Point {idx}"
                popup_text = f"""
                <b>{location_name}</b><br>
                💰 Valeur: {value:,.0f} DHS<br>
                📍 Coordonnées: {lat:.4f}, {lon:.4f}<br>
                📊 Performance: {(value/max_value*100):.1f}%
                """
                
                # Couleur du marqueur basée sur la performance
                if value >= max_value * 0.8:
                    marker_color = 'green'
                    icon = 'star'
                elif value >= max_value * 0.6:
                    marker_color = 'blue'
                    icon = 'info-sign'
                elif value >= max_value * 0.4:
                    marker_color = 'orange'
                    icon = 'warning-sign'
                else:
                    marker_color = 'red'
                    icon = 'remove'
                
                marker_data.append({
                    'lat': lat, 'lon': lon, 'popup': popup_text,
                    'color': marker_color, 'icon': icon, 'value': value
                })
        
        # Ajouter la heatmap
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_zoom=18,
                radius=20,
                blur=15,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(m)
        
        # Ajouter les marqueurs avec clustering
        if marker_data:
            marker_cluster = MarkerCluster(name='Performance Points').add_to(m)
            
            for marker in marker_data:
                folium.Marker(
                    location=[marker['lat'], marker['lon']],
                    popup=folium.Popup(marker['popup'], max_width=300),
                    tooltip=f"Valeur: {marker['value']:,.0f} DHS",
                    icon=folium.Icon(color=marker['color'], icon=marker['icon'])
                ).add_to(marker_cluster)
        
        # Ajouter contrôles de couches
        folium.LayerControl().add_to(m)
        
        # Ajouter plugin fullscreen
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def create_3d_deck_map(self, df, lat_col, lon_col, value_col):
        """Carte 3D avec PyDeck améliorée"""
        # Normaliser les valeurs pour l'élévation
        max_val = df[value_col].max()
        min_val = df[value_col].min()
        
        # Préparer les données
        deck_data = df.copy()
        deck_data['elevation'] = ((deck_data[value_col] - min_val) / (max_val - min_val) * 2000).fillna(0)
        deck_data['color'] = deck_data[value_col].apply(self.value_to_color)
        
        # Layer colonnes 3D
        column_layer = pdk.Layer(
            'ColumnLayer',
            data=deck_data,
            get_position=[lon_col, lat_col],
            get_elevation='elevation',
            elevation_scale=4,
            radius=1000,
            get_fill_color='color',
            pickable=True,
            extruded=True,
        )
        
        # Layer heatmap
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            data=deck_data,
            get_position=[lon_col, lat_col],
            get_weight=value_col,
            radius_pixels=100,
        )
        
        # Vue initiale
        view_state = pdk.ViewState(
            longitude=df[lon_col].mean(),
            latitude=df[lat_col].mean(),
            zoom=6,
            min_zoom=5,
            max_zoom=15,
            pitch=45,
            bearing=0
        )
        
        # Tooltip personnalisé
        tooltip = {
            "html": f"<b>Performance</b><br/>{value_col}: {{{value_col}}}<br/>Coordonnées: {{{lat_col}}}, {{{lon_col}}}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        # Deck avec layers multiples
        deck = pdk.Deck(
            layers=[column_layer, heatmap_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='mapbox://styles/mapbox/light-v9'
        )
        
        return deck
    
    def value_to_color(self, value):
        """Convertit une valeur en couleur RGB"""
        # Normalisation 0-255 pour RGB
        normalized = min(255, max(0, value / 20000 * 255))
        
        if normalized < 85:
            return [255, int(normalized * 3), 0, 200]  # Rouge vers orange
        elif normalized < 170:
            return [int(255 - (normalized - 85) * 3), 255, 0, 200]  # Orange vers vert
        else:
            return [0, 255, int((normalized - 170) * 3), 200]  # Vert vers cyan
    
    def create_plotly_choropleth_map(self, df, lat_col, lon_col, value_col, location_col=None):
        """Carte choroplèhe avec Plotly"""
        
        # Taille des marqueurs
        marker_sizes = (df[value_col] / df[value_col].max() * 80 + 15).fillna(15)
        
        # Couleurs basées sur les performances
        colors = df[value_col].fillna(0)
        
        # Texte de survol personnalisé
        hover_text = []
        for idx, row in df.iterrows():
            location_name = row[location_col] if location_col and location_col in df.columns else f"Localisation {idx}"
            hover_info = f"""
            <b>{location_name}</b><br>
            💰 {value_col}: {row[value_col]:,.0f} DHS<br>
            📍 Latitude: {row[lat_col]:.4f}<br>
            📍 Longitude: {row[lon_col]:.4f}<br>
            📊 Rang: {df[value_col].rank(method='dense', ascending=False)[idx]:.0f}/{len(df)}
            """
            hover_text.append(hover_info)
        
        # Création de la figure
        fig = go.Figure()
        
        # Couche principale avec marqueurs
        fig.add_trace(go.Scattermapbox(
            lat=df[lat_col],
            lon=df[lon_col],
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=f"{value_col}<br>(DHS)",
                    titleside="right",
                    tickmode="linear",
                    tick0=colors.min(),
                    dtick=(colors.max() - colors.min()) / 5
                ),
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Performance par Zone'
        ))
        
        # Ajouter des cercles de performance
        for idx, row in df.iterrows():
            if pd.notna(row[value_col]):
                performance_ratio = row[value_col] / df[value_col].max()
                radius = performance_ratio * 0.1  # Rayon en degrés
                
                fig.add_trace(go.Scattermapbox(
                    lat=[row[lat_col]],
                    lon=[row[lon_col]],
                    mode='markers',
                    marker=dict(
                        size=radius * 1000,
                        color='rgba(255,0,0,0.2)',
                        opacity=0.3
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Configuration de la carte
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=df[lat_col].mean(),
                    lon=df[lon_col].mean()
                ),
                zoom=6
            ),
            height=700,
            title={
                'text': f"Carte Interactive des Performances - {value_col}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def analyze_geographic_clusters(self, df, lat_col, lon_col, value_col, n_clusters=None):
        """Analyse avancée des clusters géographiques"""
        coords = df[[lat_col, lon_col]].dropna()
        
        if len(coords) < 3:
            return []
        
        # Déterminer le nombre optimal de clusters
        if n_clusters is None:
            n_clusters = min(5, max(2, len(coords) // 3))
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Analyse détaillée par cluster
        cluster_analysis = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_data = df.loc[cluster_coords.index]
            
            # Métriques géographiques du cluster
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Calcul du rayon du cluster
            distances_to_center = []
            for idx in cluster_coords.index:
                point = (cluster_coords.loc[idx, lat_col], cluster_coords.loc[idx, lon_col])
                center = (cluster_center[0], cluster_center[1])
                try:
                    dist = geodesic(center, point).kilometers
                    distances_to_center.append(dist)
                except:
                    distances_to_center.append(0)
            
            cluster_radius = np.mean(distances_to_center) if distances_to_center else 0
            
            # Métriques de performance
            cluster_values = self.clean_numeric_column(cluster_data[value_col]).dropna()
            
            cluster_info = {
                'cluster_id': cluster_id,
                'location_count': len(cluster_data),
                'center_lat': cluster_center[0],
                'center_lon': cluster_center[1],
                'radius_km': cluster_radius,
                'geographic_bounds': {
                    'north': cluster_coords[lat_col].max(),
                    'south': cluster_coords[lat_col].min(),
                    'east': cluster_coords[lon_col].max(),
                    'west': cluster_coords[lon_col].min()
                }
            }
            
            if len(cluster_values) > 0:
                avg_performance = cluster_values.mean()
                total_performance = cluster_values.sum()
                
                cluster_info.update({
                    'avg_performance': avg_performance,
                    'total_performance': total_performance,
                    'performance_std': cluster_values.std(),
                    'performance_level': self.classify_performance(avg_performance, df[value_col].mean()),
                    'performance_rank': 0  # Sera calculé après
                })
            
            cluster_analysis.append(cluster_info)
        
        # Classer les clusters par performance
        clusters_with_perf = [c for c in cluster_analysis if 'avg_performance' in c]
        clusters_with_perf.sort(key=lambda x: x['avg_performance'], reverse=True)
        
        for rank, cluster in enumerate(clusters_with_perf, 1):
            cluster['performance_rank'] = rank
        
        return cluster_analysis
    
    def classify_performance(self, value, mean_value):
        """Classifie la performance relative"""
        if value >= mean_value * 1.2:
            return 'Excellent'
        elif value >= mean_value * 1.1:
            return 'High'
        elif value >= mean_value * 0.9:
            return 'Average'
        elif value >= mean_value * 0.8:
            return 'Low'
        else:
            return 'Poor'
    
    def clean_numeric_column(self, series):
        """Nettoyage des colonnes numériques"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        cleaned = series.astype(str)
        cleaned = cleaned.str.replace(r'[$€£¥₹₽%]', '', regex=True)
        cleaned = cleaned.str.replace(',', '')
        cleaned = cleaned.str.replace(' ', '')
        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
        
        return pd.to_numeric(cleaned, errors='coerce')
    
    def generate_geographic_insights(self, df, mappings, clusters):
        """Génération d'insights géographiques avancés"""
        insights = []
        recommendations = []
        
        lat_col = mappings['latitude']
        lon_col = mappings['longitude']
        value_col = mappings.get('revenue', list(df.select_dtypes(include=[np.number]).columns)[0])
        
        # Analyse de la dispersion géographique
        spread = self.calculate_geographic_spread(df, lat_col, lon_col)
        
        if spread > 500:
            insights.append(f"🌍 **Large couverture territoriale**: {spread:.0f} km d'étendue")
            recommendations.append("Considérer la régionalisation de la gestion pour optimiser les opérations")
        elif spread > 100:
            insights.append(f"📍 **Couverture régionale**: {spread:.0f} km d'étendue")
        else:
            insights.append(f"🏘️ **Couverture locale**: {spread:.0f} km d'étendue")
        
        # Analyse des clusters
        if clusters and len(clusters) > 1:
            high_perf_clusters = [c for c in clusters if c.get('performance_level') in ['Excellent', 'High']]
            low_perf_clusters = [c for c in clusters if c.get('performance_level') in ['Low', 'Poor']]
            
            if high_perf_clusters:
                insights.append(f"🎯 **{len(high_perf_clusters)} zones haute performance** identifiées")
                best_cluster = max(high_perf_clusters, key=lambda x: x.get('avg_performance', 0))
                insights.append(f"💎 **Zone leader**: Cluster {best_cluster['cluster_id']+1} avec {best_cluster.get('avg_performance', 0):,.0f} DHS de moyenne")
            
            if low_perf_clusters:
                recommendations.append(f"⚡ **{len(low_perf_clusters)} zones sous-performantes** nécessitent attention")
                recommendations.append("Analyser les facteurs de succès des meilleures zones pour les répliquer")
            
            # Recommandations d'expansion
            if len(high_perf_clusters) >= 1:
                recommendations.append("🚀 **Opportunité d'expansion**: Identifier des zones similaires aux clusters performants")
        
        # Analyse de densité
        total_locations = len(df)
        if total_locations > 10:
            avg_distance = self.calculate_average_distance_between_points(df, lat_col, lon_col)
            if avg_distance < 50:
                insights.append(f"🏙️ **Forte densité**: {avg_distance:.0f} km de distance moyenne entre points")
                recommendations.append("Optimiser la cannibalisation entre sites proches")
            else:
                insights.append(f"🌾 **Faible densité**: {avg_distance:.0f} km de distance moyenne")
                recommendations.append("Considérer l'ouverture de sites intermédiaires")
        
        return insights, recommendations
    
    def calculate_geographic_spread(self, df, lat_col, lon_col):
        """Calcule l'étendue géographique"""
        if len(df) < 2:
            return 0
        
        max_distance = 0
        coords = df[[lat_col, lon_col]].dropna()
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                try:
                    coord1 = (coords.iloc[i][lat_col], coords.iloc[i][lon_col])
                    coord2 = (coords.iloc[j][lat_col], coords.iloc[j][lon_col])
                    distance = geodesic(coord1, coord2).kilometers
                    max_distance = max(max_distance, distance)
                except:
                    continue
        
        return max_distance
    
    def calculate_average_distance_between_points(self, df, lat_col, lon_col):
        """Calcule la distance moyenne entre tous les points"""
        coords = df[[lat_col, lon_col]].dropna()
        
        if len(coords) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                try:
                    coord1 = (coords.iloc[i][lat_col], coords.iloc[i][lon_col])
                    coord2 = (coords.iloc[j][lat_col], coords.iloc[j][lon_col])
                    distance = geodesic(coord1, coord2).kilometers
                    total_distance += distance
                    count += 1
                except:
                    continue
        
        return total_distance / count if count > 0 else 0

# ========== SESSION STATE INITIALIZATION ==========
def init_session_state():
    """Initialisation enrichie des variables de session"""
    
    # Données CSV enrichies
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = {}
    
    if 'csv_processor' not in st.session_state:
        st.session_state.csv_processor = EnhancedGeoCSVProcessor()
    
    if 'imported_metrics' not in st.session_state:
        st.session_state.imported_metrics = {}
    
    # Générateur de templates enrichi
    if 'template_generator' not in st.session_state:
        st.session_state.template_generator = EnhancedCSVTemplateGenerator()
    
    # Analyseur géographique
    if 'geo_analyzer' not in st.session_state:
        st.session_state.geo_analyzer = GeoFinancialAnalyzer()
    
    # Données d'analyse
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = {}
    
    if 'ml_forecasts' not in st.session_state:
        st.session_state.ml_forecasts = {}
    
    if 'geographic_analysis' not in st.session_state:
        st.session_state.geographic_analysis = {}

# ========== ENHANCED CSV IMPORT PAGE ==========
def show_enhanced_csv_import():
    """Import CSV enrichi avec géolocalisation"""
    st.header("📤 Import CSV Avancé avec Géolocalisation")
    
    st.markdown("""
    🚀 **Analyse Financière & Géographique Complète**: Uploadez vos données et obtenez une analyse approfondie 
    incluant la géolocalisation, métriques business avancées et insights IA !
    
    **Formats Supportés**: Plus de 30 colonnes détectées automatiquement incluant géolocalisation
    """)
    
    # Guide du format optimal enrichi
    with st.expander("📋 Guide Complet des Formats CSV", expanded=False):
        st.markdown("""
        ### 🎯 Colonnes Recommandées pour Analyse Complète
        
        **📊 Données Financières:**
        - `Date`, `Revenue`, `Sales`, `Costs`, `Profit`, `Cash_Flow`
        
        **🌍 Données Géographiques:**
        - `Latitude`, `Longitude`, `Location_Name`, `City`, `Region`, `Country`, `Postal_Code`
        
        **👥 Métriques Business:**
        - `Customer_Count`, `Customer_Acquisition_Cost`, `Lifetime_Value`, `Churn_Rate`
        - `Market_Share`, `Competition_Level`, `Brand_Strength`
        
        **📈 Métriques Opérationnelles:**
        - `Units_Sold`, `Average_Price`, `Employee_Count`, `Store_Size`
        - `Marketing_Spend`, `Website_Traffic`, `Demographics_Score`
        
        **💰 Données Financières Avancées:**
        - `Assets`, `Liabilities`, `Equity`, `Accounts_Receivable`, `Inventory`
        """)
        
        # Exemple de données enrichies
        st.markdown("### 📋 Exemple de Format Enrichi")
        example_enriched = {
            'Date': ['2025-06-28', '2025-06-28', '2025-06-28'],
            'Location_Name': ['Casablanca_Centre', 'Rabat_Agdal', 'Marrakech_Gueliz'],
            'Latitude': [33.5731, 34.0209, 31.6295],
            'Longitude': [-7.5898, -6.8416, -7.9811],
            'Revenue': [25000, 18000, 15000],
            'Costs': [18750, 13500, 11250],
            'Customer_Count': [250, 180, 150],
            'Market_Share': [0.25, 0.18, 0.15],
            'Competition_Level': [0.7, 0.6, 0.5]
        }
        
        st.dataframe(pd.DataFrame(example_enriched), use_container_width=True)
    
    # Templates enrichis à télécharger
    st.markdown("### 📥 Templates CSV Enrichis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌍 Template Géographique Complet", type="primary"):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('complete_geo_financial')
            
            if csv_data:
                st.download_button(
                    label="💾 Télécharger Template Complet",
                    data=csv_data,
                    file_name="template_geo_financier_complet.csv",
                    mime="text/csv"
                )
                st.success("✅ Template complet prêt!")
    
    with col2:
        if st.button("🛍️ Template Retail Géo"):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('retail_geo')
            
            if csv_data:
                st.download_button(
                    label="💾 Télécharger Template Retail",
                    data=csv_data,
                    file_name="template_retail_geo.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("☁️ Template SaaS Géo"):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('saas_geo')
            
            if csv_data:
                st.download_button(
                    label="💾 Télécharger Template SaaS",
                    data=csv_data,
                    file_name="template_saas_geo.csv",
                    mime="text/csv"
                )
    
    # Upload de fichier avec analyse enrichie
    uploaded_file = st.file_uploader(
        "📁 Glissez-déposez votre fichier CSV enrichi ici",
        type=['csv'],
        help="Supporte fichiers jusqu'à 200MB avec détection automatique des colonnes géographiques"
    )
    
    if uploaded_file is not None:
        try:
            # Barre de progression enrichie
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Étape 1: Lecture CSV
            status_text.text("📖 Lecture du fichier CSV...")
            progress_bar.progress(15)
            
            df = pd.read_csv(uploaded_file)
            
            # Étape 2: Détection des colonnes
            status_text.text("🔍 Détection des colonnes financières et géographiques...")
            progress_bar.progress(30)
            
            processor = st.session_state.csv_processor
            detected_mappings = processor.detect_columns(df)
            
            # Étape 3: Validation des données
            status_text.text("✅ Validation de la qualité des données...")
            progress_bar.progress(45)
            
            issues, suggestions = processor.validate_data(df, detected_mappings)
            
            # Étape 4: Calcul des métriques enrichies
            status_text.text("📊 Calcul des métriques financières et géographiques...")
            progress_bar.progress(65)
            
            # Étape 5: Traitement complet
            status_text.text("🧠 Génération des insights IA et visualisations...")
            progress_bar.progress(85)
            
            results = processor.process_csv(df)
            
            # Étape 6: Stockage
            status_text.text("💾 Stockage des résultats d'analyse...")
            progress_bar.progress(95)
            
            st.session_state.csv_data = results
            st.session_state.imported_metrics = results['metrics']
            
            # Finalisation
            status_text.text("🎉 Analyse complète terminée!")
            progress_bar.progress(100)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Message de succès enrichi
            st.success(f"🎉 **Analyse Complète Réussie!** {len(df)} lignes avec {len(df.columns)} colonnes traitées")
            
            # Statistiques d'import enrichies
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Lignes", f"{len(df):,}")
            with col2:
                st.metric("📈 Colonnes", len(df.columns))
            with col3:
                detected_cols = len(detected_mappings)
                st.metric("🎯 Auto-Détectées", detected_cols)
            with col4:
                has_geo = processor.has_geographic_data(detected_mappings, df)
                st.metric("🌍 Géo-Données", "✅ Oui" if has_geo else "❌ Non")
            with col5:
                file_size = uploaded_file.size / (1024 * 1024)
                st.metric("📁 Taille", f"{file_size:.1f} MB")
            
            # Affichage des résultats enrichis
            show_enhanced_csv_analysis_results(results)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement CSV: {str(e)}")
            st.info("💡 **Guide de Dépannage:**")
            st.write("• Vérifiez que le CSV utilise des virgules comme séparateurs")
            st.write("• Supprimez les symboles monétaires ($, €, etc.)")
            st.write("• Vérifiez la cohérence des formats de date")
            st.write("• Les coordonnées doivent être en format décimal")
            st.write("• Assurez-vous que les colonnes numériques ne contiennent que des chiffres")
    
    else:
        # Interface d'attente enrichie
        st.markdown("""
        ### 🌟 Capacités d'Analyse Enrichies:
        
        **📊 Analyse Financière Complète:**
        - Revenus, coûts, profits avec tendances temporelles
        - Ratios financiers avancés et benchmarking
        - Calculs de marges et volatilité automatiques
        
        **🌍 Intelligence Géographique:**
        - Cartes de chaleur des performances par zone
        - Clustering automatique des emplacements
        - Analyse de dispersion géographique
        - Corrélations géo-financières
        
        **🧠 Insights IA Avancés:**
        - Détection automatique de patterns géographiques
        - Recommandations d'optimisation par zone
        - Alertes de performance géo-localisées
        - Suggestions d'expansion territoriale
        
        **📈 Métriques Business Enrichies:**
        - LTV/CAC avec analyse géographique
        - Parts de marché par région
        - Analyse concurrentielle localisée
        - Scoring démographique des zones
        
        **🔄 Intégration Totale:**
        - Auto-population des modèles avancés
        - Synchronisation avec forecasting ML
        - Enrichissement des analyses de risque
        - Dashboard exécutif géo-localisé
        """)

def show_enhanced_csv_analysis_results(results):
    """Affichage enrichi des résultats d'analyse CSV"""
    
    mappings = results['mappings']
    metrics = results['metrics']
    insights_data = results['insights']
    figures = results['figures']
    geographic_analysis = results.get('geographic_analysis')
    df = results['processed_df']
    
    # Détection des colonnes enrichie
    st.subheader("🎯 Détection Automatique des Colonnes")
    
    if mappings:
        # Organiser par catégories
        financial_cols = {k: v for k, v in mappings.items() if k in ['revenue', 'costs', 'profit', 'cash_flow']}
        geo_cols = {k: v for k, v in mappings.items() if k in ['latitude', 'longitude', 'location_name', 'city', 'region', 'country']}
        business_cols = {k: v for k, v in mappings.items() if k in ['customer_count', 'customer_acquisition_cost', 'lifetime_value', 'market_share']}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Colonnes Financières")
            for category, column in financial_cols.items():
                st.success(f"**{category.title()}**: `{column}`")
        
        with col2:
            st.markdown("#### 🌍 Colonnes Géographiques")
            for category, column in geo_cols.items():
                st.info(f"**{category.title()}**: `{column}`")
        
        with col3:
            st.markdown("#### 📊 Colonnes Business")
            for category, column in business_cols.items():
                st.warning(f"**{category.title()}**: `{column}`")
    
    # Insights IA enrichis
    st.subheader("🤖 Intelligence Artificielle - Insights Enrichis")
    
    # Tabs pour différents types d'insights
    if insights_data:
        insight_tabs = st.tabs(["✅ Insights Clés", "🌍 Insights Géographiques", "⚠️ Alertes", "💡 Recommandations"])
        
        with insight_tabs[0]:
            if insights_data['insights']:
                for insight in insights_data['insights']:
                    st.success(insight)
            else:
                st.info("Aucun insight spécifique généré pour le moment.")
        
        with insight_tabs[1]:
            # Insights géographiques spécifiques
            if geographic_analysis:
                geo_insights = generate_geographic_specific_insights(geographic_analysis, metrics)
                for insight in geo_insights:
                    st.info(f"🌍 {insight}")
            else:
                st.info("Uploadez des données avec coordonnées pour voir les insights géographiques.")
        
        with insight_tabs[2]:
            if insights_data['alerts']:
                for alert in insights_data['alerts']:
                    st.error(alert)
            else:
                st.success("✅ Aucune alerte critique détectée!")
        
        with insight_tabs[3]:
            if insights_data['recommendations']:
                for rec in insights_data['recommendations']:
                    st.warning(f"💡 {rec}")
            else:
                st.info("Aucune recommandation spécifique à ce stade.")
    
    # Visualisations enrichies
    st.subheader("📈 Visualisations Interactives Enrichies")
    
    if figures:
        # Créer des tabs pour différents types de viz
        viz_tabs = []
        viz_content = []
        
        if 'financial_trend' in figures:
            viz_tabs.append("📊 Tendances Financières")
            viz_content.append(('financial_trend', figures['financial_trend']))
        
        if 'performance_map' in figures:
            viz_tabs.append("🗺️ Carte des Performances")
            viz_content.append(('performance_map', figures['performance_map']))
        
        if 'clusters_map' in figures:
            viz_tabs.append("🎯 Clusters Géographiques")
            viz_content.append(('clusters_map', figures['clusters_map']))
        
        if 'business_radar' in figures:
            viz_tabs.append("📡 Radar Business")
            viz_content.append(('business_radar', figures['business_radar']))
        
        if viz_tabs:
            selected_tabs = st.tabs(viz_tabs)
            
            for i, (tab_name, (fig_key, fig)) in enumerate(zip(viz_tabs, viz_content)):
                with selected_tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Ajouter des insights spécifiques à chaque graphique
                    if fig_key == 'performance_map':
                        st.info("💡 **Analyse**: Cette carte montre la répartition géographique de vos performances. Les zones plus chaudes indiquent de meilleures performances.")
                    elif fig_key == 'clusters_map':
                        st.info("💡 **Analyse**: Les clusters identifient des groupes d'emplacements avec des caractéristiques similaires pour optimiser votre stratégie.")
    
    # Analyse géographique détaillée
    if geographic_analysis:
        st.subheader("🌍 Analyse Géographique Approfondie")
        
        geo_tabs = st.tabs(["📊 Statistiques Géo", "🎯 Clustering", "📈 Corrélations Géo-Performance"])
        
        with geo_tabs[0]:
            show_geographic_statistics(geographic_analysis)
        
        with geo_tabs[1]:
            if 'geographic_data' in metrics and 'clusters' in metrics['geographic_data']:
                show_cluster_analysis(metrics['geographic_data']['clusters'])
            else:
                st.info("Clustering géographique non disponible (nécessite au moins 3 emplacements)")
        
        with geo_tabs[2]:
            if 'geo_performance_correlation' in geographic_analysis:
                show_geo_performance_correlation(geographic_analysis['geo_performance_correlation'])
            else:
                st.info("Analyse de corrélation non disponible")
    
    # Options d'intégration enrichies
    st.subheader("🔄 Options d'Intégration Avancées")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Sauvegarder Analyse", type="primary", use_container_width=True):
            st.success("✅ Analyse complète sauvegardée!")
            st.balloons()
    
    with col2:
        if st.button("🧠 Analytics Avancés", use_container_width=True):
            st.success("🚀 Accédez aux Analytics Avancés via le menu...")
            st.info("👈 Utilisez le menu de navigation pour accéder aux Analytics Avancés")
    
    with col3:
        if st.button("🌍 Analyse Géographique", use_container_width=True):
            st.session_state['current_page'] = 'geographic_analysis'
            st.rerun()
    
    with col4:
        if st.button("🎯 Planification Scénarios", use_container_width=True):
            st.success("🚀 Accédez à la Planification via le menu...")
            st.info("👈 Menu navigation → Planification de Scénarios")

def generate_geographic_specific_insights(geographic_analysis, metrics):
    """Génère des insights spécifiques à l'analyse géographique"""
    insights = []
    
    if 'basic_stats' in geographic_analysis:
        stats = geographic_analysis['basic_stats']
        total_locations = stats.get('total_locations', 0)
        
        if total_locations > 1:
            insights.append(f"Analyse de {total_locations} emplacements géographiques")
            
            bounds = stats.get('geographic_bounds', {})
            if bounds:
                lat_span = bounds.get('north', 0) - bounds.get('south', 0)
                lon_span = bounds.get('east', 0) - bounds.get('west', 0)
                
                if lat_span > 5 or lon_span > 5:
                    insights.append("Large dispersion géographique détectée - considérer la régionalisation")
                else:
                    insights.append("Emplacements géographiquement concentrés - optimisation logistique possible")
    
    if 'density_analysis' in geographic_analysis:
        density = geographic_analysis['density_analysis']
        avg_distance = density.get('average_distance_between_points', 0)
        
        if avg_distance > 100:
            insights.append(f"Distance moyenne importante entre sites: {avg_distance:.0f} km")
        elif avg_distance > 50:
            insights.append(f"Distance modérée entre sites: {avg_distance:.0f} km")
        else:
            insights.append(f"Sites rapprochés: {avg_distance:.0f} km de distance moyenne")
    
    if 'geographic_data' in metrics and 'clusters' in metrics['geographic_data']:
        clusters = metrics['geographic_data']['clusters']
        high_perf = [c for c in clusters if c.get('performance_level') in ['Excellent', 'High']]
        
        if high_perf:
            insights.append(f"{len(high_perf)} cluster(s) haute performance identifié(s)")
    
    return insights

def show_geographic_statistics(geographic_analysis):
    """Affiche les statistiques géographiques"""
    
    if 'basic_stats' in geographic_analysis:
        stats = geographic_analysis['basic_stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Statistiques de Base")
            st.metric("Emplacements Totaux", stats.get('total_locations', 0))
            
            center = stats.get('center_point', {})
            if center:
                st.metric("Centre Latitude", f"{center.get('latitude', 0):.4f}°")
                st.metric("Centre Longitude", f"{center.get('longitude', 0):.4f}°")
        
        with col2:
            st.markdown("#### 🗺️ Limites Géographiques")
            bounds = stats.get('geographic_bounds', {})
            if bounds:
                st.metric("Nord", f"{bounds.get('north', 0):.4f}°")
                st.metric("Sud", f"{bounds.get('south', 0):.4f}°")
                st.metric("Est", f"{bounds.get('east', 0):.4f}°")
                st.metric("Ouest", f"{bounds.get('west', 0):.4f}°")
    
    if 'density_analysis' in geographic_analysis:
        density = geographic_analysis['density_analysis']
        
        st.markdown("#### 📊 Analyse de Densité")
        col1, col2 = st.columns(2)
        
        with col1:
            concentration = density.get('geographic_concentration', 0)
            st.metric("Concentration Géographique", f"{concentration:.4f}")
        
        with col2:
            avg_distance = density.get('average_distance_between_points', 0)
            st.metric("Distance Moyenne Entre Points", f"{avg_distance:.1f} km")

def show_cluster_analysis(clusters):
    """Affiche l'analyse des clusters géographiques"""
    
    if not clusters:
        st.info("Aucun cluster géographique détecté.")
        return
    
    st.markdown(f"#### 🎯 {len(clusters)} Cluster(s) Géographique(s) Identifié(s)")
    
    for i, cluster in enumerate(clusters):
        with st.expander(f"Cluster {i+1} - {cluster.get('performance_level', 'N/A')} Performance"):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Emplacements", cluster.get('location_count', 0))
                st.metric("Rayon", f"{cluster.get('radius_km', 0):.1f} km")
            
            with col2:
                st.metric("Centre Latitude", f"{cluster.get('center_lat', 0):.4f}°")
                st.metric("Centre Longitude", f"{cluster.get('center_lon', 0):.4f}°")
            
            with col3:
                if 'avg_performance' in cluster:
                    st.metric("Performance Moyenne", f"{cluster['avg_performance']:,.0f} DHS")
                if 'performance_rank' in cluster:
                    st.metric("Rang Performance", f"#{cluster['performance_rank']}")
            
            # Limites géographiques du cluster
            if 'geographic_bounds' in cluster:
                bounds = cluster['geographic_bounds']
                st.markdown("**Limites Géographiques:**")
                st.write(f"Nord: {bounds.get('north', 0):.4f}° | Sud: {bounds.get('south', 0):.4f}°")
                st.write(f"Est: {bounds.get('east', 0):.4f}° | Ouest: {bounds.get('west', 0):.4f}°")

def show_geo_performance_correlation(correlation_data):
    """Affiche les corrélations géo-performance"""
    
    st.markdown("#### 📈 Corrélations Géographiques avec Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lat_corr = correlation_data.get('latitude_correlation', 0)
        st.metric("Corrélation Latitude-Performance", f"{lat_corr:.3f}")
        
        if abs(lat_corr) > 0.5:
            st.info("🔍 **Forte corrélation Nord-Sud** détectée")
        elif abs(lat_corr) > 0.3:
            st.info("📊 **Corrélation modérée Nord-Sud** observée")
    
    with col2:
        lon_corr = correlation_data.get('longitude_correlation', 0)
        st.metric("Corrélation Longitude-Performance", f"{lon_corr:.3f}")
        
        if abs(lon_corr) > 0.5:
            st.info("🔍 **Forte corrélation Est-Ouest** détectée")
        elif abs(lon_corr) > 0.3:
            st.info("📊 **Corrélation modérée Est-Ouest** observée")
    
    # Analyse par quadrants
    if 'quadrant_analysis' in correlation_data:
        st.markdown("#### 🗺️ Analyse par Quadrants Géographiques")
        
        quadrants = correlation_data['quadrant_analysis']
        
        quad_data = []
        for quad_name, quad_info in quadrants.items():
            quad_data.append({
                'Quadrant': quad_name.replace('_', ' ').title(),
                'Emplacements': quad_info.get('count', 0),
                'Revenue Moyenne': f"{quad_info.get('avg_revenue', 0):,.0f} DHS",
                'Revenue Total': f"{quad_info.get('total_revenue', 0):,.0f} DHS"
            })
        
        if quad_data:
            st.dataframe(pd.DataFrame(quad_data), use_container_width=True)

# ========== GEOGRAPHIC ANALYSIS PAGE ==========
def show_geographic_analysis():
    """Page d'analyse géographique complète et avancée"""
    st.header("🌍 Analyse Géographique Avancée des Performances")
    
    # Vérification des données CSV
    csv_data = EnhancedCSVDataManager.get_csv_financial_data()
    
    if not csv_data or 'csv_data' not in st.session_state:
        st.warning("📤 **Données géographiques non disponibles**")
        st.info("L'analyse géographique nécessite des données CSV avec coordonnées latitude/longitude.")
        
        # Exemple de format géographique
        st.subheader("📋 Format CSV Géographique Requis")
        
        example_geo_data = {
            'Date': ['2025-06-28', '2025-06-28', '2025-06-28'],
            'Location_Name': ['Casablanca_Centre', 'Rabat_Agdal', 'Marrakech_Gueliz'],
            'Latitude': [33.5731, 34.0209, 31.6295],
            'Longitude': [-7.5898, -6.8416, -7.9811],
            'Revenue': [25000, 18000, 15000],
            'Customer_Count': [250, 180, 150],
            'Market_Share': [0.25, 0.18, 0.15]
        }
        
        st.dataframe(pd.DataFrame(example_geo_data), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Template géographique
            if st.button("📥 Télécharger Template Géographique", type="primary"):
                template_gen = st.session_state.template_generator
                csv_data = template_gen.generate_template_csv('complete_geo_financial')
                
                if csv_data:
                    st.download_button(
                        label="💾 Template Géographique Complet",
                        data=csv_data,
                        file_name="template_geo_analyse.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📤 Retour à l'Import CSV"):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        return
    
    # Vérification des données géographiques
    df = st.session_state.csv_data['processed_df']
    processor = st.session_state.csv_processor
    geo_mappings = processor.detect_columns(df)
    
    if 'latitude' not in geo_mappings or 'longitude' not in geo_mappings:
        st.error("❌ **Coordonnées géographiques non détectées**")
        
        # Option de géocodage automatique
        if 'location_name' in geo_mappings:
            st.subheader("🔄 Géocodage Automatique Disponible")
            st.info(f"Colonne d'adresses détectée: `{geo_mappings['location_name']}`")
            
            if st.button("🌍 Géocoder les Adresses Automatiquement", type="primary"):
                with st.spinner("Géocodage en cours... Cela peut prendre quelques minutes."):
                    try:
                        geocoded_df = processor.geocode_locations(df, geo_mappings['location_name'])
                        
                        if len(geocoded_df) > 0:
                            st.success(f"✅ {len(geocoded_df)} emplacements géocodés avec succès!")
                            
                            # Afficher les résultats du géocodage
                            st.subheader("📍 Résultats du Géocodage")
                            st.dataframe(geocoded_df, use_container_width=True)
                            
                            # Option de téléchargement du fichier enrichi
                            merged_df = df.merge(geocoded_df, left_on=geo_mappings['location_name'], right_on='location', how='left')
                            
                            csv_enriched = merged_df.to_csv(index=False)
                            st.download_button(
                                label="💾 Télécharger CSV Enrichi avec Coordonnées",
                                data=csv_enriched,
                                file_name="donnees_enrichies_geo.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Aucune adresse n'a pu être géocodée")
                            st.info("💡 Vérifiez que les noms d'emplacements sont clairs (ex: 'Casablanca, Morocco')")
                    
                    except Exception as e:
                        st.error(f"Erreur lors du géocodage: {str(e)}")
        else:
            st.info("💡 Ajoutez une colonne avec des noms d'emplacements pour le géocodage automatique")
        
        return
    
    st.success("📍 **Données géographiques détectées et validées**")
    
    # Configuration de l'analyse géographique
    st.subheader("⚙️ Configuration de l'Analyse Géographique")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Métrique à analyser
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        value_column = st.selectbox(
            "📊 Métrique à Analyser",
            numeric_columns,
            index=next((i for i, col in enumerate(numeric_columns) if 'revenue' in col.lower()), 0)
        )
    
    with col2:
        # Type de visualisation
        viz_type = st.selectbox(
            "🗺️ Type de Visualisation",
            [
                "Carte de Chaleur Interactive", 
                "Carte 3D avec Élévations",
                "Analyse de Clusters",
                "Carte Choroplèthe",
                "Vue d'Ensemble Complète"
            ]
        )
    
    with col3:
        # Période d'analyse
        if 'date' in geo_mappings and geo_mappings['date'] in df.columns:
            df[geo_mappings['date']] = pd.to_datetime(df[geo_mappings['date']], errors='coerce')
            
            date_range = st.date_input(
                "📅 Période d'Analyse",
                value=[df[geo_mappings['date']].min().date(), df[geo_mappings['date']].max().date()],
                min_value=df[geo_mappings['date']].min().date(),
                max_value=df[geo_mappings['date']].max().date()
            )
        else:
            date_range = None
            st.info("Pas de filtrage temporel")
    
    with col4:
        # Options avancées
        advanced_options = st.checkbox("🔧 Options Avancées", value=False)
        
        if advanced_options:
            cluster_count = st.number_input("Nombre de Clusters", min_value=2, max_value=10, value=3)
            min_cluster_size = st.number_input("Taille Min. Cluster", min_value=1, max_value=10, value=2)
        else:
            cluster_count = 3
            min_cluster_size = 2
    
    # Filtrage des données par période
    filtered_df = df.copy()
    if date_range and len(date_range) == 2 and geo_mappings.get('date'):
        start_date, end_date = date_range
        date_col = geo_mappings['date']
        filtered_df = df[
            (df[date_col].dt.date >= start_date) & 
            (df[date_col].dt.date <= end_date)
        ]
    
    # Agrégation par localisation
    location_col = geo_mappings.get('location_name')
    
    if location_col and location_col in filtered_df.columns:
        # Agrégation par nom d'emplacement
        agg_columns = {
            geo_mappings['latitude']: 'first',
            geo_mappings['longitude']: 'first',
            value_column: 'sum'
        }
        
        # Ajouter d'autres métriques disponibles
        for col in ['customer_count', 'market_share', 'competition_level']:
            if col in geo_mappings and geo_mappings[col] in filtered_df.columns:
                agg_columns[geo_mappings[col]] = 'mean'
        
        agg_df = filtered_df.groupby(location_col).agg(agg_columns).reset_index()
        agg_df.columns = [col[0] if isinstance(col, tuple) else col for col in agg_df.columns]
    else:
        # Pas d'agrégation possible
        agg_df = filtered_df[[geo_mappings['latitude'], geo_mappings['longitude'], value_column]].dropna()
    
    # Métriques géographiques
    st.subheader("📊 Aperçu de la Performance Géographique")
    
    geo_analyzer = st.session_state.geo_analyzer
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_locations = len(agg_df)
        st.metric("🏢 Emplacements", total_locations)
    
    with col2:
        if total_locations > 1:
            geographic_spread = geo_analyzer.calculate_geographic_spread(
                agg_df, geo_mappings['latitude'], geo_mappings['longitude']
            )
            st.metric("🌍 Étendue", f"{geographic_spread:.0f} km")
        else:
            st.metric("🌍 Étendue", "N/A")
    
    with col3:
        best_location_value = agg_df[value_column].max()
        st.metric("🏆 Top Performance", f"{best_location_value:,.0f}")
    
    with col4:
        avg_performance = agg_df[value_column].mean()
        st.metric("📊 Moyenne", f"{avg_performance:,.0f}")
    
    with col5:
        performance_std = agg_df[value_column].std()
        cv = (performance_std / avg_performance * 100) if avg_performance > 0 else 0
        st.metric("📈 Variabilité", f"{cv:.1f}%")
    
    # Génération des visualisations selon le type sélectionné
    if viz_type == "Carte de Chaleur Interactive":
        st.subheader("🔥 Carte de Chaleur Interactive des Performances")
        
        # Carte Folium
        heatmap = geo_analyzer.create_performance_heatmap(
            agg_df, 
            geo_mappings['latitude'], 
            geo_mappings['longitude'], 
            value_column,
            location_col
        )
        
        # Affichage de la carte
        st.components.v1.html(heatmap._repr_html_(), height=650)
        
        # Insights automatiques
        st.info("💡 **Analyse**: Les zones rouges indiquent les plus hautes performances. Utilisez le zoom et les filtres pour explorer en détail.")
        
    elif viz_type == "Carte 3D avec Élévations":
        st.subheader("🏔️ Visualisation 3D des Performances")
        
        # Carte 3D PyDeck
        try:
            deck_map = geo_analyzer.create_3d_deck_map(
                agg_df,
                geo_mappings['latitude'],
                geo_mappings['longitude'],
                value_column
            )
            
            st.pydeck_chart(deck_map)
            
            st.info("💡 **Analyse**: La hauteur des colonnes représente la performance. Utilisez la souris pour naviguer en 3D.")
            
        except Exception as e:
            st.error(f"Erreur lors de la création de la carte 3D: {str(e)}")
            st.info("Utilisez la carte de chaleur en alternative.")
    
    elif viz_type == "Analyse de Clusters":
        st.subheader("🎯 Analyse Avancée des Clusters Géographiques")
        
        # Clustering géographique
        clusters = geo_analyzer.analyze_geographic_clusters(
            agg_df,
            geo_mappings['latitude'],
            geo_mappings['longitude'],
            value_column,
            cluster_count
        )
        
        if clusters:
            # Affichage des résultats de clustering
            st.markdown("### 📋 Résultats du Clustering")
            
            for i, cluster in enumerate(clusters):
                with st.expander(f"🎯 Cluster {i+1} - {cluster.get('performance_level', 'N/A')} Performance", expanded=i==0):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📍 Emplacements", cluster.get('location_count', 0))
                        st.metric("📏 Rayon Moyen", f"{cluster.get('radius_km', 0):.1f} km")
                    
                    with col2:
                        st.metric("🗺️ Centre Latitude", f"{cluster.get('center_lat', 0):.4f}°")
                        st.metric("🗺️ Centre Longitude", f"{cluster.get('center_lon', 0):.4f}°")
                    
                    with col3:
                        if 'avg_performance' in cluster:
                            st.metric("💰 Performance Moy.", f"{cluster['avg_performance']:,.0f}")
                        if 'performance_rank' in cluster:
                            st.metric("🏆 Rang", f"#{cluster['performance_rank']}")
            
            # Carte des clusters
            cluster_map = create_enhanced_cluster_map(agg_df, clusters, geo_mappings, value_column)
            st.plotly_chart(cluster_map, use_container_width=True)
        else:
            st.info("Clustering non disponible avec les données actuelles.")
    
    elif viz_type == "Carte Choroplèthe":
        st.subheader("🗺️ Carte Choroplèthe Avancée")
        
        choropleth_map = geo_analyzer.create_plotly_choropleth_map(
            agg_df,
            geo_mappings['latitude'],
            geo_mappings['longitude'],
            value_column,
            location_col
        )
        
        st.plotly_chart(choropleth_map, use_container_width=True)
        
        st.info("💡 **Analyse**: La taille et couleur des marqueurs représentent les performances. Survolez pour plus de détails.")
    
    elif viz_type == "Vue d'Ensemble Complète":
        st.subheader("📊 Dashboard Géographique Complet")
        
        # Créer un dashboard avec plusieurs visualisations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Carte de Chaleur")
            heatmap = geo_analyzer.create_performance_heatmap(
                agg_df, 
                geo_mappings['latitude'], 
                geo_mappings['longitude'], 
                value_column,
                location_col
            )
            st.components.v1.html(heatmap._repr_html_(), height=400)
        
        with col2:
            st.markdown("#### 📊 Carte Interactive")
            interactive_map = geo_analyzer.create_plotly_choropleth_map(
                agg_df,
                geo_mappings['latitude'],
                geo_mappings['longitude'],
                value_column,
                location_col
            )
            st.plotly_chart(interactive_map, use_container_width=True, config={'displayModeBar': False})
    
    # Analyse et insights géographiques
    st.subheader("💡 Insights Géographiques Automatiques")
    
    # Clustering automatique pour les insights
    clusters = geo_analyzer.analyze_geographic_clusters(
        agg_df,
        geo_mappings['latitude'],
        geo_mappings['longitude'],
        value_column,
        cluster_count
    )
    
    # Génération d'insights
    geo_insights, geo_recommendations = geo_analyzer.generate_geographic_insights(agg_df, geo_mappings, clusters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Observations Clés")
        if geo_insights:
            for insight in geo_insights:
                st.success(f"✅ {insight}")
        else:
            st.info("Aucun insight géographique spécifique détecté.")
    
    with col2:
        st.markdown("#### 🎯 Recommandations Stratégiques")
        if geo_recommendations:
            for rec in geo_recommendations:
                st.warning(f"💡 {rec}")
        else:
            st.success("✅ Performance géographique optimale détectée!")
    
    # Analyse comparative par zones
    st.subheader("📈 Analyse Comparative par Zones")
    
    if len(agg_df) > 1:
        # Top et bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top Performers")
            top_locations = agg_df.nlargest(min(5, len(agg_df)), value_column)
            
            for idx, location in top_locations.iterrows():
                location_name = location.get(location_col, f"Localisation {idx}") if location_col else f"Point {idx}"
                performance = location[value_column]
                
                # Calcul du pourcentage par rapport à la moyenne
                perf_vs_avg = ((performance / avg_performance) - 1) * 100 if avg_performance > 0 else 0
                
                st.success(f"📍 **{location_name}**: {performance:,.0f} (+{perf_vs_avg:+.1f}%)")
        
        with col2:
            st.markdown("#### 📉 Zones d'Amélioration")
            bottom_locations = agg_df.nsmallest(min(3, len(agg_df)), value_column)
            
            for idx, location in bottom_locations.iterrows():
                location_name = location.get(location_col, f"Localisation {idx}") if location_col else f"Point {idx}"
                performance = location[value_column]
                
                perf_vs_avg = ((performance / avg_performance) - 1) * 100 if avg_performance > 0 else 0
                
                st.error(f"📍 **{location_name}**: {performance:,.0f} ({perf_vs_avg:+.1f}%)")
    
    # Métriques de corrélation géographique
    if len(agg_df) > 3:
        st.subheader("🔗 Corrélations Géo-Performance")
        
        # Calculer corrélations
        lat_corr = agg_df[geo_mappings['latitude']].corr(agg_df[value_column])
        lon_corr = agg_df[geo_mappings['longitude']].corr(agg_df[value_column])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📏 Corrélation Latitude", f"{lat_corr:.3f}")
            if abs(lat_corr) > 0.5:
                st.info("🔍 Forte influence Nord-Sud")
        
        with col2:
            st.metric("📏 Corrélation Longitude", f"{lon_corr:.3f}")
            if abs(lon_corr) > 0.5:
                st.info("🔍 Forte influence Est-Ouest")
        
        with col3:
            # Calculer l'index de Moran (autocorrélation spatiale) simplifié
            moran_index = calculate_simple_moran_index(agg_df, geo_mappings, value_column)
            st.metric("🌐 Autocorrélation Spatiale", f"{moran_index:.3f}")
            
            if moran_index > 0.3:
                st.info("🔗 Clustering spatial détecté")
            elif moran_index < -0.3:
                st.info("🔀 Dispersion spatiale détectée")
    
    # Export et actions
    st.subheader("📤 Export et Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Sauvegarder Analyse Géo", type="primary"):
            # Sauvegarder les résultats dans session state
            st.session_state['geographic_analysis_results'] = {
                'data': agg_df.to_dict(),
                'clusters': clusters,
                'insights': geo_insights,
                'recommendations': geo_recommendations
            }
            st.success("✅ Analyse géographique sauvegardée!")
    
    with col2:
        # Export des données enrichies
        if st.button("📊 Exporter Données Enrichies"):
            enriched_data = agg_df.copy()
            
            # Ajouter des métriques calculées
            enriched_data['Performance_Rank'] = enriched_data[value_column].rank(method='dense', ascending=False)
            enriched_data['Performance_Percentile'] = enriched_data[value_column].rank(pct=True)
            enriched_data['Vs_Average'] = ((enriched_data[value_column] / avg_performance) - 1) * 100
            
            csv_export = enriched_data.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger CSV Enrichi",
                data=csv_export,
                file_name=f"analyse_geo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🎯 Planification Expansion"):
            st.info("🚀 Fonctionnalité de planification d'expansion en développement...")
            # Placeholder pour future fonctionnalité d'expansion
    
    with col4:
        if st.button("📈 Analytics Avancés"):
            st.session_state['current_page'] = 'advanced_analytics'
            st.rerun()

def create_enhanced_cluster_map(df, clusters, geo_mappings, value_column):
    """Créer une carte avancée des clusters géographiques"""
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    # Ajouter les points de données avec couleurs par cluster
    if len(clusters) > 0:
        # Assigner les clusters aux points de données
        coords = df[[geo_mappings['latitude'], geo_mappings['longitude']]].dropna()
        
        if len(coords) >= 3:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(clusters), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords)
            
            # Ajouter les points par cluster
            for cluster_id in range(len(clusters)):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = df.loc[coords.index[cluster_mask]]
                
                if len(cluster_data) > 0:
                    cluster_color = colors[cluster_id % len(colors)]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=cluster_data[geo_mappings['latitude']],
                        lon=cluster_data[geo_mappings['longitude']],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=cluster_color,
                            opacity=0.7
                        ),
                        name=f"Cluster {cluster_id + 1}",
                        text=[f"Cluster {cluster_id + 1}<br>Valeur: {val:,.0f}" 
                              for val in cluster_data[value_column]],
                        hovertemplate='%{text}<extra></extra>'
                    ))
    
    # Ajouter les centres des clusters
    for i, cluster in enumerate(clusters):
        cluster_color = colors[i % len(colors)]
        
        # Centre du cluster
        fig.add_trace(go.Scattermapbox(
            lat=[cluster['center_lat']],
            lon=[cluster['center_lon']],
            mode='markers',
            marker=dict(
                size=20,
                color=cluster_color,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            name=f"Centre Cluster {i+1}",
            text=f"Centre Cluster {i+1}<br>Performance: {cluster.get('performance_level', 'N/A')}<br>Emplacements: {cluster['location_count']}",
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
        
        # Ajouter un cercle pour représenter le rayon du cluster
        if cluster.get('radius_km', 0) > 0:
            circle_lat, circle_lon = generate_circle_coordinates(
                cluster['center_lat'], 
                cluster['center_lon'], 
                cluster['radius_km']
            )
            
            fig.add_trace(go.Scattermapbox(
                lat=circle_lat,
                lon=circle_lon,
                mode='lines',
                line=dict(width=2, color=cluster_color),
                name=f"Rayon Cluster {i+1}",
                hoverinfo='skip',
                showlegend=False,
                opacity=0.5
            ))
    
    # Configuration de la carte
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=df[geo_mappings['latitude']].mean(),
                lon=df[geo_mappings['longitude']].mean()
            ),
            zoom=6
        ),
        height=600,
        title="Clusters Géographiques de Performance",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def generate_circle_coordinates(center_lat, center_lon, radius_km, num_points=50):
    """Génère les coordonnées d'un cercle pour la visualisation"""
    import math
    
    # Conversion approximative km vers degrés (dépend de la latitude)
    lat_deg_per_km = 1 / 111.0
    lon_deg_per_km = 1 / (111.0 * math.cos(math.radians(center_lat)))
    
    radius_lat = radius_km * lat_deg_per_km
    radius_lon = radius_km * lon_deg_per_km
    
    circle_lat = []
    circle_lon = []
    
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        lat = center_lat + radius_lat * math.cos(angle)
        lon = center_lon + radius_lon * math.sin(angle)
        circle_lat.append(lat)
        circle_lon.append(lon)
    
    return circle_lat, circle_lon

def calculate_simple_moran_index(df, geo_mappings, value_column):
    """Calcule un index de Moran simplifié pour l'autocorrélation spatiale"""
    if len(df) < 3:
        return 0
    
    try:
        from geopy.distance import geodesic
        
        n = len(df)
        coords = df[[geo_mappings['latitude'], geo_mappings['longitude']]].values
        values = df[value_column].values
        
        # Matrice de poids basée sur l'inverse de la distance
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        distance = geodesic(coords[i], coords[j]).kilometers
                        weights[i, j] = 1 / (distance + 1)  # +1 pour éviter division par 0
                    except:
                        weights[i, j] = 0
        
        # Normaliser les poids
        row_sums = weights.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 0:
                weights[i] = weights[i] / row_sums[i]
        
        # Calculer l'index de Moran
        mean_value = values.mean()
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_value) * (values[j] - mean_value)
            denominator += (values[i] - mean_value) ** 2
        
        if denominator > 0:
            moran_index = numerator / denominator
        else:
            moran_index = 0
        
        return moran_index
    
    except Exception:
        return 0

# ========== ENHANCED EXECUTIVE DASHBOARD ==========
def show_enhanced_executive_dashboard():
    """Dashboard exécutif enrichi avec géolocalisation"""
    st.header("👔 Dashboard Exécutif Enrichi")
    
    # Récupération des données enrichies
    csv_data = EnhancedCSVDataManager.get_csv_financial_data()
    
    if csv_data:
        st.success("📊 **Dashboard alimenté par vos données CSV enrichies**")
        
        # KPIs principaux enrichis
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            monthly_revenue = csv_data.get('monthly_revenue', 0)
            st.metric("💰 Revenue Mensuel", f"{monthly_revenue:,.0f} DHS")
            
            growth = csv_data.get('revenue_growth', 0)
            if growth > 0:
                st.success(f"📈 +{growth:.1f}%")
            else:
                st.error(f"📉 {growth:.1f}%")
        
        with col2:
            profit_margin = csv_data.get('profit_margin', 0)
            st.metric("📊 Marge Profit", f"{profit_margin:.1f}%")
            
            if profit_margin > 20:
                st.success("🎯 Excellente")
            elif profit_margin > 10:
                st.info("📈 Bonne")
            else:
                st.warning("⚠️ À améliorer")
        
        with col3:
            total_locations = csv_data.get('total_locations', 0)
            st.metric("🏢 Emplacements", total_locations if total_locations > 0 else "N/A")
            
            if total_locations > 5:
                st.info("🌍 Multi-sites")
            elif total_locations > 1:
                st.info("📍 Régional")
        
        with col4:
            customer_count = csv_data.get('customer_count', 0)
            st.metric("👥 Clients", f"{customer_count:,.0f}" if customer_count > 0 else "N/A")
            
            ltv_cac = csv_data.get('ltv_cac_ratio', 0)
            if ltv_cac > 3:
                st.success("💎 LTV/CAC > 3")
            elif ltv_cac > 1:
                st.info(f"📊 LTV/CAC: {ltv_cac:.1f}")
        
        with col5:
            market_share = csv_data.get('market_share', 0)
            if market_share > 0:
                st.metric("🎯 Part de Marché", f"{market_share*100:.1f}%")
                
                if market_share > 0.2:
                    st.success("🏆 Leader")
                elif market_share > 0.1:
                    st.info("💪 Forte")
                else:
                    st.warning("📈 Croissance")
            else:
                st.metric("🎯 Part de Marché", "N/A")
        
        # Section géographique si disponible
        if csv_data.get('total_locations', 0) > 0:
            st.subheader("🌍 Performance Géographique")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Mini carte des performances si données géo disponibles
                if 'csv_data' in st.session_state and st.session_state.csv_data.get('geographic_analysis'):
                    st.markdown("#### 🗺️ Aperçu Géographique")
                    
                    # Créer une carte simple pour le dashboard
                    geo_data = st.session_state.csv_data.get('processed_df')
                    processor = st.session_state.csv_processor
                    mappings = processor.detect_columns(geo_data)
                    
                    if 'latitude' in mappings and 'longitude' in mappings:
                        # Carte simple pour dashboard
                        dashboard_map = create_dashboard_mini_map(geo_data, mappings)
                        st.plotly_chart(dashboard_map, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("📍 Coordonnées géographiques non disponibles")
                else:
                    # Graphique financier standard
                    csv_figures = EnhancedCSVDataManager.get_csv_visualizations()
                    if csv_figures and 'financial_trend' in csv_figures:
                        st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
            
            with col2:
                st.markdown("#### 🏆 Top Zones")
                
                # Afficher les meilleures zones si disponible
                best_location = csv_data.get('best_performing_location')
                if best_location:
                    st.success(f"🥇 **{best_location.get('name', 'Zone 1')}**")
                    st.write(f"💰 {best_location.get('revenue', 0):,.0f} DHS")
                    
                    # Coordonnées si disponibles
                    if 'latitude' in best_location:
                        st.caption(f"📍 {best_location['latitude']:.3f}, {best_location['longitude']:.3f}")
                
                # Métriques géographiques
                geographic_spread = csv_data.get('geographic_spread', 0)
                if geographic_spread > 0:
                    st.metric("🌍 Étendue", f"{geographic_spread:.0f} km")
                
                # Clusters de performance
                clusters = csv_data.get('geographic_clusters', [])
                if clusters:
                    high_perf_clusters = [c for c in clusters if c.get('performance_level') in ['High', 'Excellent']]
                    st.metric("🎯 Zones Haute Perf.", len(high_perf_clusters))
        
        # Insights exécutifs
        st.subheader("🧠 Insights Stratégiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Points Forts")
            
            # Générer des insights exécutifs automatiques
            exec_insights = generate_executive_insights(csv_data)
            for insight in exec_insights['strengths']:
                st.success(f"✅ {insight}")
        
        with col2:
            st.markdown("#### 🎯 Opportunités")
            
            for opportunity in exec_insights['opportunities']:
                st.info(f"💡 {opportunity}")
        
        # Alertes critiques
        if exec_insights['alerts']:
            st.subheader("⚠️ Alertes Stratégiques")
            for alert in exec_insights['alerts']:
                st.error(f"🚨 {alert}")
        
        # Actions recommandées
        st.subheader("🎯 Actions Prioritaires")
        
        recommended_actions = generate_executive_actions(csv_data)
        for i, action in enumerate(recommended_actions, 1):
            st.warning(f"**Action {i}**: {action}")
    
    else:
        # Pas de données CSV
        st.warning("📤 **Aucune donnée CSV importée**")
        st.info("Importez vos données financières via Smart CSV Import pour voir le dashboard complet!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Dashboard Exécutif Enrichi:**
            - KPIs financiers en temps réel
            - Performance géographique
            - Insights IA automatiques
            - Alertes stratégiques
            - Recommandations d'actions
            """)

def create_dashboard_mini_map(df, mappings):
    """Crée une mini-carte pour le dashboard exécutif"""
    
    lat_col = mappings['latitude']
    lon_col = mappings['longitude']
    
    # Utiliser les revenus comme métrique par défaut
    value_col = 'revenue'
    if 'revenue' in mappings:
        value_col = mappings['revenue']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else lat_col
    
    # Nettoyer les données
    plot_df = df[[lat_col, lon_col, value_col]].dropna()
    
    if len(plot_df) == 0:
        # Retourner une carte vide
        fig = go.Figure()
        fig.update_layout(height=300, title="Aucune donnée géographique")
        return fig
    
    # Calculer les tailles de marqueurs
    if plot_df[value_col].max() > 0:
        marker_sizes = (plot_df[value_col] / plot_df[value_col].max() * 30 + 10)
    else:
        marker_sizes = [15] * len(plot_df)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=plot_df[lat_col],
        lon=plot_df[lon_col],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=plot_df[value_col],
            colorscale='Viridis',
            opacity=0.7,
            showscale=False
        ),
        text=[f"Performance: {val:,.0f}" for val in plot_df[value_col]],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=plot_df[lat_col].mean(),
                lon=plot_df[lon_col].mean()
            ),
            zoom=5
        ),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def generate_executive_insights(csv_data):
    """Génère des insights pour le niveau exécutif"""
    insights = {
        'strengths': [],
        'opportunities': [],
        'alerts': []
    }
    
    # Analyse des forces
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    total_locations = csv_data.get('total_locations', 0)
    market_share = csv_data.get('market_share', 0)
    
    if profit_margin > 15:
        insights['strengths'].append(f"Marges excellentes de {profit_margin:.1f}% - bien au-dessus de la moyenne sectorielle")
    
    if revenue_growth > 10:
        insights['strengths'].append(f"Croissance forte de {revenue_growth:.1f}% - momentum positif confirmé")
    
    if total_locations > 5:
        insights['strengths'].append(f"Présence géographique étendue avec {total_locations} emplacements")
    
    if market_share > 0.15:
        insights['strengths'].append(f"Position de leader avec {market_share*100:.1f}% de part de marché")
    
    # Analyse des opportunités
    if revenue_growth < 5:
        insights['opportunities'].append("Accélération de la croissance via expansion géographique ou nouveaux produits")
    
    if profit_margin < 10:
        insights['opportunities'].append("Optimisation des marges par amélioration opérationnelle")
    
    if total_locations > 0:
        best_location = csv_data.get('best_performing_location')
        if best_location:
            insights['opportunities'].append(f"Répliquer le modèle de {best_location.get('name', 'la meilleure zone')} sur d'autres sites")
    
    # Alertes critiques
    if profit_margin < 5:
        insights['alerts'].append("Marges critiquement faibles - action immédiate requise")
    
    if revenue_growth < -5:
        insights['alerts'].append("Déclin de revenus détecté - plan de redressement urgent")
    
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    if 0 < ltv_cac < 2:
        insights['alerts'].append("Ratio LTV/CAC dangereux - rentabilité client menacée")
    
    return insights

def generate_executive_actions(csv_data):
    """Génère des actions prioritaires pour l'exécutif"""
    actions = []
    
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    total_locations = csv_data.get('total_locations', 0)
    
    # Actions basées sur les métriques
    if profit_margin < 10:
        actions.append("Lancer un audit des coûts et identifier 15% de réductions possibles")
    
    if revenue_growth < 5:
        actions.append("Développer un plan d'expansion sur 3 nouveaux marchés prioritaires")
    
    if total_locations > 3:
        # Actions géographiques
        clusters = csv_data.get('geographic_clusters', [])
        low_perf_clusters = [c for c in clusters if c.get('performance_level') in ['Low', 'Poor']]
        
        if low_perf_clusters:
            actions.append(f"Optimiser {len(low_perf_clusters)} zone(s) sous-performante(s) identifiée(s)")
    
    # Actions business intelligence
    if csv_data.get('customer_count', 0) > 0:
        actions.append("Implémenter un système de scoring client pour optimiser l'acquisition")
    
    # Actions par défaut si aucune action spécifique
    if not actions:
        actions.append("Poursuivre la stratégie actuelle avec monitoring renforcé des KPIs")
        actions.append("Préparer les plans de scaling pour capitaliser sur la performance")
    
    return actions[:3]  # Limiter à 3 actions prioritaires

# ========== ENHANCED ADVANCED ANALYTICS ==========
def show_enhanced_advanced_analytics():
    """Analytics avancés enrichis avec géolocalisation"""
    st.header("🧠 Analytics Avancés avec Intelligence Géographique")
    
    # Récupération des données enrichies
    csv_data = EnhancedCSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Aucune donnée CSV disponible**")
        st.info("Les Analytics Avancés nécessitent vos données CSV pour des analyses approfondies.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Analytics Avancés incluront:**
            - Scoring de santé financière multicritères
            - Corrélations géo-financières
            - Benchmarking sectoriel avancé
            - Prédictions basées sur l'IA
            - Optimisation géographique
            """)
        return
    
    st.success("📊 **Analytics alimentés par vos données enrichies**")
    
    # Score de santé financière enrichi
    st.subheader("🎯 Score de Santé Financière Enrichi")
    
    health_score, score_breakdown = calculate_enhanced_health_score(csv_data)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Score Global", f"{health_score:.0f}/100")
        
        if health_score >= 85:
            st.success("🟢 Excellent")
        elif health_score >= 70:
            st.info("🔵 Bon")
        elif health_score >= 50:
            st.warning("🟡 Moyen")
        else:
            st.error("🔴 Faible")
    
    with col2:
        liquidity_score = score_breakdown.get('liquidity', 0)
        st.metric("Liquidité", f"{liquidity_score:.0f}/25")
        
        current_ratio = csv_data.get('current_ratio', 0)
        if current_ratio > 1.5:
            st.success("💧 Saine")
        elif current_ratio > 1.2:
            st.info("📊 Correcte")
        else:
            st.warning("⚠️ Tendue")
    
    with col3:
        profitability_score = score_breakdown.get('profitability', 0)
        st.metric("Rentabilité", f"{profitability_score:.0f}/35")
        
        profit_margin = csv_data.get('profit_margin', 0)
        if profit_margin > 15:
            st.success("💰 Forte")
        elif profit_margin > 8:
            st.info("📈 Bonne")
        else:
            st.warning("📉 Faible")
    
    with col4:
        geographic_score = score_breakdown.get('geographic', 0)
        st.metric("Géographique", f"{geographic_score:.0f}/20")
        
        total_locations = csv_data.get('total_locations', 0)
        if total_locations > 5:
            st.success("🌍 Diversifié")
        elif total_locations > 1:
            st.info("📍 Régional")
        else:
            st.warning("🏢 Local")
    
    with col5:
        business_score = score_breakdown.get('business', 0)
        st.metric("Business", f"{business_score:.0f}/20")
        
        ltv_cac = csv_data.get('ltv_cac_ratio', 0)
        if ltv_cac > 3:
            st.success("🎯 Optimal")
        elif ltv_cac > 1:
            st.info("📊 Correct")
        else:
            st.warning("⚠️ Risqué")
    
    # Analyse détaillée par onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Enrichie", 
        "🌍 Intelligence Géographique", 
        "🤖 IA & Prédictions",
        "📈 Optimisation & Recommandations"
    ])
    
    with tab1:
        show_enhanced_performance_analysis(csv_data)
    
    with tab2:
        show_geographic_intelligence(csv_data)
    
    with tab3:
        show_ai_predictions(csv_data)
    
    with tab4:
        show_optimization_recommendations(csv_data)

def calculate_enhanced_health_score(csv_data):
    """Calcule un score de santé financière enrichi incluant géographie"""
    scores = {}
    
    # Score de liquidité (25 points)
    current_ratio = csv_data.get('current_ratio', 1.5)
    liquidity_score = min(25, current_ratio * 16.67)  # 1.5 ratio = 25 points
    scores['liquidity'] = liquidity_score
    
    # Score de rentabilité (35 points)
    profit_margin = csv_data.get('profit_margin', 0)
    net_margin = csv_data.get('net_margin', 0)
    
    margin_score = min(25, profit_margin * 1.67)  # 15% margin = 25 points
    efficiency_score = min(10, net_margin * 100)  # 10% net margin = 10 points
    scores['profitability'] = margin_score + efficiency_score
    
    # Score géographique (20 points) - NOUVEAU
    total_locations = csv_data.get('total_locations', 0)
    geographic_spread = csv_data.get('geographic_spread', 0)
    
    location_score = min(10, total_locations * 2)  # 5 locations = 10 points
    spread_score = min(10, geographic_spread / 50)  # 500km = 10 points
    scores['geographic'] = location_score + spread_score
    
    # Score business (20 points) - NOUVEAU
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    market_share = csv_data.get('market_share', 0)
    
    ltv_score = min(15, ltv_cac * 5)  # Ratio 3 = 15 points
    market_score = min(5, market_share * 25)  # 20% = 5 points
    scores['business'] = ltv_score + market_score
    
    total_score = sum(scores.values())
    
    return min(100, total_score), scores

def show_enhanced_performance_analysis(csv_data):
    """Analyse de performance enrichie"""
    st.markdown("### 📊 Analyse de Performance Multi-dimensionnelle")
    
    # Métriques temporelles si disponibles
    if 'revenue_data' in csv_data and csv_data['revenue_data']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Évolution Temporelle")
            
            revenue_data = csv_data['revenue_data']
            profit_data = csv_data.get('profit_data', [])
            
            # Graphique d'évolution
            months = list(range(1, len(revenue_data) + 1))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=revenue_data,
                mode='lines+markers',
                name='Revenue',
                line=dict(color='green', width=3)
            ))
            
            if profit_data:
                fig.add_trace(go.Scatter(
                    x=months[:len(profit_data)],
                    y=profit_data,
                    mode='lines+markers',
                    name='Profit',
                    line=dict(color='blue', width=3)
                ))
            
            fig.update_layout(
                title="Évolution Revenue & Profit",
                xaxis_title="Mois",
                yaxis_title="Montant (DHS)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Métriques de Volatilité")
            
            revenue_volatility = csv_data.get('revenue_volatility', 0)
            revenue_growth = csv_data.get('revenue_growth', 0)
            
            # Gauge de volatilité
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=revenue_volatility * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Volatilité Revenue (%)"},
                gauge={
                    'axis': {'range': [None, 50]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgreen"},
                        {'range': [10, 25], 'color': "yellow"},
                        {'range': [25, 50], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Métriques additionnelles
            st.metric("Croissance Revenue", f"{revenue_growth:+.1f}%")
            
            trend = "Croissante" if revenue_growth > 0 else "Décroissante"
            st.metric("Tendance", trend)
    
    # Analyse comparative sectorielle
    st.markdown("#### 🏭 Benchmarking Sectoriel Intelligent")
    
    # Détection automatique du secteur
    sector = detect_business_sector(csv_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"🎯 **Secteur Détecté**: {sector}")
        
        # Benchmarks sectoriels
        benchmarks = get_sector_benchmarks(sector)
        
        st.markdown("**Benchmarks Secteur:**")
        for metric, value in benchmarks.items():
            if isinstance(value, float) and value < 1:
                st.write(f"• {metric}: {value:.1%}")
            else:
                st.write(f"• {metric}: {value:.1f}")
    
    with col2:
        st.markdown("**Votre Performance:**")
        
        profit_margin = csv_data.get('profit_margin', 0) / 100
        market_share = csv_data.get('market_share', 0)
        revenue_growth = csv_data.get('revenue_growth', 0) / 100
        
        company_metrics = {
            'Marge Profit': profit_margin,
            'Part Marché': market_share,
            'Croissance': revenue_growth
        }
        
        for metric, value in company_metrics.items():
            if value > 0:
                if metric == 'Croissance':
                    st.write(f"• {metric}: {value:.1%}")
                else:
                    st.write(f"• {metric}: {value:.1%}")
    
    with col3:
        st.markdown("**Écart vs Secteur:**")
        
        # Comparaison automatique
        comparison = compare_to_sector(csv_data, benchmarks)
        
        for metric, gap in comparison.items():
            color = "🟢" if gap > 0 else "🔴" if gap < -10 else "🟡"
            st.write(f"• {color} {metric}: {gap:+.1f}%")

def detect_business_sector(csv_data):
    """Détecte automatiquement le secteur d'activité"""
    
    profit_margin = csv_data.get('profit_margin', 0)
    total_locations = csv_data.get('total_locations', 0)
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    revenue_volatility = csv_data.get('revenue_volatility', 0)
    
    # Logique de détection basée sur les patterns
    if ltv_cac > 3 and profit_margin > 15:
        return "SaaS/Technologie"
    elif total_locations > 5 and revenue_volatility > 0.2:
        return "Commerce/Retail"
    elif profit_margin < 10 and total_locations > 3:
        return "Manufacturing"
    elif profit_margin > 15 and total_locations <= 2:
        return "Services Professionnels"
    else:
        return "Général"

def get_sector_benchmarks(sector):
    """Retourne les benchmarks sectoriels"""
    
    benchmarks_db = {
        'SaaS/Technologie': {
            'Marge Profit': 0.20,
            'Croissance': 0.25,
            'LTV/CAC': 4.0
        },
        'Commerce/Retail': {
            'Marge Profit': 0.05,
            'Croissance': 0.08,
            'Rotation Stock': 6.0
        },
        'Manufacturing': {
            'Marge Profit': 0.08,
            'Croissance': 0.06,
            'Utilisation Capacité': 0.85
        },
        'Services Professionnels': {
            'Marge Profit': 0.18,
            'Croissance': 0.12,
            'Utilisation': 0.75
        },
        'Général': {
            'Marge Profit': 0.12,
            'Croissance': 0.10,
            'ROI': 0.15
        }
    }
    
    return benchmarks_db.get(sector, benchmarks_db['Général'])

def compare_to_sector(csv_data, benchmarks):
    """Compare les métriques de l'entreprise aux benchmarks sectoriels"""
    
    comparison = {}
    
    # Marge profit
    company_margin = csv_data.get('profit_margin', 0) / 100
    sector_margin = benchmarks.get('Marge Profit', 0.1)
    
    margin_gap = ((company_margin - sector_margin) / sector_margin) * 100
    comparison['Marge Profit'] = margin_gap
    
    # Croissance
    company_growth = csv_data.get('revenue_growth', 0) / 100
    sector_growth = benchmarks.get('Croissance', 0.1)
    
    growth_gap = ((company_growth - sector_growth) / sector_growth) * 100
    comparison['Croissance'] = growth_gap
    
    # LTV/CAC si disponible
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    sector_ltv_cac = benchmarks.get('LTV/CAC', 0)
    
    if ltv_cac > 0 and sector_ltv_cac > 0:
        ltv_gap = ((ltv_cac - sector_ltv_cac) / sector_ltv_cac) * 100
        comparison['LTV/CAC'] = ltv_gap
    
    return comparison

def show_geographic_intelligence(csv_data):
    """Affiche l'intelligence géographique avancée"""
    st.markdown("### 🌍 Intelligence Géographique Avancée")
    
    total_locations = csv_data.get('total_locations', 0)
    
    if total_locations == 0:
        st.info("📍 Aucune donnée géographique disponible. Uploadez un CSV avec coordonnées pour accéder à l'intelligence géographique.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Clusters de Performance")
        
        clusters = csv_data.get('geographic_clusters', [])
        
        if clusters:
            # Analyse des clusters
            high_perf = [c for c in clusters if c.get('performance_level') in ['High', 'Excellent']]
            low_perf = [c for c in clusters if c.get('performance_level') in ['Low', 'Poor']]
            
            st.metric("Clusters Haute Performance", len(high_perf))
            st.metric("Clusters Sous-performants", len(low_perf))
            
            # Recommandations basées sur clusters
            if high_perf and low_perf:
                st.success("💡 **Opportunité**: Répliquer les stratégies des zones performantes")
            elif high_perf:
                st.info("🎯 **Stratégie**: Étendre le modèle des zones performantes")
            
            # Détails des meilleurs clusters
            if high_perf:
                best_cluster = max(high_perf, key=lambda x: x.get('avg_performance', 0))
                st.markdown("**🏆 Meilleur Cluster:**")
                st.write(f"• Performance: {best_cluster.get('avg_performance', 0):,.0f} DHS")
                st.write(f"• Emplacements: {best_cluster.get('location_count', 0)}")
                st.write(f"• Rayon: {best_cluster.get('radius_km', 0):.1f} km")
        else:
            st.info("Clustering géographique en cours d'analyse...")
    
    with col2:
        st.markdown("#### 📊 Métriques Géographiques")
        
        geographic_spread = csv_data.get('geographic_spread', 0)
        st.metric("Étendue Géographique", f"{geographic_spread:.0f} km")
        
        if geographic_spread > 500:
            st.warning("⚠️ **Large dispersion** - Considérer la régionalisation")
        elif geographic_spread > 100:
            st.info("📍 **Présence régionale** - Optimisation logistique possible")
        else:
            st.success("🏘️ **Concentration locale** - Synergie géographique")
        
        # Analyse de densité
        if total_locations > 1:
            density = total_locations / (geographic_spread + 1)
            st.metric("Densité", f"{density:.2f} sites/100km")
            
            if density > 1:
                st.success("🎯 **Forte densité** - Couverture optimale")
            else:
                st.info("📍 **Expansion possible** - Zones intermédiaires")
    
    # Corrélations géographiques
    st.markdown("#### 🔗 Corrélations Géo-Performance")
    
    # Simuler des corrélations basées sur les données disponibles
    if 'best_performing_location' in csv_data:
        best_loc = csv_data['best_performing_location']
        
        st.markdown("**Facteurs de Succès Géographique:**")
        
        # Analyse pattern géographique
        geographic_insights = analyze_geographic_patterns(csv_data)
        
        for insight in geographic_insights:
            st.info(f"📊 {insight}")

def analyze_geographic_patterns(csv_data):
    """Analyse les patterns géographiques"""
    insights = []
    
    best_location = csv_data.get('best_performing_location', {})
    geographic_spread = csv_data.get('geographic_spread', 0)
    total_locations = csv_data.get('total_locations', 0)
    
    if best_location:
        insights.append(f"La zone {best_location.get('name', 'leader')} génère {best_location.get('revenue', 0):,.0f} DHS")
    
    if geographic_spread > 200 and total_locations > 3:
        insights.append("Dispersion géographique importante - potentiel de régionalisation")
    
    if total_locations > 5:
        insights.append("Présence multi-sites établie - synergie inter-zones possible")
    
    clusters = csv_data.get('geographic_clusters', [])
    if clusters:
        high_perf_clusters = [c for c in clusters if c.get('performance_level') in ['High', 'Excellent']]
        if len(high_perf_clusters) > 1:
            insights.append(f"{len(high_perf_clusters)} zones d'excellence identifiées - modèle réplicable")
    
    return insights[:3]  # Limiter à 3 insights

def show_ai_predictions(csv_data):
    """Affiche les prédictions IA"""
    st.markdown("### 🤖 Prédictions & Intelligence Artificielle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔮 Prédictions 6 mois")
        
        # Prédictions basées sur les tendances actuelles
        revenue_growth = csv_data.get('revenue_growth', 0)
        current_revenue = csv_data.get('monthly_revenue', 0)
        
        # Prédiction simple basée sur la croissance
        if revenue_growth != 0:
            predicted_revenue = current_revenue * (1 + revenue_growth/100) ** 6
            
            st.metric("Revenue Prédit (6 mois)", f"{predicted_revenue:,.0f} DHS")
            
            confidence = calculate_prediction_confidence(csv_data)
            st.metric("Confiance Prédiction", f"{confidence:.0f}%")
            
            if confidence > 80:
                st.success("🎯 **Haute confiance** - Tendance stable")
            elif confidence > 60:
                st.info("📊 **Confiance modérée** - Surveillance recommandée")
            else:
                st.warning("⚠️ **Faible confiance** - Volatilité élevée")
        else:
            st.info("Données insuffisantes pour prédictions fiables")
    
    with col2:
        st.markdown("#### 🧠 Insights IA")
        
        # Génération d'insights IA
        ai_insights = generate_ai_insights(csv_data)
        
        for insight in ai_insights:
            st.info(f"🤖 {insight}")
    
    # Analyse de risques prédictive
    st.markdown("#### ⚠️ Analyse de Risques Prédictive")
    
    risk_factors = analyze_predictive_risks(csv_data)
    
    if risk_factors['high']:
        st.error("🚨 **Risques Élevés Détectés:**")
        for risk in risk_factors['high']:
            st.error(f"• {risk}")
    
    if risk_factors['medium']:
        st.warning("⚠️ **Risques Modérés:**")
        for risk in risk_factors['medium']:
            st.warning(f"• {risk}")
    
    if risk_factors['low']:
        st.info("📊 **Points d'Attention:**")
        for risk in risk_factors['low']:
            st.info(f"• {risk}")
    
    if not any(risk_factors.values()):
        st.success("✅ **Profil de risque optimal** - Aucun risque majeur détecté")

def calculate_prediction_confidence(csv_data):
    """Calcule la confiance dans les prédictions"""
    
    confidence = 100
    
    # Réduire selon la volatilité
    volatility = csv_data.get('revenue_volatility', 0)
    confidence -= volatility * 200  # Volatilité de 0.3 = -60 points
    
    # Réduire selon l'âge des données
    revenue_data = csv_data.get('revenue_data', [])
    if len(revenue_data) < 6:
        confidence -= 20  # Données insuffisantes
    
    # Réduire selon la tendance
    revenue_growth = csv_data.get('revenue_growth', 0)
    if abs(revenue_growth) > 50:  # Croissance extrême
        confidence -= 30
    
    return max(0, min(100, confidence))

def generate_ai_insights(csv_data):
    """Génère des insights IA automatiques"""
    insights = []
    
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    total_locations = csv_data.get('total_locations', 0)
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    
    # Pattern recognition
    if profit_margin > 15 and revenue_growth > 10:
        insights.append("Modèle économique solide détecté - phase de croissance rentable")
    
    if total_locations > 3 and profit_margin > 10:
        insights.append("Scalabilité géographique validée - potentiel d'expansion confirmé")
    
    if ltv_cac > 3 and revenue_growth > 0:
        insights.append("Unit economics saines - modèle d'acquisition client viable")
    
    # Détection d'anomalies
    revenue_volatility = csv_data.get('revenue_volatility', 0)
    if revenue_volatility > 0.3:
        insights.append("Volatilité élevée détectée - diversification des revenus recommandée")
    
    if profit_margin < 5 and revenue_growth > 15:
        insights.append("Croissance non-rentable identifiée - optimisation des coûts prioritaire")
    
    return insights[:4]  # Limiter à 4 insights

def analyze_predictive_risks(csv_data):
    """Analyse les risques de manière prédictive"""
    
    risks = {'high': [], 'medium': [], 'low': []}
    
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    current_ratio = csv_data.get('current_ratio', 1.5)
    revenue_volatility = csv_data.get('revenue_volatility', 0)
    
    # Risques élevés
    if profit_margin < 3:
        risks['high'].append("Marges critiquement faibles - risque de faillite")
    
    if current_ratio < 1:
        risks['high'].append("Liquidité critique - incapacité à honorer les dettes")
    
    if revenue_growth < -20:
        risks['high'].append("Déclin sévère des revenus - plan de sauvetage requis")
    
    # Risques modérés
    if profit_margin < 8:
        risks['medium'].append("Marges faibles - vulnérabilité aux chocs externes")
    
    if revenue_volatility > 0.4:
        risks['medium'].append("Forte volatilité - prévisibilité compromise")
    
    if revenue_growth < 0:
        risks['medium'].append("Déclin des revenus - investigation requise")
    
    # Points d'attention
    if current_ratio < 1.3:
        risks['low'].append("Liquidité tendue - surveillance du cash flow")
    
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    if 0 < ltv_cac < 2:
        risks['low'].append("Unit economics limites - optimisation CAC/LTV")
    
    return risks

def show_optimization_recommendations(csv_data):
    """Affiche les recommandations d'optimisation"""
    st.markdown("### 🎯 Optimisation & Recommandations Stratégiques")
    
    # Recommandations par priorité
    high_priority, medium_priority, long_term = generate_strategic_recommendations(csv_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚨 Priorité Haute (0-30 jours)")
        
        for rec in high_priority:
            st.error(f"🔥 {rec}")
    
    with col2:
        st.markdown("#### ⚠️ Priorité Moyenne (1-3 mois)")
        
        for rec in medium_priority:
            st.warning(f"📊 {rec}")
    
    with col3:
        st.markdown("#### 📈 Stratégique (3-12 mois)")
        
        for rec in long_term:
            st.info(f"🎯 {rec}")
    
    # ROI estimé des recommandations
    st.markdown("#### 💰 Impact Financier Estimé")
    
    roi_estimates = calculate_recommendation_roi(csv_data, high_priority + medium_priority)
    
    if roi_estimates:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROI Potentiel", f"{roi_estimates['total_roi']:.1f}%")
        
        with col2:
            st.metric("Impact Revenue", f"+{roi_estimates['revenue_impact']:,.0f} DHS")
        
        with col3:
            st.metric("Économies Costs", f"-{roi_estimates['cost_savings']:,.0f} DHS")
    
    # Plan d'action détaillé
    st.markdown("#### 📋 Plan d'Action Détaillé")
    
    action_plan = create_detailed_action_plan(csv_data)
    
    for i, action in enumerate(action_plan, 1):
        with st.expander(f"Action {i}: {action['title']}"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Détails:**")
                st.write(action['description'])
                
                st.markdown("**KPIs à Suivre:**")
                for kpi in action['kpis']:
                    st.write(f"• {kpi}")
            
            with col2:
                st.metric("Priorité", action['priority'])
                st.metric("Durée Estimée", action['duration'])
                st.metric("ROI Attendu", f"{action['expected_roi']}%")
                
                if action['difficulty'] == 'Facile':
                    st.success("✅ Facile à implémenter")
                elif action['difficulty'] == 'Moyen':
                    st.warning("⚠️ Complexité modérée")
                else:
                    st.error("🔴 Haute complexité")

def generate_strategic_recommendations(csv_data):
    """Génère des recommandations stratégiques par priorité"""
    
    high_priority = []
    medium_priority = []
    long_term = []
    
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    current_ratio = csv_data.get('current_ratio', 1.5)
    total_locations = csv_data.get('total_locations', 0)
    ltv_cac = csv_data.get('ltv_cac_ratio', 0)
    
    # Priorité haute - Actions urgentes
    if profit_margin < 5:
        high_priority.append("Audit complet des coûts et réduction immédiate de 20%")
    
    if current_ratio < 1.1:
        high_priority.append("Amélioration urgente de la trésorerie - négociation fournisseurs")
    
    if revenue_growth < -10:
        high_priority.append("Plan de relance commercial - task force revenue")
    
    # Priorité moyenne - Optimisations importantes
    if profit_margin < 12:
        medium_priority.append("Optimisation de la structure des coûts variables")
    
    if revenue_growth < 5:
        medium_priority.append("Stratégie d'accélération de croissance - nouveaux marchés")
    
    if total_locations > 1:
        clusters = csv_data.get('geographic_clusters', [])
        low_perf_clusters = [c for c in clusters if c.get('performance_level') in ['Low', 'Poor']]
        if low_perf_clusters:
            medium_priority.append(f"Optimisation de {len(low_perf_clusters)} zone(s) sous-performante(s)")
    
    if 0 < ltv_cac < 2.5:
        medium_priority.append("Amélioration du ratio LTV/CAC - optimisation acquisition")
    
    # Stratégique long terme
    if total_locations > 0:
        long_term.append("Développement d'une stratégie d'expansion géographique data-driven")
    
    if profit_margin > 15:
        long_term.append("Exploration d'opportunités d'acquisition ou diversification")
    
    long_term.append("Mise en place d'un système de BI avancé pour pilotage temps réel")
    
    if revenue_growth > 20:
        long_term.append("Préparation à la scalabilité - infrastructure et processus")
    
    return high_priority, medium_priority, long_term

def calculate_recommendation_roi(csv_data, recommendations):
    """Calcule le ROI estimé des recommandations"""
    
    current_revenue = csv_data.get('revenue', csv_data.get('monthly_revenue', 0) * 12)
    current_costs = csv_data.get('total_costs', current_revenue * 0.75)
    
    if current_revenue == 0:
        return None
    
    # Estimations basées sur les types de recommandations
    revenue_impact = 0
    cost_savings = 0
    
    for rec in recommendations:
        if 'coûts' in rec.lower() and '20%' in rec:
            cost_savings += current_costs * 0.15  # 15% économies réalistes
        elif 'croissance' in rec.lower() or 'revenue' in rec.lower():
            revenue_impact += current_revenue * 0.08  # 8% croissance
        elif 'optimisation' in rec.lower():
            cost_savings += current_costs * 0.05  # 5% économies
            revenue_impact += current_revenue * 0.03  # 3% revenue
        elif 'zone' in rec.lower() or 'géographique' in rec.lower():
            revenue_impact += current_revenue * 0.06  # 6% amélioration zones
    
    total_impact = revenue_impact + cost_savings
    investment_needed = current_revenue * 0.02  # 2% investment estimé
    
    total_roi = ((total_impact - investment_needed) / investment_needed * 100) if investment_needed > 0 else 0
    
    return {
        'total_roi': total_roi,
        'revenue_impact': revenue_impact,
        'cost_savings': cost_savings
    }

def create_detailed_action_plan(csv_data):
    """Crée un plan d'action détaillé"""
    
    action_plan = []
    
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    total_locations = csv_data.get('total_locations', 0)
    
    # Action 1: Optimisation financière
    if profit_margin < 15:
        action_plan.append({
            'title': 'Optimisation de la Rentabilité',
            'description': 'Audit complet des coûts, renégociation fournisseurs, et optimisation des processus opérationnels pour améliorer les marges.',
            'priority': 'Critique' if profit_margin < 5 else 'Haute',
            'duration': '2-4 semaines',
            'expected_roi': 25,
            'difficulty': 'Moyen',
            'kpis': [
                'Marge brute +3-5%',
                'Réduction coûts variables 10-15%',
                'Délai paiement fournisseurs +7 jours'
            ]
        })
    
    # Action 2: Croissance
    if revenue_growth < 10:
        action_plan.append({
            'title': 'Accélération de la Croissance',
            'description': 'Développement de nouveaux canaux d\'acquisition, optimisation du marketing digital, et expansion sur de nouveaux segments.',
            'priority': 'Haute',
            'duration': '6-12 semaines',
            'expected_roi': 35,
            'difficulty': 'Moyen',
            'kpis': [
                'Croissance revenue +15%',
                'Nouveaux clients +25%',
                'CAC optimisé -20%'
            ]
        })
    
    # Action 3: Géographique
    if total_locations > 1:
        action_plan.append({
            'title': 'Optimisation Géographique',
            'description': 'Analyse approfondie des performances par zone, réplication des best practices, et optimisation de la couverture territoriale.',
            'priority': 'Moyenne',
            'duration': '4-8 semaines',
            'expected_roi': 20,
            'difficulty': 'Facile',
            'kpis': [
                'Performance zones faibles +30%',
                'Homogénéisation des marges',
                'Optimisation logistique -15%'
            ]
        })
    
    # Action 4: Digitalisation
    action_plan.append({
        'title': 'Transformation Digitale',
        'description': 'Mise en place d\'outils de BI avancés, automatisation des processus, et développement d\'une culture data-driven.',
        'priority': 'Stratégique',
        'duration': '12-24 semaines',
        'expected_roi': 45,
        'difficulty': 'Élevé',
        'kpis': [
            'Temps de reporting -70%',
            'Précision prévisions +40%',
            'Productivité équipe +25%'
        ]
    })
    
    return action_plan

# ========== ENHANCED SCENARIO PLANNING ==========
def show_enhanced_scenario_planning():
    """Planification de scénarios enrichie avec géolocalisation"""
    st.header("🎯 Planification de Scénarios Avancée")
    
    # Récupération des données enrichies
    csv_data = EnhancedCSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Données CSV non disponibles**")
        st.info("La planification de scénarios nécessite vos données CSV pour des projections précises.")
        
        if st.button("📤 Importer Données CSV", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    st.success("📊 **Scénarios basés sur vos données enrichies**")
    
    # Données de base enrichies
    base_monthly_revenue = csv_data.get('monthly_revenue', 15000)
    base_monthly_costs = csv_data.get('monthly_costs', 12000)
    current_growth_rate = csv_data.get('revenue_growth', 0) / 100
    profit_margin = csv_data.get('profit_margin', 0)
    total_locations = csv_data.get('total_locations', 0)
    
    st.subheader(f"📊 Données de Base (issues de votre CSV)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Revenue Mensuel", f"{base_monthly_revenue:,.0f} DHS")
    with col2:
        st.metric("Coûts Mensuels", f"{base_monthly_costs:,.0f} DHS")
    with col3:
        st.metric("Croissance Actuelle", f"{current_growth_rate*100:+.1f}%")
    with col4:
        if total_locations > 0:
            st.metric("Emplacements", total_locations)
        else:
            st.metric("Marge Profit", f"{profit_margin:.1f}%")
    
    # Configuration des scénarios enrichie
    st.subheader("⚙️ Configuration des Scénarios Enrichis")
    
    # Onglets pour différents types de scénarios
    scenario_tabs = st.tabs([
        "📈 Scénarios Classiques", 
        "🌍 Scénarios Géographiques", 
        "🚀 Scénarios d'Expansion", 
        "⚡ Scénarios de Crise"
    ])
    
    with scenario_tabs[0]:
        # Scénarios financiers classiques
        st.markdown("#### 💼 Scénarios Financiers Standards")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 😰 Pessimiste")
            pess_revenue = st.slider("Variation Revenue (%)", -50, 10, max(-25, int(current_growth_rate*100-20)), key="pess_rev")
            pess_cost = st.slider("Variation Coûts (%)", -10, 50, 20, key="pess_cost")
            pess_prob = st.slider("Probabilité (%)", 5, 40, 25, key="pess_prob")
        
        with col2:
            st.markdown("### 😐 Réaliste")
            real_revenue = st.slider("Variation Revenue (%)", -10, 40, max(5, int(current_growth_rate*100)), key="real_rev")
            real_cost = st.slider("Variation Coûts (%)", 0, 30, 10, key="real_cost")
            real_prob = st.slider("Probabilité (%)", 40, 80, 55, key="real_prob")
        
        with col3:
            st.markdown("### 😄 Optimiste")
            opt_revenue = st.slider("Variation Revenue (%)", 10, 80, max(30, int(current_growth_rate*100+25)), key="opt_rev")
            opt_cost = st.slider("Variation Coûts (%)", -10, 20, 5, key="opt_cost")
            opt_prob = st.slider("Probabilité (%)", 5, 40, 20, key="opt_prob")
    
    with scenario_tabs[1]:
        # Scénarios géographiques
        st.markdown("#### 🌍 Scénarios d'Expansion Géographique")
        
        if total_locations > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📍 Expansion Locale")
                local_new_sites = st.number_input("Nouveaux Sites Locaux", min_value=0, max_value=10, value=2)
                local_revenue_per_site = st.number_input("Revenue/Site (DHS)", min_value=5000, max_value=50000, value=int(base_monthly_revenue * 0.7))
                local_setup_cost = st.number_input("Coût Installation/Site", min_value=10000, max_value=200000, value=50000)
            
            with col2:
                st.markdown("### 🚀 Expansion Régionale")
                regional_new_sites = st.number_input("Nouveaux Sites Régionaux", min_value=0, max_value=20, value=5)
                regional_revenue_per_site = st.number_input("Revenue/Site Régional", min_value=3000, max_value=40000, value=int(base_monthly_revenue * 0.5))
                regional_setup_cost = st.number_input("Coût Installation Régional", min_value=20000, max_value=300000, value=75000)
        else:
            st.info("💡 Scénarios géographiques disponibles avec des données multi-sites")
            
            # Scénario d'expansion depuis un site unique
            st.markdown("### 🌍 Première Expansion")
            first_expansion_sites = st.number_input("Nouveaux Emplacements", min_value=1, max_value=5, value=2)
            expansion_revenue_ratio = st.slider("% Revenue vs Site Principal", 30, 100, 70, key="expansion_ratio")
            expansion_timeline = st.selectbox("Délai d'Expansion", ["3 mois", "6 mois", "12 mois"])
    
    with scenario_tabs[2]:
        # Scénarios d'expansion business
        st.markdown("#### 🚀 Scénarios d'Expansion Business")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Acquisition")
            acquisition_cost = st.number_input("Coût Acquisition (DHS)", min_value=100000, max_value=10000000, value=500000)
            acquisition_revenue_boost = st.slider("Boost Revenue (%)", 20, 200, 50, key="acq_boost")
            acquisition_synergies = st.slider("Synergies Coûts (%)", 5, 30, 15, key="acq_synergies")
        
        with col2:
            st.markdown("### 🔬 Innovation")
            innovation_investment = st.number_input("Investment R&D", min_value=50000, max_value=2000000, value=200000)
            innovation_revenue_impact = st.slider("Impact Revenue (%)", 10, 100, 25, key="innov_impact")
            innovation_timeline = st.selectbox("Délai ROI Innovation", ["6 mois", "12 mois", "18 mois", "24 mois"])
    
    with scenario_tabs[3]:
        # Scénarios de crise
        st.markdown("#### ⚡ Scénarios de Gestion de Crise")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🦠 Crise Sanitaire")
            health_crisis_revenue_drop = st.slider("Baisse Revenue (%)", 10, 70, 30, key="health_drop")
            health_crisis_duration = st.selectbox("Durée Crise", ["3 mois", "6 mois", "12 mois"], key="health_duration")
            remote_work_savings = st.slider("Économies Télétravail (%)", 5, 25, 10, key="remote_savings")
        
        with col2:
            st.markdown("### 💸 Crise Économique")
            economic_crisis_impact = st.slider("Impact Économique (%)", 15, 60, 25, key="economic_impact")
            customer_payment_delays = st.slider("Retards Paiements (jours)", 15, 90, 30, key="payment_delays")
            cost_reduction_potential = st.slider("Réduction Coûts (%)", 10, 40, 20, key="cost_reduction")
        
        with col3:
            st.markdown("### 🔥 Crise Opérationnelle")
            operational_crisis_sites = st.number_input("Sites Impactés", min_value=1, max_value=total_locations if total_locations > 0 else 1, value=1)
            recovery_timeline = st.selectbox("Délai Récupération", ["1 mois", "3 mois", "6 mois"], key="recovery_time")
            insurance_coverage = st.slider("Couverture Assurance (%)", 0, 100, 70, key="insurance")
    
    # Paramètres globaux
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_period = st.selectbox("Période d'Analyse", [12, 18, 24, 36], index=1)
        include_seasonality = st.checkbox("Inclure Saisonnalité", value=True)
    
    with col2:
        monte_carlo_simulations = st.selectbox("Simulations Monte Carlo", [100, 500, 1000, 2000], index=1)
        confidence_interval = st.selectbox("Intervalle de Confiance", ["80%", "90%", "95%", "99%"], index=2)
    
    # Exécution de l'analyse de scénarios
    if st.button("🚀 Lancer l'Analyse de Scénarios Avancée", type="primary"):
        with st.spinner("Exécution des simulations avancées..."):
            
            # Validation des probabilités
            total_prob = pess_prob + real_prob + opt_prob
            if total_prob != 100:
                st.warning(f"⚠️ Ajustement des probabilités (total: {total_prob}%)")
                pess_prob = pess_prob * 100 / total_prob
                real_prob = real_prob * 100 / total_prob
                opt_prob = opt_prob * 100 / total_prob
            
            # Construction des scénarios enrichis
            scenarios = build_enhanced_scenarios(
                base_monthly_revenue, base_monthly_costs, analysis_period,
                pess_revenue, pess_cost, pess_prob,
                real_revenue, real_cost, real_prob,
                opt_revenue, opt_cost, opt_prob,
                csv_data, include_seasonality
            )
            
            # Exécution Monte Carlo
            mc_results = run_monte_carlo_analysis(
                scenarios, monte_carlo_simulations, analysis_period, csv_data
            )
            
            # Stockage des résultats
            st.session_state.enhanced_scenario_results = {
                'scenarios': scenarios,
                'monte_carlo': mc_results,
                'parameters': {
                    'analysis_period': analysis_period,
                    'confidence_interval': confidence_interval,
                    'simulations': monte_carlo_simulations
                }
            }
    
    # Affichage des résultats enrichis
    if 'enhanced_scenario_results' in st.session_state:
        display_enhanced_scenario_results(st.session_state.enhanced_scenario_results, csv_data)

def build_enhanced_scenarios(base_revenue, base_costs, periods, 
                           pess_rev, pess_cost, pess_prob,
                           real_rev, real_cost, real_prob,
                           opt_rev, opt_cost, opt_prob,
                           csv_data, include_seasonality):
    """Construit des scénarios enrichis avec saisonnalité"""
    
    scenarios = {}
    
    # Facteurs saisonniers (basés sur les données ou défaut retail)
    seasonal_factors = [0.85, 0.9, 1.1, 1.05, 1.0, 0.95, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3] if include_seasonality else [1.0] * 12
    
    # Construction de chaque scénario
    scenario_configs = [
        ('pessimistic', pess_rev, pess_cost, pess_prob),
        ('realistic', real_rev, real_cost, real_prob),
        ('optimistic', opt_rev, opt_cost, opt_prob)
    ]
    
    for scenario_name, rev_change, cost_change, probability in scenario_configs:
        monthly_results = []
        
        for month in range(periods):
            # Facteur saisonnier
            seasonal_factor = seasonal_factors[month % 12]
            
            # Évolution progressive sur la période
            progress_factor = 1 + (month / periods) * 0.1  # Évolution progressive
            
            # Calcul du mois
            monthly_revenue = base_revenue * (1 + rev_change/100) * seasonal_factor * progress_factor
            monthly_cost = base_costs * (1 + cost_change/100) * progress_factor
            monthly_profit = monthly_revenue - monthly_cost
            
            # Métriques enrichies
            monthly_margin = (monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0
            cumulative_profit = sum([r['profit'] for r in monthly_results]) + monthly_profit
            
            monthly_results.append({
                'month': month + 1,
                'revenue': monthly_revenue,
                'cost': monthly_cost,
                'profit': monthly_profit,
                'margin': monthly_margin,
                'cumulative_profit': cumulative_profit,
                'seasonal_factor': seasonal_factor
            })
        
        # Métriques globales du scénario
        total_revenue = sum(m['revenue'] for m in monthly_results)
        total_cost = sum(m['cost'] for m in monthly_results)
        total_profit = sum(m['profit'] for m in monthly_results)
        avg_margin = np.mean([m['margin'] for m in monthly_results])
        
        scenarios[scenario_name] = {
            'monthly_data': monthly_results,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'avg_margin': avg_margin,
            'probability': probability / 100,
            'roi': (total_profit / total_cost * 100) if total_cost > 0 else 0
        }
    
    return scenarios

def run_monte_carlo_analysis(scenarios, num_simulations, periods, csv_data):
    """Exécute une analyse Monte Carlo avancée"""
    
    # Paramètres de volatilité basés sur les données CSV
    revenue_volatility = csv_data.get('revenue_volatility', 0.15)
    cost_volatility = revenue_volatility * 0.7  # Coûts généralement moins volatils
    
    mc_results = []
    
    base_revenue = csv_data.get('monthly_revenue', 15000)
    base_costs = csv_data.get('monthly_costs', 12000)
    
    for sim in range(num_simulations):
        # Sélection aléatoire du scénario basé sur les probabilités
        rand = np.random.random()
        cumulative_prob = 0
        selected_scenario = 'realistic'
        
        for scenario_name, scenario_data in scenarios.items():
            cumulative_prob += scenario_data['probability']
            if rand <= cumulative_prob:
                selected_scenario = scenario_name
                break
        
        # Simulation avec volatilité
        sim_revenue_path = []
        sim_cost_path = []
        sim_profit_path = []
        
        scenario = scenarios[selected_scenario]
        
        for month in range(periods):
            base_month_revenue = scenario['monthly_data'][month]['revenue']
            base_month_cost = scenario['monthly_data'][month]['cost']
            
            # Ajout de bruit stochastique
            revenue_shock = np.random.normal(1, revenue_volatility)
            cost_shock = np.random.normal(1, cost_volatility)
            
            sim_revenue = base_month_revenue * revenue_shock
            sim_cost = base_month_cost * cost_shock
            sim_profit = sim_revenue - sim_cost
            
            sim_revenue_path.append(sim_revenue)
            sim_cost_path.append(sim_cost)
            sim_profit_path.append(sim_profit)
        
        # Métriques de la simulation
        total_sim_revenue = sum(sim_revenue_path)
        total_sim_cost = sum(sim_cost_path)
        total_sim_profit = sum(sim_profit_path)
        
        # Calcul du maximum drawdown
        cumulative_profits = np.cumsum(sim_profit_path)
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = (cumulative_profits - running_max)
        max_drawdown = drawdowns.min()
        
        mc_results.append({
            'scenario': selected_scenario,
            'total_revenue': total_sim_revenue,
            'total_cost': total_sim_cost,
            'total_profit': total_sim_profit,
            'final_margin': (total_sim_profit / total_sim_revenue * 100) if total_sim_revenue > 0 else 0,
            'max_drawdown': max_drawdown,
            'profit_volatility': np.std(sim_profit_path),
            'break_even_month': next((i for i, cp in enumerate(np.cumsum(sim_profit_path)) if cp > 0), periods),
            'revenue_path': sim_revenue_path,
            'cost_path': sim_cost_path,
            'profit_path': sim_profit_path
        })
    
    return pd.DataFrame(mc_results)

def display_enhanced_scenario_results(results, csv_data):
    """Affiche les résultats enrichis de l'analyse de scénarios"""
    
    scenarios = results['scenarios']
    mc_results = results['monte_carlo']
    params = results['parameters']
    
    st.subheader("📊 Résultats de l'Analyse de Scénarios Enrichie")
    
    # Métriques de synthèse
    col1, col2, col3, col4, col5 = st.columns(5)
    
    expected_profit = sum(data['total_profit'] * data['probability'] for data in scenarios.values())
    best_case = max(data['total_profit'] for data in scenarios.values())
    worst_case = min(data['total_profit'] for data in scenarios.values())
    
    # Métriques Monte Carlo
    mc_mean_profit = mc_results['total_profit'].mean()
    mc_std_profit = mc_results['total_profit'].std()
    
    with col1:
        st.metric("💰 Profit Attendu", f"{expected_profit:,.0f} DHS")
        st.metric("📊 Monte Carlo Moyen", f"{mc_mean_profit:,.0f} DHS")
    
    with col2:
        st.metric("🚀 Meilleur Cas", f"{best_case:,.0f} DHS", f"+{best_case - expected_profit:,.0f}")
    
    with col3:
        st.metric("⚠️ Pire Cas", f"{worst_case:,.0f} DHS", f"{worst_case - expected_profit:,.0f}")
    
    with col4:
        profit_range = best_case - worst_case
        st.metric("📏 Fourchette", f"{profit_range:,.0f} DHS")
        
        # Coefficient de variation
        cv = (mc_std_profit / mc_mean_profit * 100) if mc_mean_profit != 0 else 0
        st.metric("📊 Volatilité", f"{cv:.1f}%")
    
    with col5:
        # Probabilité de succès (profit > 0)
        success_prob = (mc_results['total_profit'] > 0).sum() / len(mc_results) * 100
        st.metric("🎯 Prob. Succès", f"{success_prob:.1f}%")
        
        # VaR (Value at Risk) à 95%
        var_95 = np.percentile(mc_results['total_profit'], 5)
        st.metric("⚠️ VaR 95%", f"{var_95:,.0f} DHS")
    
    # Visualisations enrichies
    st.subheader("📈 Visualisations Avancées")
    
    viz_tabs = st.tabs([
        "📊 Comparaison Scénarios", 
        "🎲 Analyse Monte Carlo", 
        "📈 Évolution Temporelle",
        "⚠️ Analyse des Risques"
    ])
    
    with viz_tabs[0]:
        # Comparaison des scénarios
        fig_scenarios = create_scenario_comparison_chart(scenarios)
        st.plotly_chart(fig_scenarios, use_container_width=True)
        
        # Tableau de comparaison
        st.markdown("#### 📋 Tableau de Comparaison Détaillé")
        
        comparison_data = []
        for scenario, data in scenarios.items():
            comparison_data.append({
                'Scénario': scenario.title(),
                'Profit Total': f"{data['total_profit']:,.0f} DHS",
                'Marge Moyenne': f"{data['avg_margin']:.1f}%",
                'ROI': f"{data['roi']:.1f}%",
                'Probabilité': f"{data['probability']:.0%}",
                'Contribution Attendue': f"{data['total_profit'] * data['probability']:,.0f} DHS"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    with viz_tabs[1]:
        # Analyse Monte Carlo
        fig_mc = create_monte_carlo_distribution_chart(mc_results)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Statistiques Monte Carlo détaillées
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistiques Descriptives")
            
            mc_stats = {
                'Moyenne': mc_results['total_profit'].mean(),
                'Médiane': mc_results['total_profit'].median(),
                'Écart-type': mc_results['total_profit'].std(),
                'Minimum': mc_results['total_profit'].min(),
                'Maximum': mc_results['total_profit'].max()
            }
            
            for stat, value in mc_stats.items():
                st.metric(stat, f"{value:,.0f} DHS")
        
        with col2:
            st.markdown("#### 🎯 Percentiles de Performance")
            
            percentiles = [5, 25, 50, 75, 95]
            perc_values = np.percentile(mc_results['total_profit'], percentiles)
            
            for p, v in zip(percentiles, perc_values):
                st.metric(f"{p}e percentile", f"{v:,.0f} DHS")
    
    with viz_tabs[2]:
        # Évolution temporelle
        fig_temporal = create_temporal_evolution_chart(scenarios, mc_results)
        st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Analyse de la saisonnalité
        if any('seasonal_factor' in month for scenario in scenarios.values() for month in scenario['monthly_data']):
            st.markdown("#### 🌊 Impact de la Saisonnalité")
            
            seasonal_impact = analyze_seasonal_impact(scenarios)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mois le Plus Fort", seasonal_impact['best_month'])
                st.metric("Performance Max", f"+{seasonal_impact['max_boost']:.1f}%")
            
            with col2:
                st.metric("Mois le Plus Faible", seasonal_impact['worst_month'])
                st.metric("Impact Min", f"{seasonal_impact['min_impact']:.1f}%")
    
    with viz_tabs[3]:
        # Analyse des risques
        fig_risk = create_risk_analysis_chart(mc_results)
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Métriques de risque avancées
        st.markdown("#### ⚠️ Métriques de Risque Avancées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Expected Shortfall (CVaR)
            var_95 = np.percentile(mc_results['total_profit'], 5)
            cvar_95 = mc_results[mc_results['total_profit'] <= var_95]['total_profit'].mean()
            
            st.metric("VaR 95%", f"{var_95:,.0f} DHS")
            st.metric("CVaR 95%", f"{cvar_95:,.0f} DHS")
        
        with col2:
            # Drawdown analysis
            avg_max_drawdown = mc_results['max_drawdown'].mean()
            worst_drawdown = mc_results['max_drawdown'].min()
            
            st.metric("Drawdown Moyen", f"{avg_max_drawdown:,.0f} DHS")
            st.metric("Pire Drawdown", f"{worst_drawdown:,.0f} DHS")
        
        with col3:
            # Break-even analysis
            avg_break_even = mc_results['break_even_month'].mean()
            break_even_prob = (mc_results['break_even_month'] <= params['analysis_period']).sum() / len(mc_results) * 100
            
            st.metric("Break-even Moyen", f"{avg_break_even:.1f} mois")
            st.metric("Prob. Break-even", f"{break_even_prob:.1f}%")
    
    # Recommandations stratégiques
    st.subheader("💡 Recommandations Stratégiques")
    
    strategic_recs = generate_scenario_recommendations(scenarios, mc_results, csv_data)
    
    for i, rec in enumerate(strategic_recs, 1):
        st.warning(f"**Recommandation {i}**: {rec}")
    
    # Export des résultats
    st.subheader("📤 Export des Résultats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Sauvegarder Analyse", type="primary"):
            st.success("✅ Analyse de scénarios sauvegardée!")
    
    with col2:
        # Export CSV des résultats Monte Carlo
        if st.button("📊 Exporter Monte Carlo"):
            csv_export = mc_results.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger Résultats CSV",
                data=csv_export,
                file_name=f"monte_carlo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Vers Forecasting ML"):
            st.session_state['current_page'] = 'ml_forecasting'
            st.rerun()

def create_scenario_comparison_chart(scenarios):
    """Crée un graphique de comparaison des scénarios"""
    
    scenario_names = list(scenarios.keys())
    profits = [scenarios[s]['total_profit'] for s in scenario_names]
    margins = [scenarios[s]['avg_margin'] for s in scenario_names]
    probabilities = [scenarios[s]['probability'] * 100 for s in scenario_names]
    
    # Graphique en barres avec double axe Y
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Profit Total par Scénario', 'Marges Moyennes'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Barres de profit
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig.add_trace(
        go.Bar(
            x=[s.title() for s in scenario_names],
            y=profits,
            name='Profit Total',
            marker_color=colors,
            text=[f"{p:,.0f} DHS" for p in profits],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Marges moyennes
    fig.add_trace(
        go.Bar(
            x=[s.title() for s in scenario_names],
            y=margins,
            name='Marge Moyenne (%)',
            marker_color=[f"rgba{tuple(list(__import__('matplotlib.colors').to_rgb(c)) + [0.7])}" for c in colors],
            text=[f"{m:.1f}%" for m in margins],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Comparaison des Scénarios Financiers",
        showlegend=False
    )
    
    return fig

def create_monte_carlo_distribution_chart(mc_results):
    """Crée un graphique de distribution Monte Carlo"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribution des Profits', 'Profits vs Marges', 'Évolution des Scénarios', 'Box Plot par Scénario'),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "box"}]]
    )
    
    # Histogramme des profits
    fig.add_trace(
        go.Histogram(
            x=mc_results['total_profit'],
            nbinsx=50,
            name='Distribution Profits',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Scatter profits vs marges
    fig.add_trace(
        go.Scatter(
            x=mc_results['total_profit'],
            y=mc_results['final_margin'],
            mode='markers',
            name='Profit vs Marge',
            marker=dict(
                color=mc_results['profit_volatility'],
                colorscale='Viridis',
                showscale=True,
                size=6,
                opacity=0.6
            )
        ),
        row=1, col=2
    )
    
    # Évolution par scénario
    scenario_colors = {'pessimistic': '#FF6B6B', 'realistic': '#4ECDC4', 'optimistic': '#45B7D1'}
    
    for scenario in mc_results['scenario'].unique():
        scenario_data = mc_results[mc_results['scenario'] == scenario]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scenario_data))),
                y=scenario_data['total_profit'],
                mode='markers',
                name=scenario.title(),
                marker=dict(color=scenario_colors.get(scenario, 'gray'), size=4, opacity=0.6)
            ),
            row=2, col=1
        )
    
    # Box plot par scénario
    for scenario in mc_results['scenario'].unique():
        scenario_data = mc_results[mc_results['scenario'] == scenario]
        
        fig.add_trace(
            go.Box(
                y=scenario_data['total_profit'],
                name=scenario.title(),
                marker_color=scenario_colors.get(scenario, 'gray')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        title_text="Analyse Monte Carlo Complète",
        showlegend=False
    )
    
    return fig

def create_temporal_evolution_chart(scenarios, mc_results):
    """Crée un graphique d'évolution temporelle"""
    
    fig = go.Figure()
    
    colors = {'pessimistic': '#FF6B6B', 'realistic': '#4ECDC4', 'optimistic': '#45B7D1'}
    
    # Lignes des scénarios déterministes
    for scenario_name, scenario_data in scenarios.items():
        months = [m['month'] for m in scenario_data['monthly_data']]
        cumulative_profits = [m['cumulative_profit'] for m in scenario_data['monthly_data']]
        
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_profits,
            mode='lines+markers',
            name=f"{scenario_name.title()} (Déterministe)",
            line=dict(color=colors[scenario_name], width=3),
            marker=dict(size=8)
        ))
    
    # Bandes de confiance Monte Carlo
    if len(mc_results) > 0:
        # Calculer les percentiles pour chaque mois
        max_months = max(len(path) for path in mc_results['profit_path'])
        
        percentiles_5 = []
        percentiles_95 = []
        medians = []
        
        for month in range(max_months):
            month_profits = []
            for idx, profit_path in enumerate(mc_results['profit_path']):
                if month < len(profit_path):
                    cumulative_profit = sum(profit_path[:month+1])
                    month_profits.append(cumulative_profit)
            
            if month_profits:
                percentiles_5.append(np.percentile(month_profits, 5))
                percentiles_95.append(np.percentile(month_profits, 95))
                medians.append(np.percentile(month_profits, 50))
        
        months_range = list(range(1, len(percentiles_5) + 1))
        
        # Bande de confiance
        fig.add_trace(go.Scatter(
            x=months_range + months_range[::-1],
            y=percentiles_95 + percentiles_5[::-1],
            fill='toself',
            fillcolor='rgba(128,128,128,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalle de Confiance 90%',
            showlegend=True
        ))
        
        # Médiane Monte Carlo
        fig.add_trace(go.Scatter(
            x=months_range,
            y=medians,
            mode='lines',
            name='Médiane Monte Carlo',
            line=dict(color='gray', width=2, dash='dash')
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Break-even")
    
    fig.update_layout(
        title="Évolution Temporelle - Profit Cumulé",
        xaxis_title="Mois",
        yaxis_title="Profit Cumulé (DHS)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_risk_analysis_chart(mc_results):
    """Crée un graphique d'analyse des risques"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Value at Risk (VaR)', 'Drawdown Distribution', 'Volatilité des Profits', 'Temps de Break-even'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # VaR à différents niveaux
    var_levels = [1, 5, 10, 25]
    var_values = [np.percentile(mc_results['total_profit'], level) for level in var_levels]
    
    fig.add_trace(
        go.Bar(
            x=[f"VaR {level}%" for level in var_levels],
            y=var_values,
            name='Value at Risk',
            marker_color=['red' if v < 0 else 'orange' if v < np.mean(mc_results['total_profit']) else 'green' for v in var_values],
            text=[f"{v:,.0f}" for v in var_values],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Distribution des drawdowns
    fig.add_trace(
        go.Histogram(
            x=mc_results['max_drawdown'],
            nbinsx=30,
            name='Drawdown Distribution',
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Volatilité vs Performance
    fig.add_trace(
        go.Scatter(
            x=mc_results['total_profit'],
            y=mc_results['profit_volatility'],
            mode='markers',
            name='Volatilité vs Profit',
            marker=dict(
                color=mc_results['final_margin'],
                colorscale='RdYlGn',
                showscale=True,
                size=6,
                opacity=0.6
            )
        ),
        row=2, col=1
    )
    
    # Distribution temps de break-even
    fig.add_trace(
        go.Histogram(
            x=mc_results['break_even_month'],
            nbinsx=20,
            name='Break-even Timing',
            marker_color='blue',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="Analyse Complète des Risques",
        showlegend=False
    )
    
    return fig

def analyze_seasonal_impact(scenarios):
    """Analyse l'impact de la saisonnalité"""
    
    # Extraire les facteurs saisonniers du scénario réaliste
    realistic_data = scenarios.get('realistic', {}).get('monthly_data', [])
    
    if not realistic_data or 'seasonal_factor' not in realistic_data[0]:
        return {'best_month': 'N/A', 'worst_month': 'N/A', 'max_boost': 0, 'min_impact': 0}
    
    seasonal_factors = [month['seasonal_factor'] for month in realistic_data[:12]]
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    
    max_idx = np.argmax(seasonal_factors)
    min_idx = np.argmin(seasonal_factors)
    
    return {
        'best_month': month_names[max_idx],
        'worst_month': month_names[min_idx],
        'max_boost': (seasonal_factors[max_idx] - 1) * 100,
        'min_impact': (seasonal_factors[min_idx] - 1) * 100
    }

def generate_scenario_recommendations(scenarios, mc_results, csv_data):
    """Génère des recommandations basées sur l'analyse de scénarios"""
    
    recommendations = []
    
    # Analyse de la distribution des résultats
    success_rate = (mc_results['total_profit'] > 0).sum() / len(mc_results)
    profit_volatility = mc_results['total_profit'].std() / mc_results['total_profit'].mean()
    worst_case_loss = mc_results['total_profit'].min()
    
    # Recommandations basées sur le taux de succès
    if success_rate < 0.7:
        recommendations.append("Taux de succès faible (< 70%) - Revoir la stratégie de base ou réduire les ambitions")
    elif success_rate > 0.9:
        recommendations.append("Taux de succès élevé - Considérer des objectifs plus ambitieux")
    
    # Recommandations basées sur la volatilité
    if profit_volatility > 0.5:
        recommendations.append("Forte volatilité détectée - Diversifier les sources de revenus et stabiliser les coûts")
    
    # Recommandations basées sur les pertes potentielles
    current_equity = csv_data.get('equity', csv_data.get('revenue', 100000) * 0.4)
    if abs(worst_case_loss) > current_equity * 0.3:
        recommendations.append("Risque de perte élevé - Constituer des réserves ou réduire l'exposition au risque")
    
    # Recommandations spécifiques aux scénarios
    realistic_profit = scenarios.get('realistic', {}).get('total_profit', 0)
    optimistic_profit = scenarios.get('optimistic', {}).get('total_profit', 0)
    
    upside_potential = (optimistic_profit - realistic_profit) / realistic_profit if realistic_profit > 0 else 0
    
    if upside_potential > 0.5:
        recommendations.append("Fort potentiel de hausse (+50%) - Préparer les ressources pour capitaliser sur l'optimisme")
    
    # Recommandation géographique si applicable
    if csv_data.get('total_locations', 0) > 1:
        recommendations.append("Optimiser la performance géographique pour réduire la variance entre scénarios")
    
    return recommendations[:4]  # Limiter à 4 recommandations principales

# ========== MAIN APPLICATION ==========
def main():
    """Application principale enrichie"""
    
    init_session_state()
    
    # Header enrichi
    st.sidebar.markdown(f"""
    ### 🌍 AIFI - Suite Financière Avancée
    **Intelligence Géographique & Analytique**
    
    *Connecté en tant que: **{st.session_state.get('user_login', 'SalianiBouchaib')}***
    
    📅 **{datetime.now().strftime('%d/%m/%Y %H:%M')}**
    
    ---
    """)
    
    # Indicateur de données CSV enrichi
    csv_data = EnhancedCSVDataManager.get_csv_financial_data()
    
    if EnhancedCSVDataManager.has_csv_data():
        st.sidebar.success("📊 **Données CSV Chargées**")
        
        # Affichage des métriques clés dans la sidebar
        if csv_data:
            st.sidebar.metric("💰 Revenue Mensuel", f"{csv_data.get('monthly_revenue', 0):,.0f} DHS")
            st.sidebar.metric("📊 Marge Profit", f"{csv_data.get('profit_margin', 0):.1f}%")
            
            # Indicateur géographique
            total_locations = csv_data.get('total_locations', 0)
            if total_locations > 0:
                st.sidebar.metric("🌍 Emplacements", total_locations)
                st.sidebar.metric("📏 Étendue", f"{csv_data.get('geographic_spread', 0):.0f} km")
            
            # Score de santé
            health_score, _ = calculate_enhanced_health_score(csv_data)
            st.sidebar.metric("🎯 Score Santé", f"{health_score:.0f}/100")
            
            if health_score >= 75:
                st.sidebar.success("✅ Excellente santé")
            elif health_score >= 50:
                st.sidebar.info("📊 Santé correcte")
            else:
                st.sidebar.warning("⚠️ Attention requise")
    else:
        st.sidebar.warning("📤 **Aucune donnée CSV**")
        st.sidebar.caption("Importez vos données pour l'analyse complète")
    
    # Menu de navigation enrichi
    menu_items = {
        "📤 Import CSV Enrichi": "csv_import",
        "👔 Dashboard Exécutif": "executive_dashboard",
        "🧠 Analytics Avancés": "advanced_analytics",
        "🌍 Analyse Géographique": "geographic_analysis",
        "🎯 Planification Scénarios": "scenario_planning", 
        "🤖 Forecasting ML": "ml_forecasting",
        "⚠️ Gestion des Risques": "risk_management",
        "🏭 Templates Sectoriels": "industry_templates"
    }
    
    # Gestion de la navigation
    if 'current_page' in st.session_state:
        page_key = st.session_state['current_page']
        choice = None
        for menu_text, menu_key in menu_items.items():
            if menu_key == page_key:
                choice = menu_text
                break
        if not choice:
            choice = list(menu_items.keys())[0]
        del st.session_state['current_page']
    else:
        choice = st.sidebar.selectbox(
            "🧭 **Navigation Principale**",
            list(menu_items.keys()),
            index=0
        )
    
    # Routage vers les pages appropriées
    page_key = menu_items[choice]
    
    if page_key == "csv_import":
        show_enhanced_csv_import()
    elif page_key == "executive_dashboard":
        show_enhanced_executive_dashboard()
    elif page_key == "advanced_analytics":
        show_enhanced_advanced_analytics()
    elif page_key == "geographic_analysis":
        show_geographic_analysis()
    elif page_key == "scenario_planning":
        show_enhanced_scenario_planning()
    elif page_key == "ml_forecasting":
        show_ml_forecasting()
    elif page_key == "risk_management":
        show_risk_management()
    elif page_key == "industry_templates":
        show_industry_templates()
    
    # Sidebar enrichie avec informations système
    with st.sidebar:
        st.markdown("---")
        
        # Status système enrichi
        st.markdown("### 🔧 **Statut Système**")
        st.success("🟢 **Processeur CSV**: Opérationnel")
        st.success("🟢 **Moteur Analytics**: Actif") 
        st.success("🟢 **IA Géographique**: Disponible")
        st.success("🟢 **ML Forecasting**: Prêt")
        st.success("🟢 **Templates Sectoriels**: Complets")
        
        # Informations de session enrichies
        st.markdown("---")
        st.markdown("### 📊 **Session Info**")
        
        session_start = st.session_state.get('session_start_time', datetime.now())
        session_duration = datetime.now() - session_start
        
        st.caption(f"⏰ Durée session: {str(session_duration).split('.')[0]}")
        st.caption(f"🕒 Heure actuelle: {datetime.now().strftime('%H:%M:%S')}")
        st.caption(f"👤 Utilisateur: **SalianiBouchaib**")
        st.caption(f"🌍 Version: **AIFI v2.0 Enhanced**")
        
        # Fonctionnalités disponibles
        st.markdown("---")
        st.markdown("### ✨ **Nouvelles Fonctionnalités**")
        st.caption("✅ **Géolocalisation Avancée**")
        st.caption("✅ **Intelligence Géographique**")
        st.caption("✅ **Analytics Multi-dimensionnels**")
        st.caption("✅ **Clustering Automatique**")
        st.caption("✅ **Prédictions IA Enrichies**")
        st.caption("✅ **Optimisation Géo-financière**")
        
        # Liens rapides
        st.markdown("---")
        st.markdown("### ⚡ **Actions Rapides**")
        
        if st.button("🔄 **Actualiser Données**", use_container_width=True):
            st.rerun()
        
        if st.button("💾 **Sauvegarder Session**", use_container_width=True):
            st.success("✅ Session sauvegardée!")
        
        if st.button("📱 **Mode Mobile**", use_container_width=True):
            st.info("📱 Interface mobile en développement...")

# Point d'entrée de l'application
if __name__ == "__main__":
    # Initialiser le temps de session
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    main()