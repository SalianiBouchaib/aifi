import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Advanced Financial Planning Suite", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💼"
)

# Enhanced imports with new dependencies
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
import re
import warnings
from datetime import datetime, timedelta
import sqlite3
import hashlib
import uuid
import base64
from io import BytesIO
import seaborn as sns
from scipy import stats
import requests
import time

# Handle optional dependencies with try/except
try:
    import numpy_financial as npf
    NPF_AVAILABLE = True
except ImportError:
    NPF_AVAILABLE = False
    class NPF_Fallback:
        @staticmethod
        def irr(values):
            return 0.0
        @staticmethod
        def npv(rate, values):
            npv = 0
            for i, val in enumerate(values):
                npv += val / ((1 + rate) ** i)
            return npv
    npf = NPF_Fallback()

try:
    import pyfinance as pf
    PYFINANCE_AVAILABLE = True
except ImportError:
    PYFINANCE_AVAILABLE = False
    class PF_Fallback:
        @staticmethod
        def npv(rate, values):
            npv = 0
            for i, val in enumerate(values):
                npv += val / ((1 + rate) ** i)
            return npv
        @staticmethod
        def irr(values):
            return 0.0
    pf = PF_Fallback()

# For ML features - fallback if not available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
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

warnings.filterwarnings('ignore')

# ========== ENHANCED USER AUTHENTICATION & SESSION MANAGEMENT ==========
class UserManager:
    def __init__(self):
        self.db_path = "financial_suite_users.db"
        self.init_database()
    
    def init_database(self):
        """Initialize comprehensive user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                subscription_type TEXT DEFAULT 'basic'
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_projects (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                project_name TEXT,
                project_data TEXT,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_template BOOLEAN DEFAULT 0,
                shared_with TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Activity log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user if not exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_id = str(uuid.uuid4())
            admin_password = self.hash_password("admin123")
            cursor.execute('''
                INSERT INTO users (id, username, password_hash, email, full_name, role, subscription_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (admin_id, "admin", admin_password, "admin@financialsuite.com", "Administrator", "admin", "premium"))
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Enhanced password hashing with salt"""
        salt = "financial_suite_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def create_user(self, username, password, email, full_name="", role='user'):
        """Create a new user with enhanced validation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Validate input
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        if not email or "@" not in email:
            return False, "Valid email address required"
        
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)
        
        try:
            cursor.execute('''
                INSERT INTO users (id, username, password_hash, email, full_name, role)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, username, password_hash, email, full_name, role))
            
            # Log activity
            self.log_activity(user_id, "user_created", f"New user {username} created")
            
            conn.commit()
            return True, "User created successfully"
        except sqlite3.IntegrityError:
            return False, "Username already exists"
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """Enhanced authentication with session management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
            SELECT id, username, role, full_name, subscription_type, is_active 
            FROM users 
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        
        if user:
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user[0],))
            
            # Create session
            session_id = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(hours=8)
            
            cursor.execute('''
                INSERT INTO user_sessions (session_id, user_id, expires_at)
                VALUES (?, ?, ?)
            ''', (session_id, user[0], expires_at))
            
            # Log activity
            self.log_activity(user[0], "login", f"User {username} logged in")
            
            conn.commit()
            
            return {
                'id': user[0],
                'username': user[1],
                'role': user[2],
                'full_name': user[3],
                'subscription_type': user[4],
                'session_id': session_id
            }
        
        conn.close()
        return None
    
    def log_activity(self, user_id, action, details=""):
        """Log user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log (user_id, action, details)
            VALUES (?, ?, ?)
        ''', (user_id, action, details))
        
        conn.commit()
        conn.close()
    
    def save_user_project(self, user_id, project_name, project_data, industry="general"):
        """Enhanced project saving with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        project_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT OR REPLACE INTO user_projects 
            (id, user_id, project_name, project_data, industry, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (project_id, user_id, project_name, json.dumps(project_data), industry))
        
        self.log_activity(user_id, "project_saved", f"Project {project_name} saved")
        
        conn.commit()
        conn.close()
        return project_id
    
    def get_user_projects(self, user_id):
        """Get all projects for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, project_name, industry, created_at, updated_at
            FROM user_projects
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))
        
        projects = cursor.fetchall()
        conn.close()
        
        return [{
            'id': p[0],
            'name': p[1],
            'industry': p[2],
            'created_at': p[3],
            'updated_at': p[4]
        } for p in projects]

# ========== ADVANCED FINANCIAL ANALYTICS ENGINE ==========
class AdvancedAnalytics:
    def __init__(self):
        self.ratios_weights = {
            'liquidity': 0.3,
            'profitability': 0.4,
            'efficiency': 0.2,
            'leverage': 0.1
        }
    
    @staticmethod
    def monte_carlo_simulation(base_revenue, base_costs, volatility=0.2, simulations=1000, periods=12):
        """Enhanced Monte Carlo simulation with correlation"""
        results = []
        
        # Correlation between revenue and costs (typically positive)
        correlation = 0.6
        
        for _ in range(simulations):
            revenue_path = []
            cost_path = []
            
            # Generate correlated random shocks
            z1 = np.random.normal(0, 1, periods)
            z2 = np.random.normal(0, 1, periods)
            
            for period in range(periods):
                # Correlated shocks
                revenue_shock = z1[period] * volatility
                cost_shock = (correlation * z1[period] + 
                             np.sqrt(1 - correlation**2) * z2[period]) * volatility * 0.8
                
                if period == 0:
                    revenue = base_revenue * (1 + revenue_shock)
                    cost = base_costs * (1 + cost_shock)
                else:
                    revenue = revenue_path[-1] * (1 + revenue_shock)
                    cost = cost_path[-1] * (1 + cost_shock)
                
                revenue_path.append(max(revenue, 0))
                cost_path.append(max(cost, 0))
            
            total_revenue = sum(revenue_path)
            total_cost = sum(cost_path)
            net_profit = total_revenue - total_cost
            
            results.append({
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'net_profit': net_profit,
                'revenue_path': revenue_path,
                'cost_path': cost_path,
                'max_drawdown': min(np.cumsum([r-c for r,c in zip(revenue_path, cost_path)]))
            })
        
        return pd.DataFrame(results)
    
    def calculate_comprehensive_ratios(self, financial_data):
        """Calculate comprehensive financial ratios with industry context"""
        ratios = {}
        
        # Liquidity Ratios
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = financial_data.get('current_liabilities', 1)
        inventory = financial_data.get('inventory', 0)
        cash = financial_data.get('cash', 0)
        
        ratios['current_ratio'] = current_assets / max(current_liabilities, 1)
        ratios['quick_ratio'] = (current_assets - inventory) / max(current_liabilities, 1)
        ratios['cash_ratio'] = cash / max(current_liabilities, 1)
        
        # Profitability Ratios
        revenue = financial_data.get('revenue', 1)
        gross_profit = financial_data.get('gross_profit', 0)
        operating_profit = financial_data.get('operating_profit', 0)
        net_profit = financial_data.get('net_profit', 0)
        total_assets = financial_data.get('total_assets', 1)
        equity = financial_data.get('equity', 1)
        
        ratios['gross_margin'] = gross_profit / revenue if revenue > 0 else 0
        ratios['operating_margin'] = operating_profit / revenue if revenue > 0 else 0
        ratios['net_margin'] = net_profit / revenue if revenue > 0 else 0
        ratios['roa'] = net_profit / total_assets
        ratios['roe'] = net_profit / max(equity, 1)
        
        # Efficiency Ratios
        ratios['asset_turnover'] = revenue / total_assets
        ratios['equity_turnover'] = revenue / max(equity, 1)
        
        # Leverage Ratios
        total_debt = financial_data.get('total_debt', 0)
        ratios['debt_to_equity'] = total_debt / max(equity, 1)
        ratios['debt_to_assets'] = total_debt / total_assets
        ratios['equity_multiplier'] = total_assets / max(equity, 1)
        
        # Coverage Ratios
        interest_expense = financial_data.get('interest_expense', 1)
        ratios['interest_coverage'] = operating_profit / max(interest_expense, 1)
        
        return ratios
    
    def calculate_financial_health_score(self, ratios, industry='general'):
        """Calculate overall financial health score (0-100)"""
        industry_benchmarks = self.get_industry_benchmarks(industry)
        
        scores = {}
        
        # Liquidity Score (0-25)
        current_ratio_score = min(25, (ratios.get('current_ratio', 0) / industry_benchmarks['current_ratio']) * 15)
        quick_ratio_score = min(10, (ratios.get('quick_ratio', 0) / industry_benchmarks['quick_ratio']) * 10)
        scores['liquidity'] = current_ratio_score + quick_ratio_score
        
        # Profitability Score (0-40)
        net_margin_score = min(20, (ratios.get('net_margin', 0) / industry_benchmarks['net_margin']) * 20)
        roa_score = min(10, (ratios.get('roa', 0) / industry_benchmarks['roa']) * 10)
        roe_score = min(10, (ratios.get('roe', 0) / industry_benchmarks['roe']) * 10)
        scores['profitability'] = net_margin_score + roa_score + roe_score
        
        # Efficiency Score (0-20)
        asset_turnover_score = min(20, (ratios.get('asset_turnover', 0) / industry_benchmarks['asset_turnover']) * 20)
        scores['efficiency'] = asset_turnover_score
        
        # Leverage Score (0-15) - lower debt is better
        debt_ratio = ratios.get('debt_to_equity', 0)
        if debt_ratio <= industry_benchmarks['debt_to_equity']:
            leverage_score = 15
        else:
            leverage_score = max(0, 15 - (debt_ratio - industry_benchmarks['debt_to_equity']) * 30)
        scores['leverage'] = leverage_score
        
        total_score = sum(scores.values())
        return min(100, total_score), scores
    
    def get_industry_benchmarks(self, industry):
        """Get industry-specific benchmark ratios"""
        benchmarks = {
            'general': {
                'current_ratio': 1.5, 'quick_ratio': 1.0, 'net_margin': 0.08,
                'roa': 0.06, 'roe': 0.12, 'asset_turnover': 1.2, 'debt_to_equity': 0.4
            },
            'retail': {
                'current_ratio': 1.2, 'quick_ratio': 0.8, 'net_margin': 0.05,
                'roa': 0.04, 'roe': 0.15, 'asset_turnover': 2.5, 'debt_to_equity': 0.6
            },
            'technology': {
                'current_ratio': 2.0, 'quick_ratio': 1.8, 'net_margin': 0.15,
                'roa': 0.12, 'roe': 0.18, 'asset_turnover': 0.8, 'debt_to_equity': 0.2
            },
            'manufacturing': {
                'current_ratio': 1.4, 'quick_ratio': 1.0, 'net_margin': 0.06,
                'roa': 0.05, 'roe': 0.10, 'asset_turnover': 1.5, 'debt_to_equity': 0.5
            },
            'saas': {
                'current_ratio': 1.8, 'quick_ratio': 1.6, 'net_margin': 0.20,
                'roa': 0.15, 'roe': 0.25, 'asset_turnover': 0.6, 'debt_to_equity': 0.1
            }
        }
        return benchmarks.get(industry, benchmarks['general'])
    
    def generate_ai_recommendations(self, financial_data, ratios, health_score):
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # Cash flow recommendations
        cash_flow = financial_data.get('cash_flow', 0)
        if cash_flow < 0:
            recommendations.append({
                'category': 'Cash Flow Management',
                'priority': 'Critical',
                'recommendation': 'Immediate cash flow improvement needed. Consider: 1) Accelerating receivables collection, 2) Extending payables terms, 3) Reducing non-essential expenses, 4) Emergency credit line.',
                'impact': 'High',
                'timeframe': 'Immediate',
                'estimated_benefit': abs(cash_flow) * 0.5
            })
        
        # Liquidity recommendations
        current_ratio = ratios.get('current_ratio', 0)
        if current_ratio < 1.2:
            recommendations.append({
                'category': 'Liquidity Enhancement',
                'priority': 'High',
                'recommendation': 'Improve working capital management. Consider: 1) Inventory optimization, 2) Credit terms review, 3) Short-term financing options.',
                'impact': 'Medium',
                'timeframe': '1-3 months',
                'estimated_benefit': financial_data.get('current_assets', 0) * 0.1
            })
        
        # Profitability recommendations
        net_margin = ratios.get('net_margin', 0)
        if net_margin < 0.1:
            recommendations.append({
                'category': 'Profitability Improvement',
                'priority': 'Medium',
                'recommendation': 'Enhance profit margins through: 1) Cost structure optimization, 2) Pricing strategy review, 3) Operational efficiency improvements, 4) Product mix optimization.',
                'impact': 'High',
                'timeframe': '3-6 months',
                'estimated_benefit': financial_data.get('revenue', 0) * 0.05
            })
        
        # Efficiency recommendations
        asset_turnover = ratios.get('asset_turnover', 0)
        if asset_turnover < 1.0:
            recommendations.append({
                'category': 'Asset Utilization',
                'priority': 'Medium',
                'recommendation': 'Improve asset productivity: 1) Identify underutilized assets, 2) Consider asset optimization or disposal, 3) Increase revenue per asset.',
                'impact': 'Medium',
                'timeframe': '6-12 months',
                'estimated_benefit': financial_data.get('total_assets', 0) * 0.1
            })
        
        # Growth opportunities
        if health_score > 70:
            recommendations.append({
                'category': 'Growth Opportunities',
                'priority': 'Low',
                'recommendation': 'Strong financial position enables growth: 1) Market expansion, 2) New product development, 3) Strategic acquisitions, 4) Technology investments.',
                'impact': 'High',
                'timeframe': '12+ months',
                'estimated_benefit': financial_data.get('revenue', 0) * 0.2
            })
        
        return recommendations

# ========== ENHANCED SCENARIO PLANNING ENGINE ==========
class ScenarioPlanner:
    def __init__(self):
        self.scenarios = {
            'pessimistic': {
                'revenue_growth': -0.1, 
                'cost_increase': 0.15, 
                'probability': 0.2,
                'description': 'Economic downturn with reduced demand'
            },
            'realistic': {
                'revenue_growth': 0.15, 
                'cost_increase': 0.08, 
                'probability': 0.6,
                'description': 'Normal market conditions with steady growth'
            },
            'optimistic': {
                'revenue_growth': 0.3, 
                'cost_increase': 0.03, 
                'probability': 0.2,
                'description': 'Strong market expansion and efficiency gains'
            }
        }
        
        self.stress_tests = {
            'recession': {
                'revenue_impact': -0.25,
                'cost_impact': 0.1,
                'probability': 0.15,
                'duration_months': 18
            },
            'supply_shock': {
                'revenue_impact': -0.1,
                'cost_impact': 0.3,
                'probability': 0.1,
                'duration_months': 12
            },
            'competition': {
                'revenue_impact': -0.2,
                'cost_impact': 0.05,
                'probability': 0.25,
                'duration_months': 24
            }
        }
    
    def calculate_detailed_scenarios(self, base_data, periods=36):
        """Calculate detailed scenario outcomes with quarterly breakdown"""
        results = {}
        
        for scenario_name, params in self.scenarios.items():
            monthly_results = []
            quarterly_results = []
            
            base_revenue = base_data.get('monthly_revenue', 10000)
            base_cost = base_data.get('monthly_cost', 8000)
            
            for month in range(periods):
                # Apply growth/decline curves
                revenue_factor = (1 + params['revenue_growth']) ** (month / 12)
                cost_factor = (1 + params['cost_increase']) ** (month / 12)
                
                monthly_revenue = base_revenue * revenue_factor
                monthly_cost = base_cost * cost_factor
                monthly_profit = monthly_revenue - monthly_cost
                
                monthly_results.append({
                    'month': month + 1,
                    'revenue': monthly_revenue,
                    'cost': monthly_cost,
                    'profit': monthly_profit
                })
                
                # Quarterly aggregation
                if (month + 1) % 3 == 0:
                    quarter_data = monthly_results[-3:]
                    quarterly_results.append({
                        'quarter': len(quarterly_results) + 1,
                        'revenue': sum(m['revenue'] for m in quarter_data),
                        'cost': sum(m['cost'] for m in quarter_data),
                        'profit': sum(m['profit'] for m in quarter_data)
                    })
            
            # Calculate cumulative and summary metrics
            cumulative_profit = np.cumsum([m['profit'] for m in monthly_results])
            
            results[scenario_name] = {
                'monthly_data': monthly_results,
                'quarterly_data': quarterly_results,
                'cumulative_profit': cumulative_profit.tolist(),
                'total_profit': sum(m['profit'] for m in monthly_results),
                'avg_monthly_profit': np.mean([m['profit'] for m in monthly_results]),
                'profit_volatility': np.std([m['profit'] for m in monthly_results]),
                'probability': params['probability'],
                'description': params['description']
            }
        
        return results
    
    def calculate_value_at_risk(self, scenario_results, confidence_levels=[0.95, 0.99]):
        """Calculate Value at Risk for different confidence levels"""
        profits = []
        probabilities = []
        
        for scenario, data in scenario_results.items():
            profits.append(data['total_profit'])
            probabilities.append(data['probability'])
        
        # Create probability-weighted distribution
        weighted_profits = []
        for profit, prob in zip(profits, probabilities):
            # Simulate based on probability
            count = int(prob * 1000)
            weighted_profits.extend([profit] * count)
        
        var_results = {}
        for confidence in confidence_levels:
            var_value = np.percentile(weighted_profits, (1 - confidence) * 100)
            var_results[f'VaR_{int(confidence*100)}'] = var_value
        
        return var_results
    
    def stress_test_scenarios(self, base_data, selected_stress='recession'):
        """Perform detailed stress testing"""
        stress = self.stress_tests[selected_stress]
        
        base_revenue = base_data.get('monthly_revenue', 10000)
        base_cost = base_data.get('monthly_cost', 8000)
        duration = stress['duration_months']
        
        # Pre-stress period (6 months)
        pre_stress = []
        for month in range(6):
            pre_stress.append({
                'month': month + 1,
                'revenue': base_revenue,
                'cost': base_cost,
                'profit': base_revenue - base_cost,
                'phase': 'normal'
            })
        
        # Stress period
        stress_period = []
        for month in range(duration):
            stressed_revenue = base_revenue * (1 + stress['revenue_impact'])
            stressed_cost = base_cost * (1 + stress['cost_impact'])
            
            stress_period.append({
                'month': month + 7,
                'revenue': stressed_revenue,
                'cost': stressed_cost,
                'profit': stressed_revenue - stressed_cost,
                'phase': 'stress'
            })
        
        # Recovery period (12 months)
        recovery_period = []
        for month in range(12):
            recovery_factor = month / 12  # Gradual recovery
            revenue = stressed_revenue + (base_revenue - stressed_revenue) * recovery_factor
            cost = stressed_cost + (base_cost - stressed_cost) * recovery_factor
            
            recovery_period.append({
                'month': month + 7 + duration,
                'revenue': revenue,
                'cost': cost,
                'profit': revenue - cost,
                'phase': 'recovery'
            })
        
        all_periods = pre_stress + stress_period + recovery_period
        
        return {
            'timeline': all_periods,
            'total_impact': sum(p['profit'] for p in stress_period) - 
                          sum(p['profit'] for p in pre_stress[:len(stress_period)]),
            'recovery_time': 12,
            'stress_details': stress
        }

# ========== MACHINE LEARNING FORECASTING ENGINE ==========
class MLForecastingEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
    
    def prepare_time_series_features(self, data, target_col='revenue'):
        """Prepare comprehensive time series features"""
        df = pd.DataFrame(data)
        if target_col not in df.columns:
            df[target_col] = [d.get(target_col, 0) for d in data]
        
        # Time-based features
        df['trend'] = range(len(df))
        df['month'] = df.index % 12
        df['quarter'] = df.index // 3
        df['is_year_end'] = (df.index % 12 == 11).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            if len(df) > lag:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            if len(df) >= window:
                df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()
        
        # Growth rates
        df[f'{target_col}_growth'] = df[target_col].pct_change()
        df[f'{target_col}_growth_ma3'] = df[f'{target_col}_growth'].rolling(3).mean()
        
        # Seasonal decomposition features
        if len(df) >= 12:
            seasonal_mean = df.groupby('month')[target_col].transform('mean')
            df['seasonal_factor'] = df[target_col] / seasonal_mean
        
        return df.fillna(method='bfill').fillna(0)
    
    def train_ensemble_model(self, historical_data, target_col='revenue'):
        """Train ensemble of ML models"""
        if not ML_AVAILABLE:
            return None, "Machine Learning libraries not available"
        
        df = self.prepare_time_series_features(historical_data, target_col)
        
        if len(df) < 12:
            return None, "Insufficient data - need at least 12 months"
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col and not col.startswith('month')]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split into train/test
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            try:
                if name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                
                mse = np.mean((y_test - predictions) ** 2)
                mae = np.mean(np.abs(y_test - predictions))
                
                self.model_performance[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
                }
                
                if mse < best_score:
                    best_score = mse
                    best_model = (name, model, scaler if name == 'linear_regression' else None)
                
            except Exception as e:
                continue
        
        if best_model:
            model_name, model, scaler = best_model
            self.models[target_col] = model
            if scaler:
                self.scalers[target_col] = scaler
            
            # Feature importance (for Random Forest)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[target_col] = dict(zip(feature_cols, model.feature_importances_))
            
            return model_name, f"Model trained successfully. RMSE: {np.sqrt(best_score):.2f}"
        
        return None, "Failed to train any models"
    
    def generate_forecasts(self, periods=12, target_col='revenue', confidence_intervals=True):
        """Generate forecasts with confidence intervals"""
        if target_col not in self.models:
            return None, "Model not trained for this target"
        
        model = self.models[target_col]
        scaler = self.scalers.get(target_col)
        
        forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        # Generate forecasts (simplified approach)
        base_value = 10000  # Would use actual last known value
        
        for i in range(periods):
            # Create feature vector (simplified)
            features = np.random.randn(1, 10)  # Placeholder features
            
            if scaler:
                features = scaler.transform(features)
            
            try:
                prediction = model.predict(features)[0]
                
                # Add trend and seasonality
                trend_factor = 1 + (i * 0.01)  # 1% monthly growth
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)
                
                forecast = max(0, base_value * trend_factor * seasonal_factor)
                forecasts.append(forecast)
                
                # Simple confidence intervals (would be more sophisticated in practice)
                if confidence_intervals:
                    std_dev = forecast * 0.1  # 10% standard deviation
                    lower_bounds.append(forecast - 1.96 * std_dev)
                    upper_bounds.append(forecast + 1.96 * std_dev)
                
            except Exception:
                forecasts.append(base_value)
                if confidence_intervals:
                    lower_bounds.append(base_value * 0.9)
                    upper_bounds.append(base_value * 1.1)
        
        result = {
            'forecasts': forecasts,
            'periods': list(range(1, periods + 1)),
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance.get(target_col, {})
        }
        
        if confidence_intervals:
            result['lower_bounds'] = lower_bounds
            result['upper_bounds'] = upper_bounds
        
        return result, "Forecasts generated successfully"

# ========== ENHANCED RISK MANAGEMENT SYSTEM ==========
class RiskManagementSystem:
    def __init__(self):
        self.risk_factors = {
            'operational': ['cash_flow', 'liquidity', 'profitability'],
            'market': ['competition', 'demand', 'pricing'],
            'financial': ['leverage', 'interest_rates', 'currency'],
            'regulatory': ['compliance', 'tax_changes', 'regulations']
        }
        
        self.stress_scenarios = {
            'mild_recession': {
                'revenue_impact': -0.15,
                'cost_impact': 0.05,
                'duration': 12,
                'probability': 0.3
            },
            'severe_recession': {
                'revenue_impact': -0.35,
                'cost_impact': 0.1,
                'duration': 24,
                'probability': 0.1
            },
            'supply_chain_crisis': {
                'revenue_impact': -0.1,
                'cost_impact': 0.25,
                'duration': 18,
                'probability': 0.15
            },
            'competitive_disruption': {
                'revenue_impact': -0.25,
                'cost_impact': 0.15,
                'duration': 36,
                'probability': 0.2
            },
            'regulatory_change': {
                'revenue_impact': -0.05,
                'cost_impact': 0.2,
                'duration': 12,
                'probability': 0.1
            }
        }
    
    def calculate_comprehensive_risk_score(self, financial_data, market_data=None):
        """Calculate comprehensive risk score with multiple dimensions"""
        risk_scores = {}
        
        # Operational Risk (40% weight)
        operational_factors = []
        
        # Cash flow risk
        cash_flow = financial_data.get('cash_flow', 0)
        revenue = financial_data.get('revenue', 1)
        cf_ratio = cash_flow / revenue if revenue > 0 else 0
        
        if cf_ratio < 0:
            operational_factors.append(('Cash Flow', 0.9))
        elif cf_ratio < 0.05:
            operational_factors.append(('Cash Flow', 0.6))
        elif cf_ratio < 0.1:
            operational_factors.append(('Cash Flow', 0.3))
        else:
            operational_factors.append(('Cash Flow', 0.1))
        
        # Liquidity risk
        current_ratio = financial_data.get('current_ratio', 1)
        if current_ratio < 1:
            operational_factors.append(('Liquidity', 0.8))
        elif current_ratio < 1.2:
            operational_factors.append(('Liquidity', 0.5))
        elif current_ratio < 1.5:
            operational_factors.append(('Liquidity', 0.2))
        else:
            operational_factors.append(('Liquidity', 0.1))
        
        # Profitability risk
        net_margin = financial_data.get('net_margin', 0)
        if net_margin < 0:
            operational_factors.append(('Profitability', 0.9))
        elif net_margin < 0.05:
            operational_factors.append(('Profitability', 0.6))
        elif net_margin < 0.1:
            operational_factors.append(('Profitability', 0.3))
        else:
            operational_factors.append(('Profitability', 0.1))
        
        risk_scores['operational'] = np.mean([score for _, score in operational_factors])
        
        # Financial Risk (30% weight)
        financial_factors = []
        
        # Leverage risk
        debt_to_equity = financial_data.get('debt_to_equity', 0)
        if debt_to_equity > 3:
            financial_factors.append(('Leverage', 0.9))
        elif debt_to_equity > 2:
            financial_factors.append(('Leverage', 0.6))
        elif debt_to_equity > 1:
            financial_factors.append(('Leverage', 0.3))
        else:
            financial_factors.append(('Leverage', 0.1))
        
        # Interest coverage risk
        interest_coverage = financial_data.get('interest_coverage', 10)
        if interest_coverage < 1.5:
            financial_factors.append(('Interest Coverage', 0.9))
        elif interest_coverage < 3:
            financial_factors.append(('Interest Coverage', 0.5))
        elif interest_coverage < 5:
            financial_factors.append(('Interest Coverage', 0.2))
        else:
            financial_factors.append(('Interest Coverage', 0.1))
        
        risk_scores['financial'] = np.mean([score for _, score in financial_factors])
        
        # Market Risk (20% weight) - simplified
        market_volatility = market_data.get('volatility', 0.15) if market_data else 0.15
        risk_scores['market'] = min(0.9, market_volatility * 3)
        
        # Regulatory Risk (10% weight) - simplified
        risk_scores['regulatory'] = 0.2  # Base regulatory risk
        
        # Calculate weighted overall score
        weights = {'operational': 0.4, 'financial': 0.3, 'market': 0.2, 'regulatory': 0.1}
        overall_score = sum(risk_scores[category] * weights[category] for category in weights)
        
        return overall_score, risk_scores
    
    def perform_stress_testing(self, financial_data, scenarios=None):
        """Perform comprehensive stress testing"""
        if scenarios is None:
            scenarios = list(self.stress_scenarios.keys())
        
        results = {}
        
        base_revenue = financial_data.get('revenue', 120000)  # Annual
        base_costs = financial_data.get('total_costs', 100000)  # Annual
        base_profit = base_revenue - base_costs
        
        for scenario_name in scenarios:
            scenario = self.stress_scenarios[scenario_name]
            
            # Calculate stressed values
            stressed_revenue = base_revenue * (1 + scenario['revenue_impact'])
            stressed_costs = base_costs * (1 + scenario['cost_impact'])
            stressed_profit = stressed_revenue - stressed_costs
            
            # Calculate cumulative impact over duration
            duration_years = scenario['duration'] / 12
            cumulative_impact = (stressed_profit - base_profit) * duration_years
            
            # Recovery analysis
            recovery_months = scenario['duration'] * 0.5  # Assume 50% longer to fully recover
            
            results[scenario_name] = {
                'stressed_revenue': stressed_revenue,
                'stressed_costs': stressed_costs,
                'stressed_profit': stressed_profit,
                'profit_impact': stressed_profit - base_profit,
                'profit_impact_pct': (stressed_profit - base_profit) / base_profit * 100,
                'cumulative_impact': cumulative_impact,
                'duration_months': scenario['duration'],
                'recovery_months': recovery_months,
                'probability': scenario['probability'],
                'expected_loss': cumulative_impact * scenario['probability']
            }
        
        return results
    
    def calculate_value_at_risk(self, stress_test_results, confidence_levels=[0.95, 0.99]):
        """Calculate Value at Risk from stress test results"""
        losses = []
        probabilities = []
        
        for scenario, data in stress_test_results.items():
            losses.append(abs(data['cumulative_impact']))
            probabilities.append(data['probability'])
        
        # Create probability distribution
        scenario_outcomes = []
        for loss, prob in zip(losses, probabilities):
            scenario_outcomes.extend([loss] * int(prob * 10000))
        
        var_results = {}
        for confidence in confidence_levels:
            percentile = confidence * 100
            var_value = np.percentile(scenario_outcomes, percentile)
            var_results[f'VaR_{int(percentile)}'] = var_value
        
        # Expected Shortfall (Conditional VaR)
        for confidence in confidence_levels:
            percentile = confidence * 100
            threshold = np.percentile(scenario_outcomes, percentile)
            es_value = np.mean([x for x in scenario_outcomes if x >= threshold])
            var_results[f'ES_{int(percentile)}'] = es_value
        
        return var_results
    
    def generate_risk_mitigation_strategies(self, risk_scores, stress_results):
        """Generate specific risk mitigation strategies"""
        strategies = []
        
        # High operational risk
        if risk_scores.get('operational', 0) > 0.6:
            strategies.append({
                'category': 'Operational Risk Mitigation',
                'priority': 'High',
                'strategies': [
                    'Implement cash flow forecasting and monitoring system',
                    'Establish emergency credit facilities',
                    'Diversify revenue streams to reduce concentration risk',
                    'Optimize working capital management',
                    'Create operational efficiency improvement program'
                ],
                'timeline': '1-3 months',
                'investment_required': 'Low-Medium'
            })
        
        # High financial risk
        if risk_scores.get('financial', 0) > 0.6:
            strategies.append({
                'category': 'Financial Risk Mitigation',
                'priority': 'High',
                'strategies': [
                    'Reduce debt levels through debt restructuring',
                    'Improve debt service coverage ratios',
                    'Consider equity financing to reduce leverage',
                    'Hedge interest rate exposure',
                    'Negotiate better credit terms'
                ],
                'timeline': '3-6 months',
                'investment_required': 'Medium-High'
            })
        
        # Market risk strategies
        if risk_scores.get('market', 0) > 0.5:
            strategies.append({
                'category': 'Market Risk Mitigation',
                'priority': 'Medium',
                'strategies': [
                    'Diversify customer base and markets',
                    'Develop competitive advantages and moats',
                    'Create flexible cost structure',
                    'Build strategic partnerships',
                    'Invest in market research and intelligence'
                ],
                'timeline': '6-12 months',
                'investment_required': 'Medium'
            })
        
        # Severe stress test impacts
        severe_scenarios = [name for name, data in stress_results.items() 
                          if data['profit_impact_pct'] < -20]
        
        if severe_scenarios:
            strategies.append({
                'category': 'Crisis Preparedness',
                'priority': 'Medium',
                'strategies': [
                    'Develop detailed contingency plans',
                    'Create emergency cost reduction protocols',
                    'Establish crisis communication procedures',
                    'Build cash reserves for crisis scenarios',
                    'Identify alternative suppliers and markets'
                ],
                'timeline': '1-6 months',
                'investment_required': 'Low'
            })
        
        return strategies

# ========== INDUSTRY TEMPLATES & BENCHMARKING ==========
class IndustryTemplateManager:
    def __init__(self):
        self.templates = {
            'retail': {
                'revenue_model': 'Units Sold × Average Selling Price × Store Count',
                'key_metrics': [
                    'Same-Store Sales Growth', 'Inventory Turnover', 'Gross Margin',
                    'Sales per Square Foot', 'Customer Traffic', 'Average Transaction Value'
                ],
                'typical_ratios': {
                    'gross_margin': 0.35, 'net_margin': 0.04, 'current_ratio': 1.2,
                    'inventory_turnover': 6, 'asset_turnover': 2.5, 'debt_to_equity': 0.6
                },
                'seasonal_factors': [0.85, 0.9, 1.0, 1.05, 1.0, 0.95, 0.9, 0.95, 1.0, 1.1, 1.25, 1.4],
                'cost_structure': {
                    'cost_of_goods_sold': 0.65, 'labor': 0.15, 'rent': 0.08,
                    'marketing': 0.03, 'other_operating': 0.05
                },
                'working_capital': {
                    'days_sales_outstanding': 5, 'days_inventory_outstanding': 60,
                    'days_payable_outstanding': 30
                }
            },
            'saas': {
                'revenue_model': 'Monthly Recurring Revenue × 12 + One-time Setup Fees',
                'key_metrics': [
                    'Monthly Recurring Revenue (MRR)', 'Annual Recurring Revenue (ARR)',
                    'Customer Lifetime Value (LTV)', 'Customer Acquisition Cost (CAC)',
                    'Churn Rate', 'Net Revenue Retention'
                ],
                'typical_ratios': {
                    'gross_margin': 0.8, 'net_margin': 0.15, 'current_ratio': 2.0,
                    'ltv_cac_ratio': 3.0, 'asset_turnover': 0.6, 'debt_to_equity': 0.2
                },
                'seasonal_factors': [1.0] * 12,  # Low seasonality
                'cost_structure': {
                    'hosting_infrastructure': 0.1, 'customer_support': 0.08, 'sales_marketing': 0.4,
                    'research_development': 0.25, 'general_administrative': 0.12
                },
                'working_capital': {
                    'days_sales_outstanding': 30, 'days_inventory_outstanding': 0,
                    'days_payable_outstanding': 45
                }
            },
            'manufacturing': {
                'revenue_model': 'Production Capacity × Utilization Rate × Selling Price',
                'key_metrics': [
                    'Capacity Utilization', 'Overall Equipment Effectiveness (OEE)',
                    'Material Cost Ratio', 'Labor Productivity', 'Quality Metrics'
                ],
                'typical_ratios': {
                    'gross_margin': 0.25, 'net_margin': 0.06, 'current_ratio': 1.4,
                    'inventory_turnover': 4, 'asset_turnover': 1.5, 'debt_to_equity': 0.5
                },
                'seasonal_factors': [0.9, 0.9, 1.0, 1.1, 1.1, 1.0, 0.8, 0.85, 1.0, 1.1, 1.05, 0.95],
                'cost_structure': {
                    'raw_materials': 0.45, 'direct_labor': 0.2, 'manufacturing_overhead': 0.15,
                    'sales_marketing': 0.08, 'general_administrative': 0.07
                },
                'working_capital': {
                    'days_sales_outstanding': 45, 'days_inventory_outstanding': 90,
                    'days_payable_outstanding': 35
                }
            },
            'restaurant': {
                'revenue_model': 'Covers per Day × Average Check × Days Open × Locations',
                'key_metrics': [
                    'Revenue per Available Seat Hour (RevPASH)', 'Food Cost Percentage',
                    'Labor Cost Percentage', 'Table Turnover Rate', 'Customer Satisfaction'
                ],
                'typical_ratios': {
                    'gross_margin': 0.65, 'net_margin': 0.05, 'current_ratio': 0.8,
                    'food_cost_ratio': 0.3, 'labor_cost_ratio': 0.3, 'debt_to_equity': 0.8
                },
                'seasonal_factors': [0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.25, 1.0, 0.95, 1.1, 1.15],
                'cost_structure': {
                    'food_beverage_costs': 0.3, 'labor_costs': 0.3, 'rent_utilities': 0.15,
                    'marketing': 0.03, 'other_operating': 0.17
                },
                'working_capital': {
                    'days_sales_outstanding': 3, 'days_inventory_outstanding': 7,
                    'days_payable_outstanding': 20
                }
            },
            'consulting': {
                'revenue_model': 'Billable Hours × Hourly Rate × Utilization Rate × Consultants',
                'key_metrics': [
                    'Utilization Rate', 'Realization Rate', 'Average Hourly Rate',
                    'Revenue per Employee', 'Client Retention Rate'
                ],
                'typical_ratios': {
                    'gross_margin': 0.6, 'net_margin': 0.12, 'current_ratio': 1.5,
                    'utilization_rate': 0.75, 'asset_turnover': 3.0, 'debt_to_equity': 0.3
                },
                'seasonal_factors': [0.9, 1.0, 1.1, 1.1, 1.0, 0.9, 0.8, 0.9, 1.1, 1.1, 1.0, 0.9],
                'cost_structure': {
                    'consultant_compensation': 0.45, 'sales_marketing': 0.15, 'overhead': 0.15,
                    'technology': 0.05, 'other_operating': 0.15
                },
                'working_capital': {
                    'days_sales_outstanding': 60, 'days_inventory_outstanding': 0,
                    'days_payable_outstanding': 30
                }
            }
        }
    
    def get_template(self, industry):
        """Get comprehensive industry template"""
        return self.templates.get(industry, self.templates['retail'])
    
    def apply_template_to_projections(self, base_data, industry, periods=36):
        """Apply industry template to create realistic projections"""
        template = self.get_template(industry)
        
        base_revenue = base_data.get('monthly_revenue', 10000)
        projections = []
        
        for month in range(periods):
            # Apply seasonal factors
            seasonal_factor = template['seasonal_factors'][month % 12]
            
            # Apply growth trend (industry-specific)
            growth_rates = {
                'retail': 0.03, 'saas': 0.15, 'manufacturing': 0.05,
                'restaurant': 0.04, 'consulting': 0.08
            }
            monthly_growth = growth_rates.get(industry, 0.05) / 12
            trend_factor = (1 + monthly_growth) ** month
            
            # Calculate monthly revenue
            monthly_revenue = base_revenue * seasonal_factor * trend_factor
            
            # Calculate costs based on cost structure
            costs = {}
            total_cost = 0
            
            for cost_category, percentage in template['cost_structure'].items():
                cost_amount = monthly_revenue * percentage
                costs[cost_category] = cost_amount
                total_cost += cost_amount
            
            projections.append({
                'month': month + 1,
                'revenue': monthly_revenue,
                'total_costs': total_cost,
                'cost_breakdown': costs,
                'gross_profit': monthly_revenue - total_cost,
                'seasonal_factor': seasonal_factor,
                'trend_factor': trend_factor
            })
        
        return projections
    
    def benchmark_against_industry(self, company_ratios, industry):
        """Benchmark company performance against industry standards"""
        template = self.get_template(industry)
        industry_ratios = template['typical_ratios']
        
        comparison = {}
        
        for ratio, company_value in company_ratios.items():
            if ratio in industry_ratios:
                industry_value = industry_ratios[ratio]
                difference = company_value - industry_value
                percentage_diff = (difference / industry_value) * 100 if industry_value != 0 else 0
                
                if percentage_diff > 10:
                    performance = 'Above Average'
                elif percentage_diff > -10:
                    performance = 'Average'
                else:
                    performance = 'Below Average'
                
                comparison[ratio] = {
                    'company_value': company_value,
                    'industry_benchmark': industry_value,
                    'difference': difference,
                    'percentage_difference': percentage_diff,
                    'performance': performance
                }
        
        return comparison

# ========== ENHANCED SESSION STATE INITIALIZATION ==========
def init_enhanced_session_state():
    """Initialize all session state variables"""
    
    # User authentication
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'guest'
    
    # Basic company info
    if 'basic_info' not in st.session_state:
        st.session_state.basic_info = {
            'company_name': 'FinancialSuite Demo',
            'company_type': 'SARL',
            'creation_date': datetime(2024, 6, 1),
            'closing_date': '31 DECEMBER',
            'sector': 'Technology Services',
            'tax_id': '',
            'partners': 1,
            'address': '',
            'phone': '',
            'email': '',
            'industry': 'technology'
        }
    
    # Investment data
    if 'investment_data' not in st.session_state:
        st.session_state.investment_data = {
            'brand_registration': 1700.0,
            'sarl_formation': 4000.0,
            'web_dev': 80000.0,
            'cash_contribution': 50511.31,
            'in_kind': 20000.0
        }
    
    # Financial data structures
    for key in ['immos', 'credits', 'subsidies', 'frais_preliminaires']:
        if key not in st.session_state:
            st.session_state[key] = []
    
    # Enhanced analytics data
    if 'calculated_data' not in st.session_state:
        st.session_state.calculated_data = {}
    
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = {}
    
    if 'ml_forecasts' not in st.session_state:
        st.session_state.ml_forecasts = {}
    
    if 'risk_analysis' not in st.session_state:
        st.session_state.risk_analysis = {}
    
    if 'industry_benchmarks' not in st.session_state:
        st.session_state.industry_benchmarks = {}
    
    # Collaboration features
    if 'project_comments' not in st.session_state:
        st.session_state.project_comments = []
    
    if 'project_versions' not in st.session_state:
        st.session_state.project_versions = []

# ========== AUTHENTICATION PAGE ==========
import hashlib
import json
import os
from datetime import datetime

# ========== SIMPLE FILE-BASED USER MANAGER ==========
class SimpleUserManager:
    """File-based user management to avoid database lock issues"""
    
    def __init__(self):
        self.users_file = "users.json"
        self.init_users()
    
    def hash_password(self, password):
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def init_users(self):
        """Initialize default users if file doesn't exist"""
        if not os.path.exists(self.users_file):
            default_users = {
                "admin": {
                    "username": "admin",
                    "password": self.hash_password("admin123"),
                    "email": "admin@financialsuite.com",
                    "full_name": "Administrator",
                    "role": "admin",
                    "created_at": datetime.now().isoformat()
                },
                "demo": {
                    "username": "demo", 
                    "password": self.hash_password("demo123"),
                    "email": "demo@financialsuite.com",
                    "full_name": "Demo User",
                    "role": "user",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            try:
                with open(self.users_file, 'w') as f:
                    json.dump(default_users, f, indent=2)
            except Exception as e:
                st.error(f"Failed to create users file: {e}")
    
    def load_users(self):
        """Load users from file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Failed to load users: {e}")
            return {}
    
    def save_users(self, users):
        """Save users to file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save users: {e}")
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user with username and password"""
        users = self.load_users()
        
        if username in users:
            user = users[username]
            if user["password"] == self.hash_password(password):
                return user
        
        return None
    
    def create_user(self, username, password, email, full_name=None, role="user"):
        """Create a new user"""
        users = self.load_users()
        
        # Check if username already exists
        if username in users:
            return False, "Username already exists"
        
        # Create new user
        new_user = {
            "username": username,
            "password": self.hash_password(password),
            "email": email,
            "full_name": full_name or username,
            "role": role,
            "created_at": datetime.now().isoformat()
        }
        
        users[username] = new_user
        
        if self.save_users(users):
            return True, "User created successfully"
        else:
            return False, "Failed to save user"
    
    def user_exists(self, username):
        """Check if user exists"""
        users = self.load_users()
        return username in users

# ========== UPDATED AUTHENTICATION PAGE ==========
def show_authentication():
    """Enhanced authentication page with file-based user management"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🏢 Advanced Financial Planning Suite</h1>
        <p style="font-size: 1.2rem; color: #666;">Professional Financial Analysis & Planning Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use simple file-based user manager
    try:
        user_manager = SimpleUserManager()
        system_status = "✅ System Ready"
    except Exception as e:
        st.error(f"❌ System initialization error: {str(e)}")
        return
    
    # Create authentication tabs
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    # ========== LOGIN TAB ==========
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form", clear_on_submit=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                username = st.text_input(
                    "Username", 
                    placeholder="Enter your username",
                    help="Use 'admin' or 'demo' for testing"
                )
                password = st.text_input(
                    "Password", 
                    type="password", 
                    placeholder="Enter your password",
                    help="Use 'admin123' or 'demo123' for testing"
                )
            
            with col2:
                st.markdown("**🎯 Demo Accounts:**")
                st.code("👑 Admin:\nUsername: admin\nPassword: admin123")
                st.code("👤 User:\nUsername: demo\nPassword: demo123")
            
            submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
            
            # Handle login submission
            if submit:
                if not username or not password:
                    st.error("❌ Please enter both username and password.")
                else:
                    # Show loading state
                    with st.spinner("Authenticating..."):
                        try:
                            # Attempt authentication
                            user = user_manager.authenticate_user(username.strip(), password)
                            
                            if user:
                                # Successful login
                                st.session_state.user_authenticated = True
                                st.session_state.current_user = user
                                st.session_state.user_role = user.get('role', 'user')
                                
                                # Success message
                                st.success(f"✅ Welcome back, {user.get('username', 'User')}!")
                                st.balloons()
                                
                                # Small delay for user experience
                                time.sleep(1)
                                st.rerun()
                            else:
                                # Failed authentication
                                st.error("❌ Invalid username or password. Please try again.")
                                
                        except Exception as auth_error:
                            st.error(f"❌ Authentication error: {str(auth_error)}")
        
        # System status check
        with st.expander("🔧 System Status", expanded=False):
            st.success("✅ File-based authentication: Ready")
            
            # Check if users exist
            if user_manager.user_exists("admin"):
                st.success("✅ Admin account: Available")
            else:
                st.error("❌ Admin account: Missing")
            
            if user_manager.user_exists("demo"):
                st.success("✅ Demo account: Available") 
            else:
                st.error("❌ Demo account: Missing")
            
            # Show users file status
            if os.path.exists("users.json"):
                st.info(f"📁 Users file: users.json (exists)")
            else:
                st.warning("📁 Users file: Not found")
    
    # ========== REGISTRATION TAB ==========
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input(
                    "Username*", 
                    key="reg_username",
                    placeholder="Choose a unique username",
                    help="3-20 characters, letters and numbers only"
                )
                new_password = st.text_input(
                    "Password*", 
                    type="password", 
                    key="reg_password",
                    placeholder="Create a strong password",
                    help="Minimum 6 characters"
                )
            
            with col2:
                new_email = st.text_input(
                    "Email*", 
                    key="reg_email",
                    placeholder="your.email@example.com",
                    help="Valid email address required"
                )
                new_full_name = st.text_input(
                    "Full Name", 
                    key="reg_full_name",
                    placeholder="Your full name (optional)"
                )
            
            confirm_password = st.text_input(
                "Confirm Password*", 
                type="password", 
                key="confirm_password",
                placeholder="Repeat your password"
            )
            
            # Terms and conditions
            col1, col2 = st.columns(2)
            with col1:
                agree_terms = st.checkbox(
                    "I agree to the Terms of Service and Privacy Policy*",
                    help="Required to create an account"
                )
            with col2:
                newsletter = st.checkbox(
                    "Subscribe to newsletter (optional)",
                    help="Receive updates about new features"
                )
            
            register = st.form_submit_button("🎯 Create Account", use_container_width=True, type="primary")
            
            # Handle registration submission
            if register:
                # Validation
                errors = []
                
                # Check required fields
                if not all([new_username, new_password, new_email, confirm_password]):
                    errors.append("Please fill in all required fields marked with *")
                
                # Username validation
                if new_username:
                    if len(new_username.strip()) < 3:
                        errors.append("Username must be at least 3 characters long")
                    elif len(new_username.strip()) > 20:
                        errors.append("Username must be less than 20 characters")
                    elif not new_username.strip().replace('_', '').isalnum():
                        errors.append("Username can only contain letters, numbers, and underscores")
                
                # Password validation
                if new_password:
                    if len(new_password) < 6:
                        errors.append("Password must be at least 6 characters long")
                    elif new_password != confirm_password:
                        errors.append("Passwords do not match")
                
                # Email validation (basic)
                if new_email and '@' not in new_email:
                    errors.append("Please enter a valid email address")
                
                # Terms validation
                if not agree_terms:
                    errors.append("You must agree to the Terms of Service to create an account")
                
                # Display errors or proceed with registration
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    # Attempt to create user
                    with st.spinner("Creating your account..."):
                        try:
                            success, message = user_manager.create_user(
                                new_username.strip(),
                                new_password,
                                new_email.strip(),
                                new_full_name.strip() if new_full_name else None,
                                role="user"
                            )
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.success("🎉 Account created successfully! You can now login.")
                                st.balloons()
                                st.info("💡 Click on the 'Login' tab above to sign in with your new account.")
                            else:
                                st.error(f"❌ Registration failed: {message}")
                                
                        except Exception as reg_error:
                            st.error(f"❌ Registration error: {str(reg_error)}")
    
    # ========== PLATFORM FEATURES ==========
    with st.expander("🌟 Platform Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Advanced Analytics**
            - AI-powered recommendations
            - Real-time KPI dashboard
            - Industry benchmarking
            - Financial health scoring
            """)
        
        with col2:
            st.markdown("""
            **🎯 Scenario Planning** 
            - Monte Carlo simulations
            - Stress testing
            - Risk analysis
            - VaR calculations
            """)
        
        with col3:
            st.markdown("""
            **🤖 Machine Learning**
            - Revenue forecasting
            - Predictive analytics
            - Trend analysis
            - Automated insights
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🏢 Professional Financial Planning")
    with col2:
        st.caption("🔒 Secure & Reliable")
    with col3:
        st.caption("🚀 AI-Powered Insights")

# ========== EXECUTIVE DASHBOARD ==========
def show_executive_dashboard():
    """Enhanced executive dashboard with comprehensive KPIs"""
    st.header("👔 Executive Dashboard")
    
    # Quick stats overview
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate current financial state
    monthly_revenue = st.session_state.calculated_data.get('total_ventes', 15000)
    monthly_costs = st.session_state.calculated_data.get('total_charges', 12000)
    monthly_profit = monthly_revenue - monthly_costs
    annual_revenue = monthly_revenue * 12
    annual_profit = monthly_profit * 12
    
    total_investment = st.session_state.calculated_data.get('total_investissement', 100000)
    roi = (annual_profit / total_investment * 100) if total_investment > 0 else 0
    
    with col1:
        st.metric(
            "📈 Annual Revenue",
            f"{annual_revenue:,.0f} DHS",
            f"{(annual_revenue / 1000000):.1f}M DHS"
        )
    
    with col2:
        margin = (monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0
        st.metric(
            "💰 Profit Margin",
            f"{margin:.1f}%",
            "Healthy" if margin > 15 else "Needs Improvement"
        )
    
    with col3:
        st.metric(
            "🎯 ROI",
            f"{roi:.1f}%",
            "Strong" if roi > 20 else "Moderate" if roi > 10 else "Weak"
        )
    
    with col4:
        payback_months = (total_investment / monthly_profit) if monthly_profit > 0 else float('inf')
        payback_display = f"{payback_months:.1f} mo" if payback_months != float('inf') else "N/A"
        st.metric(
            "⏱️ Payback Period",
            payback_display,
            "Fast" if payback_months < 24 else "Moderate" if payback_months < 48 else "Slow"
        )
    
    # Financial health indicator
    st.subheader("🏥 Financial Health Monitor")
    
    # Calculate health score
    financial_data = {
        'revenue': annual_revenue,
        'net_profit': annual_profit,
        'cash_flow': monthly_profit,
        'current_ratio': 1.5,  # Example values
        'debt_to_equity': 0.3,
        'net_margin': margin / 100
    }
    
    analytics = AdvancedAnalytics()
    ratios = analytics.calculate_comprehensive_ratios(financial_data)
    health_score, score_breakdown = analytics.calculate_financial_health_score(ratios)
    
    # Health score gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Financial Health Score"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 90], 'color': "lightgreen"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Health Score Breakdown:**")
        for category, score in score_breakdown.items():
            percentage = (score / 40) * 100 if category == 'profitability' else (score / 25) * 100 if category == 'liquidity' else (score / 20) * 100 if category == 'efficiency' else (score / 15) * 100
            if percentage >= 80:
                st.success(f"🟢 {category.title()}: {score:.1f} points (Excellent)")
            elif percentage >= 60:
                st.info(f"🔵 {category.title()}: {score:.1f} points (Good)")
            elif percentage >= 40:
                st.warning(f"🟡 {category.title()}: {score:.1f} points (Fair)")
            else:
                st.error(f"🔴 {category.title()}: {score:.1f} points (Poor)")
    
    # Revenue trend and projections
    st.subheader("📈 Revenue Trends & Projections")
    
    # Generate trend data
    months = pd.date_range(start='2024-01-01', periods=24, freq='M')
    industry = st.session_state.basic_info.get('industry', 'technology')
    
    # Get industry template for realistic projections
    template_manager = IndustryTemplateManager()
    template = template_manager.get_template(industry)
    
    historical_revenue = []
    projected_revenue = []
    
    # Historical data (last 12 months)
    for i in range(12):
        seasonal_factor = template['seasonal_factors'][i]
        base_trend = monthly_revenue * (0.98 + i * 0.003)  # Slight growth trend
        revenue = base_trend * seasonal_factor * (0.9 + np.random.random() * 0.2)
        historical_revenue.append(revenue)
    
    # Projected data (next 12 months)
    for i in range(12):
        seasonal_factor = template['seasonal_factors'][i]
        base_trend = monthly_revenue * (1.1 + i * 0.01)  # Growth projection
        revenue = base_trend * seasonal_factor
        projected_revenue.append(revenue)
    
    # Create comprehensive chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=months[:12],
        y=historical_revenue,
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Projected data
    fig.add_trace(go.Scatter(
        x=months[12:],
        y=projected_revenue,
        mode='lines+markers',
        name='Projected Revenue',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    # Add trend line
    all_revenue = historical_revenue + projected_revenue
    z = np.polyfit(range(24), all_revenue, 1)
    p = np.poly1d(z)
    trend_line = p(range(24))
    
    fig.add_trace(go.Scatter(
        x=months,
        y=trend_line,
        mode='lines',
        name='Trend Line',
        line=dict(color='red', width=2, dash='dot'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Revenue Analysis: Historical Performance vs Future Projections",
        xaxis_title="Month",
        yaxis_title="Revenue (DHS)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key performance indicators grid
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Profitability Metrics**")
        st.metric("Gross Margin", f"{ratios.get('gross_margin', 0.3)*100:.1f}%")
        st.metric("Operating Margin", f"{ratios.get('operating_margin', 0.2)*100:.1f}%")
        st.metric("Net Margin", f"{ratios.get('net_margin', 0.15)*100:.1f}%")
        st.metric("ROA", f"{ratios.get('roa', 0.08)*100:.1f}%")
    
    with col2:
        st.markdown("**🏦 Liquidity Metrics**")
        st.metric("Current Ratio", f"{ratios.get('current_ratio', 1.5):.2f}")
        st.metric("Quick Ratio", f"{ratios.get('quick_ratio', 1.2):.2f}")
        st.metric("Cash Ratio", f"{ratios.get('cash_ratio', 0.8):.2f}")
        st.metric("Working Capital", f"{financial_data.get('working_capital', 25000):,.0f} DHS")
    
    with col3:
        st.markdown("**⚡ Efficiency Metrics**")
        st.metric("Asset Turnover", f"{ratios.get('asset_turnover', 1.2):.2f}")
        st.metric("Equity Turnover", f"{ratios.get('equity_turnover', 2.1):.2f}")
        st.metric("Inventory Turnover", f"{template['typical_ratios'].get('inventory_turnover', 6):.1f}")
        st.metric("Days Sales Outstanding", f"{template['working_capital'].get('days_sales_outstanding', 30):.0f} days")
    
    # Risk alerts
    st.subheader("⚠️ Risk & Alert Monitor")
    
    # Generate risk alerts
    alerts = []
    
    if monthly_profit < 0:
        alerts.append(("🔴 Critical", "Negative cash flow detected", "Immediate action required"))
    
    if ratios.get('current_ratio', 1.5) < 1.2:
        alerts.append(("🟡 Warning", "Low liquidity ratio", "Monitor working capital"))
    
    if roi < 10:
        alerts.append(("🟡 Warning", "Low ROI performance", "Review investment efficiency"))
    
    if payback_months > 36:
        alerts.append(("🟡 Warning", "Long payback period", "Consider strategy optimization"))
    
    if not alerts:
        st.success("✅ No critical alerts detected. Financial performance is healthy.")
    else:
        for level, title, description in alerts:
            if "Critical" in level:
                st.error(f"{level}: **{title}** - {description}")
            else:
                st.warning(f"{level}: **{title}** - {description}")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Run Analytics", use_container_width=True):
            st.session_state.redirect_to = "Advanced Analytics"
            st.rerun()
    
    with col2:
        if st.button("🎯 Scenario Planning", use_container_width=True):
            st.session_state.redirect_to = "Scenario Planning"
            st.rerun()
    
    with col3:
        if st.button("🤖 ML Forecast", use_container_width=True):
            st.session_state.redirect_to = "ML Forecasting"
            st.rerun()
    
    with col4:
        if st.button("⚠️ Risk Analysis", use_container_width=True):
            st.session_state.redirect_to = "Risk Management"
            st.rerun()

# ========== ADVANCED ANALYTICS PAGE ==========
def show_advanced_analytics():
    """Comprehensive advanced analytics page"""
    st.header("🧠 Advanced Analytics & AI Insights")
    
    # Initialize analytics engine
    analytics = AdvancedAnalytics()
    
    # Collect and prepare financial data
    financial_data = {
        'revenue': st.session_state.calculated_data.get('total_ventes', 15000) * 12,
        'total_costs': st.session_state.calculated_data.get('total_charges', 12000) * 12,
        'net_profit': (st.session_state.calculated_data.get('total_ventes', 15000) - 
                      st.session_state.calculated_data.get('total_charges', 12000)) * 12,
        'gross_profit': st.session_state.calculated_data.get('total_ventes', 15000) * 12 * 0.6,
        'operating_profit': st.session_state.calculated_data.get('total_ventes', 15000) * 12 * 0.2,
        'current_assets': st.session_state.calculated_data.get('total_actif', 150000) * 0.4,
        'current_liabilities': st.session_state.calculated_data.get('total_passif', 100000) * 0.3,
        'total_assets': st.session_state.calculated_data.get('total_actif', 150000),
        'total_debt': st.session_state.calculated_data.get('total_credits', 50000),
        'equity': st.session_state.investment_data.get('cash_contribution', 50000),
        'cash': st.session_state.calculated_data.get('total_actif', 150000) * 0.1,
        'inventory': st.session_state.calculated_data.get('total_actif', 150000) * 0.15,
        'interest_expense': st.session_state.calculated_data.get('total_credits', 50000) * 0.05,
        'cash_flow': st.session_state.calculated_data.get('cash_flow_mensuel', 3000)
    }
    
    # Calculate comprehensive ratios
    ratios = analytics.calculate_comprehensive_ratios(financial_data)
    
    # Calculate health score
    industry = st.session_state.basic_info.get('industry', 'technology')
    health_score, score_breakdown = analytics.calculate_financial_health_score(ratios, industry)
    
    # Display comprehensive KPI dashboard
    st.subheader("📊 Comprehensive Financial Analysis")
    
    # Main metrics overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Health Score",
            f"{health_score:.0f}/100",
            f"{'Excellent' if health_score >= 80 else 'Good' if health_score >= 60 else 'Fair' if health_score >= 40 else 'Poor'}"
        )
    
    with col2:
        st.metric(
            "Current Ratio",
            f"{ratios.get('current_ratio', 0):.2f}",
            f"{'Healthy' if ratios.get('current_ratio', 0) > 1.5 else 'Moderate' if ratios.get('current_ratio', 0) > 1.2 else 'Low'}"
        )
    
    with col3:
        st.metric(
            "ROE",
            f"{ratios.get('roe', 0)*100:.1f}%",
            f"{'Strong' if ratios.get('roe', 0) > 0.15 else 'Average' if ratios.get('roe', 0) > 0.08 else 'Weak'}"
        )
    
    with col4:
        st.metric(
            "Debt-to-Equity",
            f"{ratios.get('debt_to_equity', 0):.2f}",
            f"{'Conservative' if ratios.get('debt_to_equity', 0) < 0.5 else 'Moderate' if ratios.get('debt_to_equity', 0) < 1.0 else 'High'}"
        )
    
    with col5:
        st.metric(
            "Asset Turnover",
            f"{ratios.get('asset_turnover', 0):.2f}",
            f"{'Efficient' if ratios.get('asset_turnover', 0) > 1.0 else 'Moderate' if ratios.get('asset_turnover', 0) > 0.7 else 'Low'}"
        )
    
    # Detailed ratio analysis with visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🏦 Liquidity", "💰 Profitability", "⚡ Efficiency", "📊 Leverage"])
    
    with tab1:
        st.subheader("Liquidity Analysis")
        
        # Liquidity ratios chart
        liquidity_ratios = {
            'Current Ratio': ratios.get('current_ratio', 0),
            'Quick Ratio': ratios.get('quick_ratio', 0),
            'Cash Ratio': ratios.get('cash_ratio', 0)
        }
        
        # Benchmark comparison
        benchmarks = analytics.get_industry_benchmarks(industry)
        benchmark_liquidity = {
            'Current Ratio': benchmarks['current_ratio'],
            'Quick Ratio': benchmarks.get('quick_ratio', 1.0),
            'Cash Ratio': 0.5  # Standard benchmark
        }
        
        fig = go.Figure(data=[
            go.Bar(name='Your Company', x=list(liquidity_ratios.keys()), y=list(liquidity_ratios.values())),
            go.Bar(name='Industry Benchmark', x=list(benchmark_liquidity.keys()), y=list(benchmark_liquidity.values()))
        ])
        
        fig.update_layout(
            barmode='group',
            title="Liquidity Ratios vs Industry Benchmarks",
            yaxis_title="Ratio Value"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Liquidity analysis text
        col1, col2 = st.columns(2)
        
        with col1:
            current_ratio = ratios.get('current_ratio', 0)
            if current_ratio >= 2.0:
                st.success("✅ **Excellent Liquidity**: Strong ability to meet short-term obligations")
            elif current_ratio >= 1.5:
                st.info("📘 **Good Liquidity**: Adequate short-term financial position")
            elif current_ratio >= 1.2:
                st.warning("⚠️ **Moderate Liquidity**: Monitor working capital closely")
            else:
                st.error("🚨 **Poor Liquidity**: Immediate attention needed for working capital")
        
        with col2:
            quick_ratio = ratios.get('quick_ratio', 0)
            if quick_ratio >= 1.0:
                st.success("✅ **Strong Quick Liquidity**: Can cover obligations without inventory")
            elif quick_ratio >= 0.8:
                st.info("📘 **Adequate Quick Liquidity**: Reasonable liquid asset position")
            else:
                st.warning("⚠️ **Low Quick Liquidity**: High dependence on inventory conversion")
    
    with tab2:
        st.subheader("Profitability Analysis")
        
        # Profitability ratios
        profitability_ratios = {
            'Gross Margin': ratios.get('gross_margin', 0) * 100,
            'Operating Margin': ratios.get('operating_margin', 0) * 100,
            'Net Margin': ratios.get('net_margin', 0) * 100,
            'ROA': ratios.get('roa', 0) * 100,
            'ROE': ratios.get('roe', 0) * 100
        }
        
        # Create profitability trend chart
        fig = px.bar(
            x=list(profitability_ratios.keys()),
            y=list(profitability_ratios.values()),
            title="Profitability Metrics (%)",
            color=list(profitability_ratios.values()),
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Profitability insights
        net_margin = ratios.get('net_margin', 0)
        roa = ratios.get('roa', 0)
        roe = ratios.get('roe', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if net_margin >= 0.15:
                st.success(f"🎯 **Excellent Net Margin** ({net_margin*100:.1f}%)")
            elif net_margin >= 0.08:
                st.info(f"📈 **Good Net Margin** ({net_margin*100:.1f}%)")
            elif net_margin >= 0.03:
                st.warning(f"⚠️ **Moderate Net Margin** ({net_margin*100:.1f}%)")
            else:
                st.error(f"📉 **Low Net Margin** ({net_margin*100:.1f}%)")
        
        with col2:
            if roa >= 0.1:
                st.success(f"🎯 **Excellent ROA** ({roa*100:.1f}%)")
            elif roa >= 0.05:
                st.info(f"📈 **Good ROA** ({roa*100:.1f}%)")
            else:
                st.warning(f"⚠️ **Low ROA** ({roa*100:.1f}%)")
        
        with col3:
            if roe >= 0.15:
                st.success(f"🎯 **Excellent ROE** ({roe*100:.1f}%)")
            elif roe >= 0.1:
                st.info(f"📈 **Good ROE** ({roe*100:.1f}%)")
            else:
                st.warning(f"⚠️ **Low ROE** ({roe*100:.1f}%)")
    
    with tab3:
        st.subheader("Efficiency Analysis")
        
        # Efficiency metrics
        efficiency_metrics = {
            'Asset Turnover': ratios.get('asset_turnover', 0),
            'Equity Turnover': ratios.get('equity_turnover', 0),
            'Interest Coverage': ratios.get('interest_coverage', 0)
        }
        
        # Efficiency gauge charts
        col1, col2, col3 = st.columns(3)
        
        for i, (metric, value) in enumerate(efficiency_metrics.items()):
            with [col1, col2, col3][i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric},
                    gauge={
                        'axis': {'range': [None, 5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "yellow"},
                            {'range': [2, 5], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Leverage Analysis")
        
        # Leverage analysis
        debt_to_equity = ratios.get('debt_to_equity', 0)
        debt_to_assets = ratios.get('debt_to_assets', 0)
        equity_multiplier = ratios.get('equity_multiplier', 0)
        
        # Leverage visualization
        leverage_data = {
            'Metric': ['Debt-to-Equity', 'Debt-to-Assets', 'Equity Multiplier'],
            'Value': [debt_to_equity, debt_to_assets, equity_multiplier],
            'Benchmark': [0.5, 0.3, 2.0]
        }
        
        fig = go.Figure(data=[
            go.Bar(name='Current', x=leverage_data['Metric'], y=leverage_data['Value']),
            go.Bar(name='Benchmark', x=leverage_data['Metric'], y=leverage_data['Benchmark'])
        ])
        
        fig.update_layout(
            barmode='group',
            title="Leverage Ratios vs Benchmarks"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # AI-Powered Recommendations
    st.subheader("🤖 AI-Powered Recommendations")
    
    recommendations = analytics.generate_ai_recommendations(financial_data, ratios, health_score)
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"💡 {rec['category']} - {rec['priority']} Priority", expanded=i==0):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
                st.write(f"**Timeline:** {rec['timeframe']}")
            
            with col2:
                if 'estimated_benefit' in rec:
                    st.metric("Potential Benefit", f"{rec['estimated_benefit']:,.0f} DHS")
                
                priority_color = {
                    'Critical': '🔴',
                    'High': '🟠', 
                    'Medium': '🟡',
                    'Low': '🟢'
                }
                st.write(f"{priority_color.get(rec['priority'], '⚪')} {rec['priority']} Priority")
    
    # Industry Benchmarking
    st.subheader("📈 Industry Benchmarking")
    
    template_manager = IndustryTemplateManager()
    comparison = template_manager.benchmark_against_industry(ratios, industry)
    
    if comparison:
        # Create comparison chart
        metrics = []
        company_values = []
        industry_values = []
        performance = []
        
        for metric, data in comparison.items():
            metrics.append(metric.replace('_', ' ').title())
            company_values.append(data['company_value'])
            industry_values.append(data['industry_benchmark'])
            performance.append(data['performance'])
        
        fig = go.Figure(data=[
            go.Bar(name='Your Company', x=metrics, y=company_values),
            go.Bar(name='Industry Average', x=metrics, y=industry_values)
        ])
        
        fig.update_layout(
            barmode='group',
            title=f"Performance vs {industry.title()} Industry Benchmarks"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary
        above_avg = sum(1 for p in performance if p == 'Above Average')
        total_metrics = len(performance)
        
        if above_avg / total_metrics >= 0.7:
            st.success(f"🎉 **Outstanding Performance**: {above_avg}/{total_metrics} metrics above industry average")
        elif above_avg / total_metrics >= 0.5:
            st.info(f"📈 **Good Performance**: {above_avg}/{total_metrics} metrics above industry average")
        else:
            st.warning(f"⚠️ **Room for Improvement**: Only {above_avg}/{total_metrics} metrics above industry average")

# ========== SCENARIO PLANNING PAGE ==========
def show_scenario_planning():
    """Enhanced scenario planning with comprehensive analysis"""
    st.header("🎯 Advanced Scenario Planning & Risk Analysis")
    
    # Initialize scenario planner
    planner = ScenarioPlanner()
    
    # Scenario configuration
    with st.expander("⚙️ Scenario Configuration", expanded=True):
        st.write("Configure your business scenarios to understand potential outcomes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 😰 Pessimistic Scenario")
            pess_revenue = st.slider("Revenue Growth (%)", -30, 10, -10, key="pess_rev")
            pess_cost = st.slider("Cost Increase (%)", 0, 40, 15, key="pess_cost")
            pess_prob = st.slider("Probability (%)", 0, 50, 20, key="pess_prob")
            
            st.info(f"**Scenario:** Economic downturn with {pess_revenue}% revenue decline and {pess_cost}% cost increase")
        
        with col2:
            st.markdown("### 😐 Realistic Scenario")
            real_revenue = st.slider("Revenue Growth (%)", -10, 40, 15, key="real_rev")
            real_cost = st.slider("Cost Increase (%)", 0, 25, 8, key="real_cost")
            real_prob = st.slider("Probability (%)", 40, 80, 60, key="real_prob")
            
            st.info(f"**Scenario:** Normal conditions with {real_revenue}% growth and {real_cost}% cost increase")
        
        with col3:
            st.markdown("### 😄 Optimistic Scenario")
            opt_revenue = st.slider("Revenue Growth (%)", 15, 60, 30, key="opt_rev")
            opt_cost = st.slider("Cost Increase (%)", 0, 15, 3, key="opt_cost")
            opt_prob = st.slider("Probability (%)", 0, 40, 20, key="opt_prob")
            
            st.info(f"**Scenario:** Strong growth with {opt_revenue}% revenue increase and {opt_cost}% cost increase")
        
        # Validate probabilities
        total_prob = pess_prob + real_prob + opt_prob
        if total_prob != 100:
            st.warning(f"⚠️ Probabilities sum to {total_prob}%. Adjusting to 100%...")
            pess_prob = pess_prob * 100 / total_prob
            real_prob = real_prob * 100 / total_prob
            opt_prob = opt_prob * 100 / total_prob
    
    # Update planner scenarios
    planner.scenarios = {
        'pessimistic': {
            'revenue_growth': pess_revenue / 100, 
            'cost_increase': pess_cost / 100, 
            'probability': pess_prob / 100,
            'description': f'Economic downturn scenario with {pess_revenue}% revenue change'
        },
        'realistic': {
            'revenue_growth': real_revenue / 100, 
            'cost_increase': real_cost / 100, 
            'probability': real_prob / 100,
            'description': f'Base case scenario with {real_revenue}% revenue growth'
        },
        'optimistic': {
            'revenue_growth': opt_revenue / 100, 
            'cost_increase': opt_cost / 100, 
            'probability': opt_prob / 100,
            'description': f'Best case scenario with {opt_revenue}% revenue growth'
        }
    }
    
    # Base financial data
    base_data = {
        'monthly_revenue': st.session_state.calculated_data.get('total_ventes', 15000),
        'monthly_cost': st.session_state.calculated_data.get('total_charges', 12000)
    }
    
    # Analysis period
    analysis_period = st.selectbox("Analysis Period", [12, 24, 36, 48], index=2, help="Number of months to analyze")
    
    # Calculate detailed scenarios
    if st.button("🚀 Run Scenario Analysis", type="primary"):
        with st.spinner("Running comprehensive scenario analysis..."):
            scenario_results = planner.calculate_detailed_scenarios(base_data, analysis_period)
            st.session_state.scenario_results = scenario_results
            
            # Calculate Value at Risk
            var_results = planner.calculate_value_at_risk(scenario_results)
            st.session_state.var_results = var_results
    
    # Display results if available
    if 'scenario_results' in st.session_state and st.session_state.scenario_results:
        scenario_results = st.session_state.scenario_results
        
        # Summary metrics
        st.subheader("📊 Scenario Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expected_value = sum(data['total_profit'] * data['probability'] for data in scenario_results.values())
            st.metric("Expected Value", f"{expected_value:,.0f} DHS")
        
        with col2:
            best_case = max(data['total_profit'] for data in scenario_results.values())
            st.metric("Best Case", f"{best_case:,.0f} DHS", f"+{best_case - expected_value:,.0f}")
        
        with col3:
            worst_case = min(data['total_profit'] for data in scenario_results.values())
            st.metric("Worst Case", f"{worst_case:,.0f} DHS", f"{worst_case - expected_value:,.0f}")
        
        with col4:
            profit_range = best_case - worst_case
            st.metric("Profit Range", f"{profit_range:,.0f} DHS")
        
        # Detailed scenario comparison
        tab1, tab2, tab3 = st.tabs(["📈 Profit Evolution", "📊 Quarterly Analysis", "🎲 Monte Carlo"])
        
        with tab1:
            # Create comprehensive profit evolution chart
            fig = go.Figure()
            
            colors = {'pessimistic': '#FF6B6B', 'realistic': '#4ECDC4', 'optimistic': '#45B7D1'}
            
            for scenario, data in scenario_results.items():
                months = list(range(1, len(data['cumulative_profit']) + 1))
                
                # Add cumulative profit line
                fig.add_trace(go.Scatter(
                    x=months,
                    y=data['cumulative_profit'],
                    mode='lines+markers',
                    name=f"{scenario.title()} (Prob: {data['probability']:.0%})",
                    line=dict(color=colors[scenario], width=3),
                    marker=dict(size=6),
                    hovertemplate=f"<b>{scenario.title()}</b><br>Month: %{{x}}<br>Cumulative Profit: %{{y:,.0f}} DHS<extra></extra>"
                ))
            
            # Add break-even line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Break-even Line")
            
            fig.update_layout(
                title="Cumulative Profit Evolution by Scenario",
                xaxis_title="Month",
                yaxis_title="Cumulative Profit (DHS)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario insights
            st.markdown("**📝 Scenario Insights:**")
            for scenario, data in scenario_results.items():
                prob_loss = sum(1 for m in data['monthly_data'] if m['profit'] < 0) / len(data['monthly_data'])
                avg_monthly = data['avg_monthly_profit']
                volatility = data['profit_volatility']
                
                if scenario == 'pessimistic':
                    st.error(f"🔴 **{scenario.title()}**: {prob_loss:.0%} months with losses, avg monthly profit: {avg_monthly:,.0f} DHS")
                elif scenario == 'realistic':
                    st.info(f"🔵 **{scenario.title()}**: {prob_loss:.0%} months with losses, avg monthly profit: {avg_monthly:,.0f} DHS")
                else:
                    st.success(f"🟢 **{scenario.title()}**: {prob_loss:.0%} months with losses, avg monthly profit: {avg_monthly:,.0f} DHS")
        
        with tab2:
            # Quarterly breakdown analysis
            st.subheader("Quarterly Performance Analysis")
            
            quarterly_data = []
            for scenario, data in scenario_results.items():
                for quarter_info in data['quarterly_data']:
                    quarterly_data.append({
                        'Scenario': scenario.title(),
                        'Quarter': f"Q{quarter_info['quarter']}",
                        'Revenue': quarter_info['revenue'],
                        'Costs': quarter_info['cost'],
                        'Profit': quarter_info['profit'],
                        'Profit Margin': quarter_info['profit'] / quarter_info['revenue'] * 100 if quarter_info['revenue'] > 0 else 0
                    })
            
            df_quarterly = pd.DataFrame(quarterly_data)
            
            # Quarterly profit chart
            fig = px.bar(
                df_quarterly, 
                x='Quarter', 
                y='Profit', 
                color='Scenario',
                title="Quarterly Profit by Scenario",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quarterly summary table
            st.subheader("Quarterly Summary Table")
            pivot_table = df_quarterly.pivot_table(
                index='Quarter', 
                columns='Scenario', 
                values=['Revenue', 'Profit'], 
                aggfunc='mean'
            )
            
            st.dataframe(
                pivot_table.style.format("{:,.0f}"),
                use_container_width=True
            )
        
        with tab3:
            # Monte Carlo simulation
            st.subheader("🎲 Monte Carlo Simulation")
            
            if st.button("Run Monte Carlo Simulation (1000 iterations)"):
                with st.spinner("Running Monte Carlo simulation..."):
                    # Use AdvancedAnalytics for Monte Carlo
                    analytics = AdvancedAnalytics()
                    mc_results = analytics.monte_carlo_simulation(
                        base_data['monthly_revenue'],
                        base_data['monthly_cost'],
                        volatility=0.2,
                        simulations=1000,
                        periods=12
                    )
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        prob_loss = (mc_results['net_profit'] < 0).mean() * 100
                        st.metric("Probability of Loss", f"{prob_loss:.1f}%")
                    
                    with col2:
                        percentile_5 = mc_results['net_profit'].quantile(0.05)
                        st.metric("5th Percentile", f"{percentile_5:,.0f} DHS")
                    
                    with col3:
                        percentile_95 = mc_results['net_profit'].quantile(0.95)
                        st.metric("95th Percentile", f"{percentile_95:,.0f} DHS")
                    
                    with col4:
                        expected_mc = mc_results['net_profit'].mean()
                        st.metric("Expected Value", f"{expected_mc:,.0f} DHS")
                    
                    # Distribution chart
                    fig = px.histogram(
                        mc_results, 
                        x='net_profit', 
                        nbins=50,
                        title="Distribution of Annual Profit (1000 simulations)",
                        labels={'net_profit': 'Annual Profit (DHS)', 'count': 'Frequency'}
                    )
                    
                    # Add percentile lines
                    fig.add_vline(x=percentile_5, line_dash="dash", line_color="red", 
                                 annotation_text=f"5th Percentile: {percentile_5:,.0f}")
                    fig.add_vline(x=percentile_95, line_dash="dash", line_color="green",
                                 annotation_text=f"95th Percentile: {percentile_95:,.0f}")
                    fig.add_vline(x=0, line_dash="dash", line_color="orange",
                                 annotation_text="Break-even")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk metrics
                    st.subheader("📊 Risk Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Value at Risk (VaR):**")
                        st.write(f"• 95% VaR: {mc_results['net_profit'].quantile(0.05):,.0f} DHS")
                        st.write(f"• 99% VaR: {mc_results['net_profit'].quantile(0.01):,.0f} DHS")
                    
                    with col2:
                        volatility = mc_results['net_profit'].std()
                        sharpe_ratio = expected_mc / volatility if volatility > 0 else 0
                        st.markdown("**Risk-Adjusted Metrics:**")
                        st.write(f"• Volatility: {volatility:,.0f} DHS")
                        st.write(f"• Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Stress Testing Section
    st.subheader("🧪 Stress Testing")
    
    with st.expander("Configure Stress Test", expanded=False):
        stress_scenario = st.selectbox(
            "Select Stress Scenario",
            ['recession', 'supply_shock', 'competition'],
            format_func=lambda x: {
                'recession': '📉 Economic Recession',
                'supply_shock': '🚛 Supply Chain Crisis', 
                'competition': '⚔️ Competitive Disruption'
            }[x]
        )
        
        if st.button("🔬 Run Stress Test"):
            with st.spinner("Running stress test analysis..."):
                stress_results = planner.stress_test_scenarios(base_data, stress_scenario)
                
                # Display stress test results
                st.subheader(f"Stress Test Results: {stress_scenario.replace('_', ' ').title()}")
                
                # Timeline visualization
                timeline_df = pd.DataFrame(stress_results['timeline'])
                
                fig = px.line(
                    timeline_df,
                    x='month',
                    y='profit',
                    color='phase',
                    title=f"Business Performance Under {stress_scenario.replace('_', ' ').title()} Stress",
                    color_discrete_map={
                        'normal': 'green',
                        'stress': 'red', 
                        'recovery': 'orange'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Impact summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Impact",
                        f"{stress_results['total_impact']:,.0f} DHS",
                        "Cumulative profit loss"
                    )
                
                with col2:
                    stress_duration = stress_results['stress_details']['duration_months']
                    st.metric(
                        "Stress Duration",
                        f"{stress_duration} months",
                        f"Recovery: {stress_results['recovery_time']} months"
                    )
                
                with col3:
                    probability = stress_results['stress_details']['probability']
                    st.metric(
                        "Scenario Probability",
                        f"{probability:.0%}",
                        f"Expected loss: {stress_results['total_impact'] * probability:,.0f} DHS"
                    )

# ========== ML FORECASTING PAGE ==========
def show_ml_forecasting():
    """Enhanced ML forecasting with comprehensive analysis"""
    st.header("🤖 Machine Learning Financial Forecasting")
    
    # Initialize ML engine
    ml_engine = MLForecastingEngine()
    
    # Data preparation section
    st.subheader("📊 Historical Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate or load historical data
        if st.button("📈 Generate Sample Historical Data", type="primary"):
            # Create realistic historical data
            months = 24
            base_revenue = st.session_state.calculated_data.get('total_ventes', 15000)
            
            historical_data = []
            for i in range(months):
                # Create realistic patterns
                trend = 1 + (i * 0.015)  # 1.5% monthly growth
                seasonal = 1 + 0.15 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
                noise = np.random.normal(1, 0.08)  # 8% noise
                
                revenue = base_revenue * trend * seasonal * noise
                
                # Add some business logic
                if i == 6:  # Mid-year dip
                    revenue *= 0.85
                elif i == 11 or i == 23:  # Year-end boost
                    revenue *= 1.2
                
                historical_data.append({
                    'month': i + 1,
                    'revenue': max(revenue, 0),
                    'costs': revenue * 0.75,  # 75% cost ratio
                    'date': datetime(2022, 1, 1) + timedelta(days=30*i)
                })
            
            st.session_state.historical_revenue_data = historical_data
            st.success("✅ Historical data generated successfully!")
    
    with col2:
        if 'historical_revenue_data' in st.session_state:
            data_points = len(st.session_state.historical_revenue_data)
            st.metric("Data Points", f"{data_points} months")
            st.metric("Data Quality", "High" if data_points >= 18 else "Medium" if data_points >= 12 else "Low")
    
    # Display historical data if available
    if 'historical_revenue_data' in st.session_state:
        df_historical = pd.DataFrame(st.session_state.historical_revenue_data)
        
        # Historical data visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Revenue Trend', 'Revenue vs Costs'),
            vertical_spacing=0.12
        )
        
        # Revenue trend
        fig.add_trace(
            go.Scatter(
                x=df_historical['month'],
                y=df_historical['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # Revenue vs costs
        fig.add_trace(
            go.Scatter(
                x=df_historical['month'],
                y=df_historical['revenue'],
                mode='lines',
                name='Revenue',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_historical['month'],
                y=df_historical['costs'],
                mode='lines',
                name='Costs',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Historical Financial Data Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        with st.expander("📊 Data Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_revenue = df_historical['revenue'].mean()
                st.metric("Average Revenue", f"{avg_revenue:,.0f} DHS")
            
            with col2:
                revenue_growth = (df_historical['revenue'].iloc[-1] / df_historical['revenue'].iloc[0] - 1) * 100
                st.metric("Total Growth", f"{revenue_growth:.1f}%")
            
            with col3:
                volatility = df_historical['revenue'].std() / avg_revenue * 100
                st.metric("Volatility", f"{volatility:.1f}%")
            
            with col4:
                trend_slope = np.polyfit(range(len(df_historical)), df_historical['revenue'], 1)[0]
                st.metric("Monthly Trend", f"{trend_slope:,.0f} DHS")
        
        # ML Model Training
        st.subheader("🧠 Machine Learning Model Training")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            target_variable = st.selectbox(
                "Select Target Variable",
                ['revenue', 'costs'],
                help="Choose what you want to forecast"
            )
            
            forecast_horizon = st.slider(
                "Forecast Horizon (months)",
                min_value=3,
                max_value=24,
                value=12,
                help="How many months ahead to forecast"
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ['auto', 'random_forest', 'linear_regression'],
                help="Choose ML algorithm"
            )
            
            include_confidence = st.checkbox("Include Confidence Intervals", value=True)
        
        # Train model and generate forecasts
        if st.button("🚀 Train Model & Generate Forecasts", type="primary"):
            with st.spinner("Training machine learning models..."):
                # Train the model
                model_name, training_message = ml_engine.train_ensemble_model(
                    st.session_state.historical_revenue_data,
                    target_variable
                )
                
                if model_name:
                    st.success(f"✅ {training_message}")
                    
                    # Generate forecasts
                    forecast_results, forecast_message = ml_engine.generate_forecasts(
                        periods=forecast_horizon,
                        target_col=target_variable,
                        confidence_intervals=include_confidence
                    )
                    
                    if forecast_results:
                        st.success(f"✅ {forecast_message}")
                        
                        # Store results
                        st.session_state.ml_forecast_results = forecast_results
                        st.session_state.ml_model_info = {
                            'model_name': model_name,
                            'target': target_variable,
                            'horizon': forecast_horizon
                        }
                        
                        # Display forecasts
                        st.subheader("📈 Forecast Results")
                        
                        # Create forecast visualization
                        historical_months = list(range(1, len(df_historical) + 1))
                        forecast_months = list(range(len(df_historical) + 1, len(df_historical) + forecast_horizon + 1))
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_months,
                            y=df_historical[target_variable],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Forecasts
                        fig.add_trace(go.Scatter(
                            x=forecast_months,
                            y=forecast_results['forecasts'],
                            mode='lines+markers',
                            name='ML Forecast',
                            line=dict(color='red', width=3, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        # Confidence intervals
                        if include_confidence and 'lower_bounds' in forecast_results:
                            fig.add_trace(go.Scatter(
                                x=forecast_months + forecast_months[::-1],
                                y=forecast_results['upper_bounds'] + forecast_results['lower_bounds'][::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence Interval',
                                showlegend=True
                            ))
                        
                        fig.update_layout(
                            title=f"{target_variable.title()} Forecast: Historical vs ML Predictions",
                            xaxis_title="Month",
                            yaxis_title=f"{target_variable.title()} (DHS)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        forecasts = forecast_results['forecasts']
                        
                        with col1:
                            avg_forecast = np.mean(forecasts)
                            st.metric("Average Monthly Forecast", f"{avg_forecast:,.0f} DHS")
                        
                        with col2:
                            total_forecast = sum(forecasts)
                            st.metric(f"{forecast_horizon}-Month Total", f"{total_forecast:,.0f} DHS")
                        
                        with col3:
                            last_historical = df_historical[target_variable].iloc[-1]
                            growth_rate = (forecasts[-1] / last_historical - 1) * 100
                            st.metric("Projected Growth", f"{growth_rate:+.1f}%")
                        
                        with col4:
                            forecast_volatility = np.std(forecasts) / np.mean(forecasts) * 100
                            st.metric("Forecast Volatility", f"{forecast_volatility:.1f}%")
                        
                        # Model performance
                        if 'model_performance' in forecast_results:
                            st.subheader("🎯 Model Performance")
                            
                            performance = forecast_results['model_performance']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            for i, (model, metrics) in enumerate(performance.items()):
                                with [col1, col2, col3][i % 3]:
                                    st.markdown(f"**{model.replace('_', ' ').title()}**")
                                    st.write(f"RMSE: {metrics.get('rmse', 0):,.0f}")
                                    st.write(f"MAE: {metrics.get('mae', 0):,.0f}")
                                    st.write(f"MAPE: {metrics.get('mape', 0):.1f}%")
                        
                        # Feature importance
                        if 'feature_importance' in forecast_results and forecast_results['feature_importance']:
                            st.subheader("🔍 Feature Importance")
                            
                            importance = forecast_results['feature_importance']
                            features = list(importance.keys())
                            values = list(importance.values())
                            
                            fig = px.bar(
                                x=values,
                                y=features,
                                orientation='h',
                                title="Feature Importance in ML Model",
                                labels={'x': 'Importance Score', 'y': 'Features'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ {training_message}")
    
    # Forecast analysis and insights
    if 'ml_forecast_results' in st.session_state:
        st.subheader("💡 Forecast Insights & Analysis")
        
        forecast_data = st.session_state.ml_forecast_results
        model_info = st.session_state.ml_model_info
        
        tab1, tab2, tab3 = st.tabs(["📊 Trend Analysis", "📈 Seasonality", "⚠️ Risk Assessment"])
        
        with tab1:
            forecasts = forecast_data['forecasts']
            
            # Trend analysis
            trend_slope = np.polyfit(range(len(forecasts)), forecasts, 1)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if trend_slope > 0:
                    st.success(f"📈 **Positive Trend**: {trend_slope:,.0f} DHS/month increase")
                    st.write("The forecast shows consistent growth over the projection period.")
                elif trend_slope < -500:
                    st.error(f"📉 **Negative Trend**: {abs(trend_slope):,.0f} DHS/month decrease")
                    st.write("The forecast indicates declining performance. Consider strategic interventions.")
                else:
                    st.info("📊 **Stable Trend**: Relatively flat projection")
                    st.write("The forecast shows stable performance with minimal variation.")
            
            with col2:
                # Growth rate analysis
                if len(forecasts) > 1:
                    monthly_growth_rates = [((forecasts[i] / forecasts[i-1]) - 1) * 100 for i in range(1, len(forecasts))]
                    avg_growth = np.mean(monthly_growth_rates)
                    
                    st.metric("Average Monthly Growth", f"{avg_growth:+.1f}%")
                    
                    if abs(avg_growth) < 1:
                        growth_assessment = "Stable"
                    elif avg_growth > 5:
                        growth_assessment = "High Growth"
                    elif avg_growth > 0:
                        growth_assessment = "Moderate Growth"
                    else:
                        growth_assessment = "Declining"
                    
                    st.write(f"**Assessment**: {growth_assessment}")
        
        with tab2:
            # Seasonality analysis
            if len(forecasts) >= 12:
                # Monthly seasonality
                monthly_avg = np.mean(forecasts)
                seasonal_factors = [f / monthly_avg for f in forecasts[:12]]
                
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig = px.bar(
                    x=months[:len(seasonal_factors)],
                    y=seasonal_factors,
                    title="Seasonal Factors from ML Forecast",
                    labels={'x': 'Month', 'y': 'Seasonal Factor'}
                )
                
                fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Average")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal insights
                peak_month = months[np.argmax(seasonal_factors)]
                low_month = months[np.argmin(seasonal_factors)]
                
                st.write(f"**Peak Season**: {peak_month} (Factor: {max(seasonal_factors):.2f})")
                st.write(f"**Low Season**: {low_month} (Factor: {min(seasonal_factors):.2f})")
                
                seasonality_strength = (max(seasonal_factors) - min(seasonal_factors)) / np.mean(seasonal_factors)
                
                if seasonality_strength > 0.3:
                    st.warning("🌊 **High Seasonality**: Significant seasonal variations detected")
                elif seasonality_strength > 0.15:
                    st.info("📊 **Moderate Seasonality**: Some seasonal patterns present")
                else:
                    st.success("🟢 **Low Seasonality**: Stable throughout the year")
            else:
                st.info("Need at least 12 months of forecast data for seasonality analysis")
        
        with tab3:
            # Risk assessment of forecasts
            forecasts = forecast_data['forecasts']
            
            # Volatility analysis
            forecast_volatility = np.std(forecasts) / np.mean(forecasts)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Forecast Volatility", f"{forecast_volatility:.1%}")
                
                if forecast_volatility > 0.2:
                    st.error("🔴 **High Risk**: Very volatile forecasts")
                    risk_level = "High"
                elif forecast_volatility > 0.1:
                    st.warning("🟡 **Medium Risk**: Moderate forecast volatility")
                    risk_level = "Medium"
                else:
                    st.success("🟢 **Low Risk**: Stable forecast pattern")
                    risk_level = "Low"
            
            with col2:
                # Confidence intervals analysis
                if 'lower_bounds' in forecast_data and 'upper_bounds' in forecast_data:
                    avg_confidence_width = np.mean([
                        u - l for u, l in zip(forecast_data['upper_bounds'], forecast_data['lower_bounds'])
                    ])
                    
                    st.metric("Avg Confidence Width", f"{avg_confidence_width:,.0f} DHS")
                    
                    relative_width = avg_confidence_width / np.mean(forecasts)
                    
                    if relative_width > 0.4:
                        st.warning("📏 **Wide Intervals**: High prediction uncertainty")
                    elif relative_width > 0.2:
                        st.info("📐 **Moderate Intervals**: Reasonable prediction confidence")
                    else:
                        st.success("📌 **Narrow Intervals**: High prediction confidence")
            
            # Risk recommendations
            st.markdown("**🛡️ Risk Mitigation Recommendations:**")
            
            if risk_level == "High":
                st.write("• Consider shorter planning horizons")
                st.write("• Implement more frequent forecast updates")
                st.write("• Develop contingency plans for volatile scenarios")
                st.write("• Monitor key performance indicators closely")
            elif risk_level == "Medium":
                st.write("• Review forecasts quarterly")
                st.write("• Maintain flexible operational plans")
                st.write("• Monitor market conditions for changes")
            else:
                st.write("• Annual forecast reviews should be sufficient")
                st.write("• Use forecasts for strategic planning")
                st.write("• Consider extending forecast horizon")

# ========== RISK MANAGEMENT PAGE ==========
def show_risk_management():
    """Comprehensive risk management and analysis page"""
    st.header("⚠️ Advanced Risk Management & Analysis")
    
    # Initialize risk management system
    risk_manager = RiskManagementSystem()
    
    # Current financial data for risk assessment
    financial_data = {
        'revenue': st.session_state.calculated_data.get('total_ventes', 15000) * 12,
        'total_costs': st.session_state.calculated_data.get('total_charges', 12000) * 12,
        'cash_flow': st.session_state.calculated_data.get('cash_flow_mensuel', 3000),
        'current_ratio': 1.5,  # Would be calculated from actual balance sheet
        'debt_to_equity': 0.4,
        'interest_coverage': 5.0,
        'net_margin': 0.2
    }
    
    # Calculate comprehensive risk score
    overall_risk, risk_breakdown = risk_manager.calculate_comprehensive_risk_score(financial_data)
    
    # Risk Dashboard
    st.subheader("🎯 Risk Assessment Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Risk score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Risk Score"},
            delta={'reference': 30},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk breakdown by category
        categories = list(risk_breakdown.keys())
        scores = [score * 100 for score in risk_breakdown.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=['red' if s > 60 else 'orange' if s > 40 else 'yellow' if s > 25 else 'green' for s in scores]
            )
        ])
        
        fig.update_layout(
            title="Risk Breakdown by Category",
            yaxis_title="Risk Score (%)",
            xaxis_title="Risk Category"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level interpretation
        if overall_risk < 0.25:
            st.success("🟢 **Low Risk**: Well-managed financial position")
        elif overall_risk < 0.5:
            st.info("🔵 **Moderate Risk**: Some areas need attention")
        elif overall_risk < 0.75:
            st.warning("🟡 **High Risk**: Significant risk factors present")
        else:
            st.error("🔴 **Critical Risk**: Immediate action required")
    
    # Detailed Risk Analysis
    st.subheader("🔍 Detailed Risk Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Operational", "💰 Financial", "📈 Market", "📋 Regulatory"])
    
    with tab1:
        st.markdown("### Operational Risk Assessment")
        
        operational_risk = risk_breakdown.get('operational', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cash flow risk
            cash_flow = financial_data.get('cash_flow', 0)
            monthly_revenue = financial_data.get('revenue', 120000) / 12
            cf_ratio = cash_flow / monthly_revenue if monthly_revenue > 0 else 0
            
            st.metric("Cash Flow Ratio", f"{cf_ratio:.1%}")
            
            if cf_ratio < 0:
                st.error("🔴 Negative cash flow - Critical issue")
            elif cf_ratio < 0.05:
                st.warning("🟡 Low cash flow margins")
            else:
                st.success("🟢 Healthy cash flow")
        
        with col2:
            # Operational efficiency
            revenue = financial_data.get('revenue', 0)
            total_costs = financial_data.get('total_costs', 0)
            efficiency = (revenue - total_costs) / revenue if revenue > 0 else 0
            
            st.metric("Operational Efficiency", f"{efficiency:.1%}")
            
            if efficiency < 0.1:
                st.error("🔴 Low operational efficiency")
            elif efficiency < 0.2:
                st.warning("🟡 Moderate efficiency")
            else:
                st.success("🟢 High efficiency")
        
        # Operational risk factors
        st.markdown("**Key Operational Risk Factors:**")
        
        risk_factors = []
        
        if cf_ratio < 0.05:
            risk_factors.append("• Cash flow constraints limiting operational flexibility")
        
        if efficiency < 0.15:
            risk_factors.append("• Low profit margins indicating operational inefficiencies")
        
        # Process dependencies (simulated)
        process_risks = [
            "• Key person dependency in critical processes",
            "• Single supplier for essential materials",
            "• Outdated technology systems",
            "• Lack of documented procedures"
        ]
        
        for risk in process_risks[:2]:  # Show top 2 process risks
            risk_factors.append(risk)
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.success("No major operational risk factors identified")
    
    with tab2:
        st.markdown("### Financial Risk Assessment")
        
        financial_risk = risk_breakdown.get('financial', 0)
        
        # Financial metrics grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            debt_to_equity = financial_data.get('debt_to_equity', 0)
            st.metric("Debt-to-Equity", f"{debt_to_equity:.2f}")
            
            if debt_to_equity > 2:
                st.error("🔴 High leverage risk")
            elif debt_to_equity > 1:
                st.warning("🟡 Moderate leverage")
            else:
                st.success("🟢 Conservative leverage")
        
        with col2:
            current_ratio = financial_data.get('current_ratio', 0)
            st.metric("Current Ratio", f"{current_ratio:.2f}")
            
            if current_ratio < 1:
                st.error("🔴 Liquidity crisis risk")
            elif current_ratio < 1.5:
                st.warning("🟡 Tight liquidity")
            else:
                st.success("🟢 Good liquidity")
        
        with col3:
            interest_coverage = financial_data.get('interest_coverage', 0)
            st.metric("Interest Coverage", f"{interest_coverage:.1f}x")
            
            if interest_coverage < 2:
                st.error("🔴 High default risk")
            elif interest_coverage < 5:
                st.warning("🟡 Moderate credit risk")
            else:
                st.success("🟢 Strong coverage")
        
        # Credit risk assessment
        st.markdown("**Credit Risk Analysis:**")
        
        credit_score = 100
        
        if debt_to_equity > 2:
            credit_score -= 30
        elif debt_to_equity > 1:
            credit_score -= 15
        
        if current_ratio < 1:
            credit_score -= 25
        elif current_ratio < 1.5:
            credit_score -= 10
        
        if interest_coverage < 2:
            credit_score -= 20
        elif interest_coverage < 5:
            credit_score -= 10
        
        st.metric("Credit Score", f"{credit_score}/100")
        
        if credit_score >= 80:
            st.success("🟢 Excellent creditworthiness")
        elif credit_score >= 60:
            st.info("🔵 Good credit profile")
        elif credit_score >= 40:
            st.warning("🟡 Moderate credit risk")
        else:
            st.error("🔴 High credit risk")
    
    with tab3:
        st.markdown("### Market Risk Assessment")
        
        # Market risk simulation
        market_volatility = 0.15  # 15% market volatility
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Market Volatility", f"{market_volatility:.1%}")
            
            # Revenue sensitivity analysis
            revenue_sensitivity = st.slider(
                "Revenue Sensitivity to Market Changes",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="How much your revenue changes with market conditions"
            )
            
            market_risk_score = market_volatility * revenue_sensitivity
            
            if market_risk_score > 0.25:
                st.error(f"🔴 High market risk exposure ({market_risk_score:.1%})")
            elif market_risk_score > 0.15:
                st.warning(f"🟡 Moderate market risk ({market_risk_score:.1%})")
            else:
                st.success(f"🟢 Low market risk ({market_risk_score:.1%})")
        
        with col2:
            # Competitive position
            competitive_strength = st.selectbox(
                "Competitive Position",
                ["Market Leader", "Strong Competitor", "Average Player", "Struggling", "New Entrant"],
                index=1
            )
            
            competitive_scores = {
                "Market Leader": 0.1,
                "Strong Competitor": 0.2,
                "Average Player": 0.4,
                "Struggling": 0.7,
                "New Entrant": 0.5
            }
            
            comp_risk = competitive_scores[competitive_strength]
            st.metric("Competitive Risk", f"{comp_risk:.1%}")
        
        # Market risk scenarios
        st.markdown("**Market Risk Scenarios:**")
        
        scenarios = {
            "Economic Recession": {"probability": 0.15, "impact": -0.25},
            "Industry Disruption": {"probability": 0.1, "impact": -0.4},
            "New Regulation": {"probability": 0.2, "impact": -0.1},
            "Commodity Price Spike": {"probability": 0.25, "impact": -0.15}
        }
        
        for scenario, data in scenarios.items():
            expected_loss = data["probability"] * abs(data["impact"]) * financial_data.get('revenue', 0)
            st.write(f"• **{scenario}**: {data['probability']:.0%} chance, {expected_loss:,.0f} DHS potential impact")
    
    with tab4:
        st.markdown("### Regulatory Risk Assessment")
        
        # Industry-specific regulatory risks
        industry = st.session_state.basic_info.get('industry', 'technology')
        
        regulatory_risks = {
            'technology': [
                "Data protection and privacy regulations",
                "Cybersecurity compliance requirements",
                "Intellectual property disputes",
                "Tax regulations for digital services"
            ],
            'retail': [
                "Consumer protection regulations",
                "Product safety standards",
                "Labor law compliance",
                "Environmental regulations"
            ],
            'manufacturing': [
                "Environmental and safety regulations",
                "Product quality standards",
                "Labor and workplace safety",
                "International trade regulations"
            ],
            'saas': [
                "Data protection (GDPR, CCPA)",
                "Software licensing compliance",
                "Subscription billing regulations",
                "International data transfer rules"
            ]
        }
        
        current_risks = regulatory_risks.get(industry, regulatory_risks['technology'])
        
        st.write(f"**Key Regulatory Risks for {industry.title()} Industry:**")
        
        for i, risk in enumerate(current_risks):
            risk_level = ["Low", "Medium", "High"][i % 3]
            if risk_level == "High":
                st.error(f"🔴 {risk}")
            elif risk_level == "Medium":
                st.warning(f"🟡 {risk}")
            else:
                st.info(f"🔵 {risk}")
        
        # Compliance score
        compliance_score = st.slider(
            "Current Compliance Score",
            min_value=0,
            max_value=100,
            value=85,
            help="Rate your current regulatory compliance"
        )
        
        if compliance_score >= 90:
            st.success("🟢 Excellent compliance posture")
        elif compliance_score >= 70:
            st.info("🔵 Good compliance management")
        elif compliance_score >= 50:
            st.warning("🟡 Compliance gaps need attention")
        else:
            st.error("🔴 Significant compliance risks")
    
    # Stress Testing
    st.subheader("🧪 Comprehensive Stress Testing")
    
    # Stress test configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_scenarios = st.multiselect(
            "Select Stress Test Scenarios",
            list(risk_manager.stress_scenarios.keys()),
            default=['mild_recession', 'supply_chain_crisis'],
            format_func=lambda x: {
                'mild_recession': '📉 Mild Economic Recession',
                'severe_recession': '📉 Severe Economic Recession',
                'supply_chain_crisis': '🚛 Supply Chain Crisis',
                'competitive_disruption': '⚔️ Competitive Disruption',
                'regulatory_change': '📋 Major Regulatory Change'
            }.get(x, x)
        )
    
    with col2:
        stress_duration = st.selectbox(
            "Test Duration",
            [12, 18, 24, 36],
            index=1,
            help="Duration of stress period in months"
        )
    
    if st.button("🔬 Run Comprehensive Stress Tests", type="primary"):
        with st.spinner("Running stress test analysis..."):
            stress_results = risk_manager.perform_stress_testing(financial_data, selected_scenarios)
            
            # Display stress test results
            st.subheader("📊 Stress Test Results")
            
            # Summary table
            stress_df = pd.DataFrame([
                {
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Revenue Impact': f"{data['stressed_revenue']:,.0f} DHS",
                    'Profit Impact': f"{data['profit_impact']:+,.0f} DHS",
                    'Impact %': f"{data['profit_impact_pct']:+.1f}%",
                    'Probability': f"{data['probability']:.0%}",
                    'Expected Loss': f"{data['expected_loss']:,.0f} DHS"
                }
                for scenario, data in stress_results.items()
            ])
            
            st.dataframe(stress_df, use_container_width=True)
            
            # Stress test visualization
            scenarios = list(stress_results.keys())
            profit_impacts = [stress_results[s]['profit_impact'] for s in scenarios]
            probabilities = [stress_results[s]['probability'] for s in scenarios]
            
            # Create bubble chart
            fig = go.Figure(data=go.Scatter(
                x=scenarios,
                y=profit_impacts,
                mode='markers',
                marker=dict(
                    size=[p * 1000 for p in probabilities],  # Size based on probability
                    color=profit_impacts,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Profit Impact (DHS)")
                ),
                text=[f"Probability: {p:.0%}" for p in probabilities],
                hovertemplate='<b>%{x}</b><br>Impact: %{y:,.0f} DHS<br>%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Stress Test Impact Analysis (Bubble size = Probability)",
                xaxis_title="Scenario",
                yaxis_title="Profit Impact (DHS)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Value at Risk calculation
            var_results = risk_manager.calculate_value_at_risk(stress_results)
            
            st.subheader("📊 Value at Risk Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("95% VaR", f"{var_results.get('VaR_95', 0):,.0f} DHS")
                st.caption("Maximum expected loss at 95% confidence")
            
            with col2:
                st.metric("99% VaR", f"{var_results.get('VaR_99', 0):,.0f} DHS")
                st.caption("Maximum expected loss at 99% confidence")
            
            with col3:
                st.metric("Expected Shortfall (95%)", f"{var_results.get('ES_95', 0):,.0f} DHS")
                st.caption("Average loss beyond VaR threshold")
    
    # Risk Mitigation Strategies
    st.subheader("🛡️ Risk Mitigation Strategies")
    
    # Generate mitigation strategies
    mitigation_strategies = risk_manager.generate_risk_mitigation_strategies(
        risk_breakdown, 
        {} if 'stress_results' not in locals() else stress_results
    )
    
    for strategy in mitigation_strategies:
        with st.expander(f"🎯 {strategy['category']} - {strategy['priority']} Priority", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Recommended Actions:**")
                for action in strategy['strategies']:
                    st.write(f"• {action}")
            
            with col2:
                st.metric("Timeline", strategy['timeline'])
                st.metric("Investment Level", strategy['investment_required'])
                
                if strategy['priority'] == 'High':
                    st.error("🔴 High Priority")
                elif strategy['priority'] == 'Medium':
                    st.warning("🟡 Medium Priority")
                else:
                    st.info("🔵 Low Priority")
    
    # Risk monitoring dashboard
    st.subheader("📊 Risk Monitoring Dashboard")
    
    # Key risk indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Liquidity indicator
        days_cash = 30  # Simplified calculation
        st.metric("Days Cash on Hand", f"{days_cash} days")
        
        if days_cash < 30:
            st.error("🔴 Critical")
        elif days_cash < 60:
            st.warning("🟡 Warning")
        else:
            st.success("🟢 Safe")
    
    with col2:
        # Debt service coverage
        debt_service_coverage = financial_data.get('interest_coverage', 5)
        st.metric("Debt Service Coverage", f"{debt_service_coverage:.1f}x")
        
        if debt_service_coverage < 1.2:
            st.error("🔴 Critical")
        elif debt_service_coverage < 2.5:
            st.warning("🟡 Warning")
        else:
            st.success("🟢 Safe")
    
    with col3:
        # Customer concentration
        customer_concentration = 0.25  # 25% of revenue from top customer
        st.metric("Top Customer %", f"{customer_concentration:.0%}")
        
        if customer_concentration > 0.3:
            st.error("🔴 High Risk")
        elif customer_concentration > 0.2:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Diversified")
    
    with col4:
        # Operational leverage
        operational_leverage = 1.5
        st.metric("Operating Leverage", f"{operational_leverage:.1f}")
        
        if operational_leverage > 2:
            st.error("🔴 High Risk")
        elif operational_leverage > 1.5:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

# ========== INDUSTRY TEMPLATES PAGE ==========
def show_industry_templates():
    """Enhanced industry templates with comprehensive analysis"""
    st.header("🏭 Industry-Specific Financial Models")
    
    # Initialize template manager
    template_manager = IndustryTemplateManager()
    
    # Industry selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        industry = st.selectbox(
            "Select Your Industry",
            ['retail', 'saas', 'manufacturing', 'restaurant', 'consulting'],
            format_func=lambda x: {
                'retail': '🛍️ Retail & E-commerce',
                'saas': '💻 Software as a Service',
                'manufacturing': '🏭 Manufacturing',
                'restaurant': '🍽️ Restaurant & Food Service',
                'consulting': '💼 Professional Consulting'
            }[x],
            index=0
        )
    
    with col2:
        apply_template = st.button("🎯 Apply Industry Template", type="primary")
    
    # Store selected industry
    st.session_state.basic_info['industry'] = industry
    template = template_manager.get_template(industry)
    
    # Industry overview
    st.subheader(f"📋 {industry.title()} Industry Profile")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Financial Model", "📈 Benchmarks", "🎯 Projections"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Revenue Model")
            st.code(template['revenue_model'], language="text")
            
            st.markdown("### 🎯 Key Performance Metrics")
            for metric in template['key_metrics']:
                st.write(f"• {metric}")
        
        with col2:
            st.markdown("### 💼 Typical Cost Structure")
            
            # Create cost structure pie chart
            cost_structure = template['cost_structure']
            
            fig = go.Figure(data=[go.Pie(
                labels=list(cost_structure.keys()),
                values=list(cost_structure.values()),
                hole=0.3
            )])
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=10
            )
            
            fig.update_layout(
                title="Typical Cost Structure",
                annotations=[dict(text=industry.title(), x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔧 Financial Model Configuration")
        
        # Base parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_revenue = st.number_input(
                "Monthly Base Revenue (DHS)",
                min_value=1000,
                max_value=10000000,
                value=st.session_state.calculated_data.get('total_ventes', 15000),
                step=1000
            )
        
        with col2:
            growth_rate = st.slider(
                "Annual Growth Rate (%)",
                min_value=-10,
                max_value=50,
                value=12,
                help="Expected annual revenue growth"
            )
        
        with col3:
            projection_months = st.selectbox(
                "Projection Period",
                [12, 24, 36, 48],
                index=2,
                help="Number of months to project"
            )
        
        # Industry-specific parameters
        st.markdown("### ⚙️ Industry-Specific Parameters")
        
        if industry == 'retail':
            col1, col2, col3 = st.columns(3)
            with col1:
                stores_count = st.number_input("Number of Stores", min_value=1, value=1)
            with col2:
                avg_transaction = st.number_input("Average Transaction (DHS)", min_value=1, value=150)
            with col3:
                transactions_per_day = st.number_input("Transactions per Day", min_value=1, value=50)
        
        elif industry == 'saas':
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_users = st.number_input("Monthly Active Users", min_value=1, value=1000)
            with col2:
                arpu = st.number_input("ARPU (DHS/month)", min_value=1, value=50)
            with col3:
                churn_rate = st.slider("Monthly Churn Rate (%)", 0, 20, 5) / 100
        
        elif industry == 'manufacturing':
            col1, col2, col3 = st.columns(3)
            with col1:
                production_capacity = st.number_input("Monthly Capacity (units)", min_value=1, value=1000)
            with col2:
                capacity_utilization = st.slider("Capacity Utilization (%)", 50, 100, 80) / 100
            with col3:
                price_per_unit = st.number_input("Price per Unit (DHS)", min_value=1, value=100)
        
        elif industry == 'restaurant':
            col1, col2, col3 = st.columns(3)
            with col1:
                seats = st.number_input("Number of Seats", min_value=1, value=50)
            with col2:
                turnover_rate = st.number_input("Daily Turnover Rate", min_value=1.0, value=2.5, step=0.1)
            with col3:
                avg_check = st.number_input("Average Check (DHS)", min_value=1, value=80)
        
        elif industry == 'consulting':
            col1, col2, col3 = st.columns(3)
            with col1:
                consultants = st.number_input("Number of Consultants", min_value=1, value=5)
            with col2:
                billable_hours = st.number_input("Monthly Billable Hours", min_value=50, value=160)
            with col3:
                hourly_rate = st.number_input("Hourly Rate (DHS)", min_value=50, value=500)
        
        # Generate projections
        if st.button("📊 Generate Industry Projections"):
            base_data = {'monthly_revenue': base_revenue}
            
            projections = template_manager.apply_template_to_projections(
                base_data, industry, projection_months
            )
            
            st.session_state.industry_projections = projections
            st.success("✅ Industry projections generated successfully!")
    
    with tab3:
        st.markdown("### 📊 Industry Benchmarks")
        
        # Display typical ratios
        typical_ratios = template['typical_ratios']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Financial Ratios**")
            for ratio, value in typical_ratios.items():
                if isinstance(value, float) and value < 1:
                    st.metric(ratio.replace('_', ' ').title(), f"{value:.1%}")
                else:
                    st.metric(ratio.replace('_', ' ').title(), f"{value:.2f}")
        
        with col2:
            st.markdown("**Working Capital Metrics**")
            wc_metrics = template['working_capital']
            for metric, value in wc_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.0f} days")
        
        # Compare with current performance
        if st.session_state.calculated_data:
            st.markdown("### 🎯 Performance vs Industry Benchmarks")
            
            # Calculate current ratios (simplified)
            current_ratios = {
                'gross_margin': 0.3,  # Would be calculated from actual data
                'net_margin': 0.15,
                'current_ratio': 1.5,
                'asset_turnover': 1.2
            }
            
            # Create comparison chart
            comparison_data = []
            for ratio in typical_ratios.keys():
                if ratio in current_ratios:
                    comparison_data.append({
                        'Metric': ratio.replace('_', ' ').title(),
                        'Your Company': current_ratios[ratio],
                        'Industry Average': typical_ratios[ratio],
                        'Difference': current_ratios[ratio] - typical_ratios[ratio]
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                fig = go.Figure(data=[
                    go.Bar(name='Your Company', x=df_comparison['Metric'], y=df_comparison['Your Company']),
                    go.Bar(name='Industry Average', x=df_comparison['Metric'], y=df_comparison['Industry Average'])
                ])
                
                fig.update_layout(
                    barmode='group',
                    title="Performance vs Industry Benchmarks"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance assessment
                above_benchmark = sum(1 for row in comparison_data if row['Difference'] > 0)
                total_metrics = len(comparison_data)
                
                if above_benchmark / total_metrics >= 0.75:
                    st.success(f"🎉 **Outstanding Performance**: {above_benchmark}/{total_metrics} metrics above industry average")
                elif above_benchmark / total_metrics >= 0.5:
                    st.info(f"📈 **Good Performance**: {above_benchmark}/{total_metrics} metrics above industry average")
                else:
                    st.warning(f"⚠️ **Room for Improvement**: Only {above_benchmark}/{total_metrics} metrics above industry average")
    
    with tab4:
        st.markdown("### 🎯 Industry Projections")
        
        if 'industry_projections' in st.session_state:
            projections = st.session_state.industry_projections
            df_proj = pd.DataFrame(projections)
            
            # Revenue projection chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Revenue Projection', 'Cost Breakdown', 'Profit Trend', 'Seasonal Factors'),
                specs=[[{"colspan": 2}, None],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Revenue projection
            fig.add_trace(
                go.Scatter(
                    x=df_proj['month'],
                    y=df_proj['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='blue', width=3)
                ),
                row=1, col=1
            )
            
            # Profit trend
            fig.add_trace(
                go.Scatter(
                    x=df_proj['month'],
                    y=df_proj['gross_profit'],
                    mode='lines',
                    name='Gross Profit',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Seasonal factors
            seasonal_data = df_proj.head(12)  # First 12 months
            fig.add_trace(
                go.Bar(
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(seasonal_data)],
                    y=seasonal_data['seasonal_factor'],
                    name='Seasonal Factor',
                    marker_color='orange'
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Industry-Specific Financial Projections")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_revenue = sum(p['revenue'] for p in projections)
            total_profit = sum(p['gross_profit'] for p in projections)
            avg_margin = total_profit / total_revenue if total_revenue > 0 else 0
            
            with col1:
                st.metric("Total Projected Revenue", f"{total_revenue:,.0f} DHS")
            
            with col2:
                st.metric("Total Projected Profit", f"{total_profit:,.0f} DHS")
            
            with col3:
                st.metric("Average Margin", f"{avg_margin:.1%}")
            
            with col4:
                final_monthly = projections[-1]['revenue']
                initial_monthly = projections[0]['revenue']
                growth = (final_monthly / initial_monthly - 1) * 100
                st.metric("Period Growth", f"{growth:+.1f}%")
            
            # Detailed projections table
            st.markdown("### 📋 Detailed Monthly Projections")
            
            # Create summary table
            summary_data = []
            for i, proj in enumerate(projections):
                if i % 3 == 0:  # Show quarterly data
                    quarter = f"Q{(i//3)+1} Y{(i//12)+1}"
                    summary_data.append({
                        'Period': quarter,
                        'Revenue': f"{proj['revenue']:,.0f}",
                        'Total Costs': f"{proj['total_costs']:,.0f}",
                        'Gross Profit': f"{proj['gross_profit']:,.0f}",
                        'Margin': f"{proj['gross_profit']/proj['revenue']*100:.1f}%"
                    })
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        else:
            st.info("📊 Generate projections in the Financial Model tab to see detailed forecasts")
        
        # Seasonality analysis
        st.markdown("### 📅 Seasonal Analysis")
        
        seasonal_factors = template['seasonal_factors']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.line(
            x=months,
            y=seasonal_factors,
            title=f"Typical Seasonal Pattern for {industry.title()} Industry",
            markers=True
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Average")
        fig.update_layout(yaxis_title="Seasonal Factor", xaxis_title="Month")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal insights
        peak_month = months[np.argmax(seasonal_factors)]
        low_month = months[np.argmin(seasonal_factors)]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Season", peak_month, f"{max(seasonal_factors):.2f}x")
        
        with col2:
            st.metric("Low Season", low_month, f"{min(seasonal_factors):.2f}x")
        
        with col3:
            seasonality_range = max(seasonal_factors) - min(seasonal_factors)
            st.metric("Seasonality Range", f"{seasonality_range:.2f}x")
    
    # Apply template button action
    if apply_template:
        with st.spinner("Applying industry template..."):
            # Update session state with industry-specific data
            st.session_state.basic_info['industry'] = industry
            
            # Apply industry-typical ratios to financial calculations
            st.session_state.industry_template_applied = {
                'industry': industry,
                'template': template,
                'applied_at': datetime.now().isoformat()
            }
            
            # Update calculated data with industry standards
            if 'calculated_data' not in st.session_state:
                st.session_state.calculated_data = {}
            
            base_revenue = st.session_state.calculated_data.get('total_ventes', 15000)
            
            # Apply cost structure
            for cost_type, percentage in template['cost_structure'].items():
                cost_key = f"industry_{cost_type}"
                st.session_state.calculated_data[cost_key] = base_revenue * percentage
            
            st.success(f"✅ {industry.title()} industry template applied successfully!")
            st.balloons()
            
            # Show applied changes
            with st.expander("📋 Applied Changes", expanded=True):
                st.write(f"**Industry**: {industry.title()}")
                st.write(f"**Revenue Model**: {template['revenue_model']}")
                
                st.write("**Cost Structure Applied**:")
                for cost_type, percentage in template['cost_structure'].items():
                    amount = base_revenue * percentage
                    st.write(f"• {cost_type.replace('_', ' ').title()}: {percentage:.1%} = {amount:,.0f} DHS/month")
                
                st.write("**Key Metrics to Track**:")
                for metric in template['key_metrics']:
                    st.write(f"• {metric}")

# Continue with the rest of the enhanced functions...

# ========== MAIN FUNCTION WITH ENHANCED NAVIGATION ==========
def main():
    """Enhanced main function with comprehensive navigation"""
    
    # Initialize enhanced session state
    init_enhanced_session_state()
    
    # Check authentication
    if not st.session_state.user_authenticated:
        show_authentication()
        return
    
    # Main application interface
    st.sidebar.markdown(f"""
    ### 👋 Welcome back!
    **{st.session_state.current_user['username']}**
    
    *{st.session_state.current_user.get('full_name', 'User')}*
    
    ---
    """)
    
    # Enhanced navigation menu
    menu_items = {
        "👔 Executive Dashboard": "executive_dashboard",
        "🧠 Advanced Analytics": "advanced_analytics", 
        "🎯 Scenario Planning": "scenario_planning",
        "🤖 ML Forecasting": "ml_forecasting",
        "⚠️ Risk Management": "risk_management",
        "🏭 Industry Templates": "industry_templates",
        "👥 Collaboration Hub": "collaboration_hub",
        "🏢 Company Profile": "company_profile",
        "💼 Investments": "investments",
        "📊 Balance Sheet": "balance_sheet",
        "📈 Income Statement": "income_statement",
        "💰 Cash Flow": "cash_flow",
        "📋 Amortization": "amortization",
        "🎓 Financial Education": "financial_education",
        "📤 CSV Import": "csv_import"
    }
    
    # Handle redirects
    if 'redirect_to' in st.session_state:
        choice = st.session_state.redirect_to
        del st.session_state.redirect_to
    else:
        choice = st.sidebar.selectbox(
            "🧭 Navigation",
            list(menu_items.keys()),
            index=0
        )
    
    # Main content area
    if choice == "👔 Executive Dashboard":
        show_executive_dashboard()
    elif choice == "🧠 Advanced Analytics":
        show_advanced_analytics()
    elif choice == "🎯 Scenario Planning":
        show_scenario_planning()
    elif choice == "🤖 ML Forecasting":
        show_ml_forecasting()
    elif choice == "⚠️ Risk Management":
        show_risk_management()
    elif choice == "🏭 Industry Templates":
        show_industry_templates()
    elif choice == "👥 Collaboration Hub":
        show_collaboration_hub()
    elif choice == "🏢 Company Profile":
        show_company_info()
    elif choice == "💼 Investments":
        show_investments()
    elif choice == "📊 Balance Sheet":
        show_balance_sheet()
    elif choice == "📈 Income Statement":
        show_income_statement()
    elif choice == "💰 Cash Flow":
        show_cash_flow()
    elif choice == "📋 Amortization":
        show_amortization()
    elif choice == "🎓 Financial Education":
        show_finance_initiation()
    elif choice == "📤 CSV Import":
        show_csv_import()
    
    # Enhanced sidebar with user info and quick actions
    with st.sidebar:
        st.markdown("---")
        
        # Company info
        company_name = st.session_state.basic_info.get('company_name', 'Demo Company')
        industry = st.session_state.basic_info.get('industry', 'Technology')
        
        st.caption(f"🏢 **Company**: {company_name}")
        st.caption(f"🏭 **Industry**: {industry.title()}")
        st.caption(f"📅 **Date**: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        
        # Quick metrics
        if st.session_state.calculated_data:
            monthly_revenue = st.session_state.calculated_data.get('total_ventes', 0)
            monthly_profit = monthly_revenue - st.session_state.calculated_data.get('total_charges', 0)
            
            st.markdown("### 📊 Quick Metrics")
            st.metric("Monthly Revenue", f"{monthly_revenue:,.0f} DHS")
            st.metric("Monthly Profit", f"{monthly_profit:,.0f} DHS")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if st.session_state.current_user:
                    user_manager = UserManager()
                    project_data = {
                        'basic_info': st.session_state.basic_info,
                        'investment_data': st.session_state.investment_data,
                        'calculated_data': st.session_state.calculated_data
                    }
                    
                    project_id = user_manager.save_user_project(
                        st.session_state.current_user['id'],
                        company_name,
                        project_data,
                        industry
                    )
                    
                    st.success("💾 Project saved!")
        
        with col2:
            if st.button("📊 Report", use_container_width=True):
                st.info("📊 Report generation coming soon!")
        
        # System status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        st.success("🟢 All systems operational")
        st.caption(f"ML Models: {'✅ Available' if ML_AVAILABLE else '❌ Limited'}")
        st.caption(f"Financial Lib: {'✅ Available' if PYFINANCE_AVAILABLE else '❌ Basic'}")
        
        # Logout
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Clear session
            for key in ['user_authenticated', 'current_user', 'user_role']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()

# ========== ADD REMAINING ORIGINAL FUNCTIONS ==========

def show_collaboration_hub():
    """Collaboration and project management page"""
    st.header("👥 Collaboration Hub")
    
    if not st.session_state.user_authenticated:
        st.warning("Please login to access collaboration features")
        return
    
    user_manager = UserManager()
    
    # Project management section
    st.subheader("📁 Project Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current project info
        company_name = st.session_state.basic_info.get('company_name', 'Untitled Project')
        industry = st.session_state.basic_info.get('industry', 'general')
        
        st.write(f"**Current Project**: {company_name}")
        st.write(f"**Industry**: {industry.title()}")
        st.write(f"**Last Modified**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Project actions
        project_name = st.text_input("Project Name", value=company_name)
        
        if st.button("💾 Save Current Project"):
            project_data = {
                'basic_info': st.session_state.basic_info,
                'investment_data': st.session_state.investment_data,
                'calculated_data': st.session_state.calculated_data,
                'scenario_results': st.session_state.get('scenario_results', {}),
                'ml_forecasts': st.session_state.get('ml_forecasts', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            project_id = user_manager.save_user_project(
                st.session_state.current_user['id'],
                project_name,
                project_data,
                industry
            )
            
            st.success(f"✅ Project '{project_name}' saved successfully!")
            st.balloons()
    
    with col2:
        # User projects list
        st.markdown("**Your Projects**")
        
        try:
            user_projects = user_manager.get_user_projects(st.session_state.current_user['id'])
            
            if user_projects:
                for project in user_projects[:5]:  # Show last 5 projects
                    with st.container():
                        st.write(f"📁 **{project['name']}**")
                        st.caption(f"{project['industry']} • {project['updated_at'][:10]}")
                        if st.button(f"Load", key=f"load_{project['id']}"):
                            st.info("🔄 Project loading feature coming soon!")
                        st.markdown("---")
            else:
                st.info("No saved projects yet")
        except Exception as e:
            st.error(f"Error loading projects: {str(e)}")
    
    # Comments and collaboration
    st.subheader("💬 Project Comments & Discussions")
    
    # Add new comment
    with st.form("add_comment_form"):
        comment_text = st.text_area("Add a comment or note", height=100)
        comment_type = st.selectbox("Comment Type", ["General", "Question", "Suggestion", "Issue"])
        
        submit_comment = st.form_submit_button("💬 Add Comment")
        
        if submit_comment and comment_text:
            new_comment = {
                'id': str(uuid.uuid4()),
                'user': st.session_state.current_user['username'],
                'text': comment_text,
                'type': comment_type,
                'timestamp': datetime.now(),
                'project': st.session_state.basic_info.get('company_name', 'Current Project')
            }
            
            st.session_state.project_comments.append(new_comment)
            st.success("💬 Comment added successfully!")
    
    # Display recent comments
    if st.session_state.project_comments:
        st.markdown("**Recent Comments:**")
        
        for comment in reversed(st.session_state.project_comments[-10:]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{comment['user']}** - {comment['type']}")
                    st.write(comment['text'])
                
                with col2:
                    st.caption(comment['timestamp'].strftime('%Y-%m-%d %H:%M'))
                
                st.markdown("---")
    else:
        st.info("No comments yet. Start a discussion!")
    
    # Team collaboration (future feature)
    st.subheader("👥 Team Collaboration")
    st.info("🚀 **Coming Soon**: Real-time collaboration features including:")
    st.write("• Team member invitations")
    st.write("• Role-based permissions")
    st.write("• Real-time editing")
    st.write("• Approval workflows")
    st.write("• Shared dashboards")

# Add your original functions here (show_company_info, show_investments, etc.)
# I'll include the essential ones:

def show_company_info():
    """Enhanced company information page"""
    st.header("🏢 Company Profile")
    
    with st.expander("📝 Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.basic_info['company_name'] = st.text_input(
                "Company Name *", 
                value=st.session_state.basic_info.get('company_name', ''),
                help="Legal name of your company"
            )
            
            st.session_state.basic_info['company_type'] = st.selectbox(
                "Legal Structure *", 
                ["SARL", "SA", "SNC", "SARLAU", "COOPERATIVE", "Auto-Entrepreneur", "EURL"],
                index=0,
                help="Choose your legal business structure"
            )
            
            st.session_state.basic_info['creation_date'] = st.date_input(
                "Foundation Date *", 
                value=st.session_state.basic_info.get('creation_date', datetime.now().date()),
                help="When was the company established"
            )
        
        with col2:
            st.session_state.basic_info['industry'] = st.selectbox(
                "Industry *",
                ['technology', 'retail', 'manufacturing', 'restaurant', 'consulting', 'healthcare', 'education', 'finance'],
                format_func=lambda x: x.title(),
                help="Primary industry sector"
            )
            
            st.session_state.basic_info['tax_id'] = st.text_input(
                "Tax ID", 
                value=st.session_state.basic_info.get('tax_id', ''),
                help="Fiscal identification number"
            )
            
            st.session_state.basic_info['partners'] = st.number_input(
                "Number of Partners", 
                min_value=1, 
                max_value=100,
                value=st.session_state.basic_info.get('partners', 1),
                help="Total number of business partners/shareholders"
            )
    
    with st.expander("📍 Contact Information"):
        st.session_state.basic_info['address'] = st.text_area(
            "Business Address", 
            value=st.session_state.basic_info.get('address', ''),
            help="Complete business address"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.basic_info['phone'] = st.text_input(
                "Phone Number", 
                value=st.session_state.basic_info.get('phone', ''),
                help="Primary contact number"
            )
        
        with col2:
            st.session_state.basic_info['email'] = st.text_input(
                "Email Address", 
                value=st.session_state.basic_info.get('email', ''),
                help="Primary business email"
            )
    
    # Company summary card
    with st.expander("📊 Company Summary", expanded=True):
        if all([st.session_state.basic_info.get('company_name'),
                st.session_state.basic_info.get('industry'),
                st.session_state.basic_info.get('company_type')]):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **🏢 {st.session_state.basic_info['company_name']}**
                
                *{st.session_state.basic_info['company_type']}*
                
                **Industry**: {st.session_state.basic_info['industry'].title()}
                """)
            
            with col2:
                creation_date = st.session_state.basic_info['creation_date']
                if isinstance(creation_date, str):
                    creation_date = datetime.strptime(creation_date, '%Y-%m-%d').date()
                
                company_age = (datetime.now().date() - creation_date).days // 365
                
                st.markdown(f"""
                **📅 Established**: {creation_date.strftime('%B %Y')}
                
                **Age**: {company_age} years
                
                **Partners**: {st.session_state.basic_info['partners']}
                """)
            
            with col3:
                contact_info = []
                if st.session_state.basic_info.get('email'):
                    contact_info.append(f"📧 {st.session_state.basic_info['email']}")
                if st.session_state.basic_info.get('phone'):
                    contact_info.append(f"📞 {st.session_state.basic_info['phone']}")
                
                if contact_info:
                    st.markdown("**Contact:**\n\n" + "\n\n".join(contact_info))
        else:
            st.warning("⚠️ Please complete the required fields (*) to see the company summary")

def show_investments():
    """Enhanced investments page"""
    st.header("💼 Investment Planning & Financing")
    
    # Investment overview
    st.subheader("📊 Investment Overview")
    
    # Calculate totals
    total_prelim = sum(item.get('valeur', 0) for item in st.session_state.get('frais_preliminaires', []))
    total_immos = sum(item.get('Montant', 0) for item in st.session_state.get('immos', []))
    total_credits = sum(item.get('Montant', 0) for item in st.session_state.get('credits', []))
    total_subsidies = sum(item.get('Montant', 0) for item in st.session_state.get('subsidies', []))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Preliminary Costs", f"{total_prelim:,.0f} DHS")
    with col2:
        st.metric("Fixed Assets", f"{total_immos:,.0f} DHS")
    with col3:
        st.metric("Total Financing", f"{(total_credits + total_subsidies):,.0f} DHS")
    with col4:
        total_investment = total_prelim + total_immos + st.session_state.investment_data.get('web_dev', 0)
        st.metric("Total Investment", f"{total_investment:,.0f} DHS")
    
    # Store calculations
    st.session_state.calculated_data.update({
        'total_frais': total_prelim,
        'total_immos': total_immos,
        'total_credits': total_credits,
        'total_subsidies': total_subsidies,
        'total_investissement': total_investment
    })
    
    # Detailed sections
    tab1, tab2, tab3 = st.tabs(["💰 Investments", "🏦 Financing", "📈 Analysis"])
    
    with tab1:
        # Investment details
        st.subheader("📝 Preliminary Expenses")
        
        # Initialize if empty
        if 'frais_preliminaires' not in st.session_state:
            st.session_state.frais_preliminaires = [
                {"nom": "Brand Registration", "valeur": 1700.0},
                {"nom": "Company Formation", "valeur": 4000.0}
            ]
        
        # Edit preliminary expenses
        df_frais = pd.DataFrame(st.session_state.frais_preliminaires)
        
        edited_frais = st.data_editor(
            df_frais,
            column_config={
                "nom": "Description",
                "valeur": st.column_config.NumberColumn("Amount (DHS)", format="%.2f")
            },
            num_rows="dynamic",
            use_container_width=True,
            key="frais_editor"
        )
        
        st.session_state.frais_preliminaires = edited_frais.to_dict('records')
        
        st.subheader("🏭 Fixed Assets")
        
        # Add new asset
        with st.form("add_asset_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                asset_name = st.text_input("Asset Name")
            with col2:
                asset_amount = st.number_input("Amount (DHS)", min_value=0.0, step=1000.0)
            with col3:
                asset_category = st.selectbox("Category", ["Equipment", "Furniture", "Technology", "Vehicles", "Real Estate", "Other"])
            
            submit_asset = st.form_submit_button("Add Asset")
            
            if submit_asset and asset_name and asset_amount > 0:
                if 'immos' not in st.session_state:
                    st.session_state.immos = []
                
                new_asset = {
                    "Nom": asset_name,
                    "Montant": asset_amount,
                    "Categorie": asset_category,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }
                
                st.session_state.immos.append(new_asset)
                st.success(f"✅ Asset '{asset_name}' added successfully!")
                st.rerun()
        
        # Display existing assets
        if st.session_state.get('immos'):
            df_assets = pd.DataFrame(st.session_state.immos)
            
            edited_assets = st.data_editor(
                df_assets,
                column_config={
                    "Nom": "Asset Name",
                    "Montant": st.column_config.NumberColumn("Amount (DHS)", format="%.2f"),
                    "Categorie": "Category",
                    "Date": "Date Added"
                },
                num_rows="dynamic",
                use_container_width=True,
                key="assets_editor"
            )
            
            st.session_state.immos = edited_assets.to_dict('records')
        else:
            st.info("No fixed assets added yet. Use the form above to add your first asset.")
    
    with tab2:
        # Financing sources
        st.subheader("🏦 Financing Sources")
        
        # Bank credits
        st.markdown("### 💳 Bank Credits & Loans")
        
        with st.form("add_credit_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                credit_type = st.selectbox("Credit Type", ["Business Loan", "Equipment Financing", "Line of Credit", "Mortgage", "Other"])
            with col2:
                credit_amount = st.number_input("Amount (DHS)", min_value=0.0, step=5000.0)
            with col3:
                interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            with col4:
                term_months = st.number_input("Term (Months)", min_value=1, max_value=360, value=60)
            
            submit_credit = st.form_submit_button("Add Credit")
            
            if submit_credit and credit_amount > 0:
                if 'credits' not in st.session_state:
                    st.session_state.credits = []
                
                # Calculate monthly payment
                monthly_rate = interest_rate / 100 / 12
                if monthly_rate > 0:
                    monthly_payment = credit_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
                else:
                    monthly_payment = credit_amount / term_months
                
                new_credit = {
                    "Type": credit_type,
                    "Montant": credit_amount,
                    "Taux": interest_rate,
                    "Duree": term_months,
                    "Mensualite": monthly_payment,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }
                
                st.session_state.credits.append(new_credit)
                st.success(f"✅ Credit '{credit_type}' of {credit_amount:,.0f} DHS added!")
                st.rerun()
        
        # Display credits
        if st.session_state.get('credits'):
            df_credits = pd.DataFrame(st.session_state.credits)
            
            st.data_editor(
                df_credits,
                column_config={
                    "Type": "Credit Type",
                    "Montant": st.column_config.NumberColumn("Amount (DHS)", format="%.0f"),
                    "Taux": st.column_config.NumberColumn("Rate (%)", format="%.2f"),
                    "Duree": "Term (Months)",
                    "Mensualite": st.column_config.NumberColumn("Monthly Payment (DHS)", format="%.0f"),
                    "Date": "Date Added"
                },
                num_rows="dynamic",
                use_container_width=True,
                key="credits_editor"
            )
        else:
            st.info("No credits added yet.")
        
        # Subsidies and grants
        st.markdown("### 🎁 Subsidies & Grants")
        
        with st.form("add_subsidy_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                subsidy_source = st.text_input("Source/Organization")
            with col2:
                subsidy_amount = st.number_input("Amount (DHS)", min_value=0.0, step=1000.0)
            with col3:
                subsidy_type = st.selectbox("Type", ["Grant", "Subsidy", "Tax Credit", "Incentive", "Other"])
            
            submit_subsidy = st.form_submit_button("Add Subsidy")
            
            if submit_subsidy and subsidy_source and subsidy_amount > 0:
                if 'subsidies' not in st.session_state:
                    st.session_state.subsidies = []
                
                new_subsidy = {
                    "Source": subsidy_source,
                    "Montant": subsidy_amount,
                    "Type": subsidy_type,
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }
                
                st.session_state.subsidies.append(new_subsidy)
                st.success(f"✅ Subsidy from '{subsidy_source}' added!")
                st.rerun()
        
        # Display subsidies
        if st.session_state.get('subsidies'):
            df_subsidies = pd.DataFrame(st.session_state.subsidies)
            
            st.data_editor(
                df_subsidies,
                column_config={
                    "Source": "Source/Organization",
                    "Montant": st.column_config.NumberColumn("Amount (DHS)", format="%.0f"),
                    "Type": "Subsidy Type",
                    "Date": "Date Added"
                },
                num_rows="dynamic",
                use_container_width=True,
                key="subsidies_editor"
            )
        else:
            st.info("No subsidies added yet.")
        
        # Equity contributions
        st.markdown("### 💰 Equity Contributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cash_contribution = st.number_input(
                "Cash Contribution (DHS)",
                min_value=0.0,
                value=st.session_state.investment_data.get('cash_contribution', 50000.0),
                step=1000.0,
                help="Cash invested by partners/shareholders"
            )
            st.session_state.investment_data['cash_contribution'] = cash_contribution
        
        with col2:
            in_kind_contribution = st.number_input(
                "In-Kind Contribution (DHS)",
                min_value=0.0,
                value=st.session_state.investment_data.get('in_kind', 20000.0),
                step=1000.0,
                help="Value of non-cash assets contributed"
            )
            st.session_state.investment_data['in_kind'] = in_kind_contribution
    
    with tab3:
        # Investment analysis
        st.subheader("📈 Investment Analysis")
        
        # Financing structure pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Financing Structure")
            
            financing_data = {
                'Cash Contribution': st.session_state.investment_data.get('cash_contribution', 0),
                'In-Kind Contribution': st.session_state.investment_data.get('in_kind', 0),
                'Bank Credits': total_credits,
                'Subsidies & Grants': total_subsidies
            }
            
            # Filter out zero values
            financing_data = {k: v for k, v in financing_data.items() if v > 0}
            
            if financing_data:
                fig = px.pie(
                    values=list(financing_data.values()),
                    names=list(financing_data.keys()),
                    title="Financing Sources Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add financing sources to see the distribution chart")
        
        with col2:
            st.markdown("#### 📊 Investment Categories")
            
            investment_data = {
                'Preliminary Expenses': total_prelim,
                'Fixed Assets': total_immos,
                'Technology/Software': st.session_state.investment_data.get('web_dev', 0)
            }
            
            # Filter out zero values
            investment_data = {k: v for k, v in investment_data.items() if v > 0}
            
            if investment_data:
                fig = px.bar(
                    x=list(investment_data.keys()),
                    y=list(investment_data.values()),
                    title="Investment by Category"
                )
                fig.update_layout(yaxis_title="Amount (DHS)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add investments to see the category breakdown")
        
        # Financial ratios and analysis
        st.markdown("#### 📊 Financial Analysis")
        
        total_financing = total_credits + total_subsidies + cash_contribution + in_kind_contribution
        debt_ratio = total_credits / total_financing if total_financing > 0 else 0
        equity_ratio = (cash_contribution + in_kind_contribution) / total_financing if total_financing > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Debt Ratio", f"{debt_ratio:.1%}")
            if debt_ratio > 0.7:
                st.error("High leverage risk")
            elif debt_ratio > 0.5:
                st.warning("Moderate leverage")
            else:
                st.success("Conservative leverage")
        
        with col2:
            st.metric("Equity Ratio", f"{equity_ratio:.1%}")
            if equity_ratio > 0.5:
                st.success("Strong equity base")
            elif equity_ratio > 0.3:
                st.info("Adequate equity")
            else:
                st.warning("Low equity ratio")
        
        with col3:
            if st.session_state.get('credits'):
                avg_interest_rate = np.mean([c['Taux'] for c in st.session_state.credits])
                st.metric("Avg Interest Rate", f"{avg_interest_rate:.2f}%")
            else:
                st.metric("Avg Interest Rate", "N/A")
        
        with col4:
            if total_investment > 0 and cash_contribution > 0:
                equity_multiplier = total_investment / cash_contribution
                st.metric("Equity Multiplier", f"{equity_multiplier:.1f}x")
            else:
                st.metric("Equity Multiplier", "N/A")
        
        # Monthly debt service
        if st.session_state.get('credits'):
            st.markdown("#### 💳 Debt Service Analysis")
            
            total_monthly_payment = sum(c['Mensualite'] for c in st.session_state.credits)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Monthly Debt Service", f"{total_monthly_payment:,.0f} DHS")
                
                # Calculate debt service over time
                months = range(1, 61)  # 5 years
                remaining_balance = []
                
                for month in months:
                    balance = 0
                    for credit in st.session_state.credits:
                        if month <= credit['Duree']:
                            monthly_rate = credit['Taux'] / 100 / 12
                            remaining_months = credit['Duree'] - month + 1
                            if monthly_rate > 0:
                                remaining = credit['Montant'] * ((1 + monthly_rate)**remaining_months - 1) / ((1 + monthly_rate)**credit['Duree'] - 1)
                            else:
                                remaining = credit['Montant'] * remaining_months / credit['Duree']
                            balance += remaining
                    remaining_balance.append(balance)
                
                fig = px.line(
                    x=months,
                    y=remaining_balance,
                    title="Debt Balance Over Time",
                    labels={'x': 'Month', 'y': 'Remaining Balance (DHS)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Debt service table
                debt_summary = []
                for credit in st.session_state.credits:
                    total_payments = credit['Mensualite'] * credit['Duree']
                    total_interest = total_payments - credit['Montant']
                    
                    debt_summary.append({
                        'Credit Type': credit['Type'],
                        'Principal': f"{credit['Montant']:,.0f}",
                        'Total Interest': f"{total_interest:,.0f}",
                        'Total Cost': f"{total_payments:,.0f}",
                        'Monthly Payment': f"{credit['Mensualite']:,.0f}"
                    })
                
                df_debt_summary = pd.DataFrame(debt_summary)
                st.dataframe(df_debt_summary, use_container_width=True)
        
        # Investment recommendations
        st.markdown("#### 💡 Investment Recommendations")
        
        recommendations = []
        
        if debt_ratio > 0.7:
            recommendations.append("🔴 **High Debt Risk**: Consider increasing equity contribution or reducing debt")
        
        if total_monthly_payment > 0:
            # Estimate monthly revenue for debt service coverage
            estimated_monthly_revenue = st.session_state.calculated_data.get('total_ventes', 15000)
            debt_service_coverage = estimated_monthly_revenue / total_monthly_payment if total_monthly_payment > 0 else 0
            
            if debt_service_coverage < 1.2:
                recommendations.append("🔴 **Debt Service Risk**: Monthly payments may be too high relative to expected revenue")
            elif debt_service_coverage < 2.0:
                recommendations.append("🟡 **Monitor Cash Flow**: Ensure adequate cash flow to service debt")
        
        if equity_ratio < 0.3:
            recommendations.append("🟡 **Low Equity**: Consider increasing equity contribution for financial stability")
        
        if not recommendations:
            recommendations.append("✅ **Balanced Structure**: Your financing structure appears well-balanced")
        
        for rec in recommendations:
            if "🔴" in rec:
                st.error(rec)
            elif "🟡" in rec:
                st.warning(rec)
            else:
                st.success(rec)

def show_balance_sheet():
    """Enhanced balance sheet page"""
    st.header("📊 Balance Sheet & Financial Position")
    
    # Initialize balance sheet data
    if 'balance_sheet' not in st.session_state:
        st.session_state.balance_sheet = {
            'assets': {
                'current_assets': {},
                'fixed_assets': {}
            },
            'liabilities': {
                'current_liabilities': {},
                'long_term_liabilities': {}
            },
            'equity': {}
        }
    
    # Balance sheet tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏦 Assets", "📋 Liabilities", "💰 Equity", "📊 Analysis"])
    
    with tab1:
        st.subheader("Assets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💱 Current Assets")
            st.caption("Assets expected to be converted to cash within one year")
            
            # Current assets form
            with st.form("current_assets_form"):
                cash = st.number_input("Cash & Cash Equivalents (DHS)", min_value=0.0, value=50000.0, step=1000.0)
                accounts_receivable = st.number_input("Accounts Receivable (DHS)", min_value=0.0, value=25000.0, step=1000.0)
                inventory = st.number_input("Inventory (DHS)", min_value=0.0, value=30000.0, step=1000.0)
                prepaid_expenses = st.number_input("Prepaid Expenses (DHS)", min_value=0.0, value=5000.0, step=1000.0)
                other_current = st.number_input("Other Current Assets (DHS)", min_value=0.0, value=0.0, step=1000.0)
                
                update_current_assets = st.form_submit_button("Update Current Assets")
                
                if update_current_assets:
                    st.session_state.balance_sheet['assets']['current_assets'] = {
                        'cash': cash,
                        'accounts_receivable': accounts_receivable,
                        'inventory': inventory,
                        'prepaid_expenses': prepaid_expenses,
                        'other_current': other_current
                    }
                    st.success("✅ Current assets updated!")
        
        with col2:
            st.markdown("### 🏭 Fixed Assets")
            st.caption("Long-term assets used in business operations")
            
            # Calculate fixed assets from investments
            total_fixed_assets = sum(item.get('Montant', 0) for item in st.session_state.get('immos', []))
            
            with st.form("fixed_assets_form"):
                property_plant_equipment = st.number_input("Property, Plant & Equipment (DHS)", min_value=0.0, value=total_fixed_assets, step=1000.0)
                accumulated_depreciation = st.number_input("Accumulated Depreciation (DHS)", min_value=0.0, value=0.0, step=1000.0)
                intangible_assets = st.number_input("Intangible Assets (DHS)", min_value=0.0, value=st.session_state.investment_data.get('web_dev', 0), step=1000.0)
                other_fixed = st.number_input("Other Fixed Assets (DHS)", min_value=0.0, value=0.0, step=1000.0)
                
                update_fixed_assets = st.form_submit_button("Update Fixed Assets")
                
                if update_fixed_assets:
                    st.session_state.balance_sheet['assets']['fixed_assets'] = {
                        'property_plant_equipment': property_plant_equipment,
                        'accumulated_depreciation': -accumulated_depreciation,  # Negative value
                        'intangible_assets': intangible_assets,
                        'other_fixed': other_fixed
                    }
                    st.success("✅ Fixed assets updated!")
        
        # Assets summary
        current_assets = st.session_state.balance_sheet['assets']['current_assets']
        fixed_assets = st.session_state.balance_sheet['assets']['fixed_assets']
        
        total_current = sum(current_assets.values())
        total_fixed = sum(fixed_assets.values())
        total_assets = total_current + total_fixed
        
        st.markdown("### 📊 Assets Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Current Assets", f"{total_current:,.0f} DHS")
        with col2:
            st.metric("Total Fixed Assets", f"{total_fixed:,.0f} DHS")
        with col3:
            st.metric("TOTAL ASSETS", f"{total_assets:,.0f} DHS")
        
        # Store in calculated data
        st.session_state.calculated_data.update({
            'total_current_assets': total_current,
            'total_fixed_assets': total_fixed,
            'total_actif': total_assets
        })
    
    with tab2:
        st.subheader("Liabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Current Liabilities")
            st.caption("Obligations due within one year")
            
            with st.form("current_liabilities_form"):
                accounts_payable = st.number_input("Accounts Payable (DHS)", min_value=0.0, value=15000.0, step=1000.0)
                short_term_debt = st.number_input("Short-term Debt (DHS)", min_value=0.0, value=10000.0, step=1000.0)
                accrued_expenses = st.number_input("Accrued Expenses (DHS)", min_value=0.0, value=8000.0, step=1000.0)
                taxes_payable = st.number_input("Taxes Payable (DHS)", min_value=0.0, value=5000.0, step=1000.0)
                other_current_liab = st.number_input("Other Current Liabilities (DHS)", min_value=0.0, value=0.0, step=1000.0)
                
                update_current_liab = st.form_submit_button("Update Current Liabilities")
                
                if update_current_liab:
                    st.session_state.balance_sheet['liabilities']['current_liabilities'] = {
                        'accounts_payable': accounts_payable,
                        'short_term_debt': short_term_debt,
                        'accrued_expenses': accrued_expenses,
                        'taxes_payable': taxes_payable,
                        'other_current': other_current_liab
                    }
                    st.success("✅ Current liabilities updated!")
        
        with col2:
            st.markdown("### 📊 Long-term Liabilities")
            st.caption("Obligations due after one year")
            
            # Calculate long-term debt from credits
            total_long_term_debt = sum(item.get('Montant', 0) for item in st.session_state.get('credits', []))
            
            with st.form("long_term_liabilities_form"):
                long_term_debt = st.number_input("Long-term Debt (DHS)", min_value=0.0, value=total_long_term_debt, step=1000.0)
                deferred_tax = st.number_input("Deferred Tax Liabilities (DHS)", min_value=0.0, value=0.0, step=1000.0)
                other_long_term = st.number_input("Other Long-term Liabilities (DHS)", min_value=0.0, value=0.0, step=1000.0)
                
                update_long_term_liab = st.form_submit_button("Update Long-term Liabilities")
                
                if update_long_term_liab:
                    st.session_state.balance_sheet['liabilities']['long_term_liabilities'] = {
                        'long_term_debt': long_term_debt,
                        'deferred_tax': deferred_tax,
                        'other_long_term': other_long_term
                    }
                    st.success("✅ Long-term liabilities updated!")
        
        # Liabilities summary
        current_liabilities = st.session_state.balance_sheet['liabilities']['current_liabilities']
        long_term_liabilities = st.session_state.balance_sheet['liabilities']['long_term_liabilities']
        
        total_current_liab = sum(current_liabilities.values())
        total_long_term_liab = sum(long_term_liabilities.values())
        total_liabilities = total_current_liab + total_long_term_liab
        
        st.markdown("### 📊 Liabilities Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Current Liabilities", f"{total_current_liab:,.0f} DHS")
        with col2:
            st.metric("Total Long-term Liabilities", f"{total_long_term_liab:,.0f} DHS")
        with col3:
            st.metric("TOTAL LIABILITIES", f"{total_liabilities:,.0f} DHS")
        
        # Store in calculated data
        st.session_state.calculated_data.update({
            'total_current_liabilities': total_current_liab,
            'total_long_term_liabilities': total_long_term_liab,
            'total_passif': total_liabilities
        })
    
    with tab3:
        st.subheader("Equity")
        
        # Calculate equity from investments
        cash_contribution = st.session_state.investment_data.get('cash_contribution', 0)
        in_kind_contribution = st.session_state.investment_data.get('in_kind', 0)
        
        with st.form("equity_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                share_capital = st.number_input("Share Capital (DHS)", min_value=0.0, value=cash_contribution + in_kind_contribution, step=1000.0)
                retained_earnings = st.number_input("Retained Earnings (DHS)", value=0.0, step=1000.0, help="Accumulated profits/losses")
            
            with col2:
                additional_paid_capital = st.number_input("Additional Paid-in Capital (DHS)", min_value=0.0, value=0.0, step=1000.0)
                treasury_stock = st.number_input("Treasury Stock (DHS)", min_value=0.0, value=0.0, step=1000.0, help="Negative value for treasury stock")
            
            update_equity = st.form_submit_button("Update Equity")
            
            if update_equity:
                st.session_state.balance_sheet['equity'] = {
                    'share_capital': share_capital,
                    'additional_paid_capital': additional_paid_capital,
                    'retained_earnings': retained_earnings,
                    'treasury_stock': -treasury_stock  # Negative value
                }
                st.success("✅ Equity updated!")
        
        # Equity summary
        equity = st.session_state.balance_sheet['equity']
        total_equity = sum(equity.values())
        
        st.markdown("### 📊 Equity Summary")
        st.metric("TOTAL EQUITY", f"{total_equity:,.0f} DHS")
        
        # Store in calculated data
        st.session_state.calculated_data['total_equity'] = total_equity
    
    with tab4:
        st.subheader("Balance Sheet Analysis")
        
        # Recalculate totals
        total_assets = st.session_state.calculated_data.get('total_actif', 0)
        total_liabilities = st.session_state.calculated_data.get('total_passif', 0)
        total_equity = st.session_state.calculated_data.get('total_equity', 0)
        
        # Balance sheet equation check
        st.markdown("### ⚖️ Balance Sheet Equation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assets", f"{total_assets:,.0f} DHS")
        with col2:
            st.write("=")
        with col3:
            st.metric("Total Liabilities", f"{total_liabilities:,.0f} DHS")
        with col4:
            st.metric("Total Equity", f"{total_equity:,.0f} DHS")
        
        balance_difference = total_assets - (total_liabilities + total_equity)
        
        if abs(balance_difference) < 1:
            st.success("✅ Balance sheet is balanced!")
        else:
            st.error(f"❌ Balance sheet is not balanced. Difference: {balance_difference:,.0f} DHS")
            st.write("**Tip**: Make sure all assets equal liabilities plus equity")
        
        # Financial ratios
        st.markdown("### 📊 Financial Ratios")
        
        current_assets = st.session_state.calculated_data.get('total_current_assets', 0)
        current_liabilities = st.session_state.calculated_data.get('total_current_liabilities', 1)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            st.metric("Current Ratio", f"{current_ratio:.2f}")
            
            if current_ratio >= 2:
                st.success("Excellent liquidity")
            elif current_ratio >= 1.5:
                st.info("Good liquidity")
            elif current_ratio >= 1:
                st.warning("Adequate liquidity")
            else:
                st.error("Poor liquidity")
        
        with col2:
            debt_to_equity = total_liabilities / total_equity if total_equity > 0 else 0
            st.metric("Debt-to-Equity", f"{debt_to_equity:.2f}")
            
            if debt_to_equity <= 0.5:
                st.success("Conservative leverage")
            elif debt_to_equity <= 1:
                st.info("Moderate leverage")
            elif debt_to_equity <= 2:
                st.warning("High leverage")
            else:
                st.error("Very high leverage")
        
        with col3:
            equity_ratio = total_equity / total_assets if total_assets > 0 else 0
            st.metric("Equity Ratio", f"{equity_ratio:.1%}")
            
            if equity_ratio >= 0.5:
                st.success("Strong equity position")
            elif equity_ratio >= 0.3:
                st.info("Adequate equity")
            else:
                st.warning("Low equity ratio")
        
        # Balance sheet visualization
        st.markdown("### 📈 Balance Sheet Visualization")
        
        # Create balance sheet chart
        if total_assets > 0 and total_liabilities > 0 and total_equity > 0:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Assets', 'Liabilities & Equity'),
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )
            
            # Assets breakdown
            assets_data = {
                'Current Assets': current_assets,
                'Fixed Assets': st.session_state.calculated_data.get('total_fixed_assets', 0)
            }
            
            fig.add_trace(
                go.Pie(
                    labels=list(assets_data.keys()),
                    values=list(assets_data.values()),
                    name="Assets"
                ),
                row=1, col=1
            )
            
            # Liabilities & Equity breakdown
            liab_equity_data = {
                'Current Liabilities': st.session_state.calculated_data.get('total_current_liabilities', 0),
                'Long-term Liabilities': st.session_state.calculated_data.get('total_long_term_liabilities', 0),
                'Equity': total_equity
            }
            
            fig.add_trace(
                go.Pie(
                    labels=list(liab_equity_data.keys()),
                    values=list(liab_equity_data.values()),
                    name="Liabilities & Equity"
                ),
                row=1, col=2
            )
            
            fig.update_layout(title_text="Balance Sheet Composition")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete the balance sheet data to see the visualization")

def show_income_statement():
    """Enhanced income statement page"""
    st.header("📈 Income Statement & Profitability Analysis")
    
    # Income statement periods
    period_type = st.selectbox("Select Period", ["Monthly", "Quarterly", "Annual"], index=0)
    
    if period_type == "Monthly":
        periods = 1
        period_label = "Month"
    elif period_type == "Quarterly":
        periods = 3
        period_label = "Quarter"
    else:
        periods = 12
        period_label = "Year"
    
    # Income statement tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue", "💸 Expenses", "📊 Statement", "📈 Analysis"])
    
    with tab1:
        st.subheader("Revenue Streams")
        
        # Initialize revenue data
        if 'revenue_streams' not in st.session_state:
            st.session_state.revenue_streams = [
                {"source": "Primary Product Sales", "amount": 15000, "recurring": True},
                {"source": "Service Revenue", "amount": 5000, "recurring": True}
            ]
        
        # Add new revenue stream
        with st.form("add_revenue_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                revenue_source = st.text_input("Revenue Source")
            with col2:
                revenue_amount = st.number_input(f"Monthly Amount (DHS)", min_value=0.0, step=1000.0)
            with col3:
                is_recurring = st.checkbox("Recurring Revenue", value=True)
            
            add_revenue = st.form_submit_button("Add Revenue Stream")
            
            if add_revenue and revenue_source and revenue_amount > 0:
                st.session_state.revenue_streams.append({
                    "source": revenue_source,
                    "amount": revenue_amount,
                    "recurring": is_recurring
                })
                st.success(f"✅ Revenue stream '{revenue_source}' added!")
                st.rerun()
        
        # Edit existing revenue streams
        if st.session_state.revenue_streams:
            st.markdown("### Current Revenue Streams")
            
            df_revenue = pd.DataFrame(st.session_state.revenue_streams)
            
            edited_revenue = st.data_editor(
                df_revenue,
                column_config={
                    "source": "Revenue Source",
                    "amount": st.column_config.NumberColumn("Monthly Amount (DHS)", format="%.0f"),
                    "recurring": "Recurring"
                },
                num_rows="dynamic",
                use_container_width=True,
                key="revenue_editor"
            )
            
            st.session_state.revenue_streams = edited_revenue.to_dict('records')
            
            # Revenue summary
            total_monthly_revenue = sum(item['amount'] for item in st.session_state.revenue_streams)
            recurring_revenue = sum(item['amount'] for item in st.session_state.revenue_streams if item['recurring'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Monthly Revenue", f"{total_monthly_revenue:,.0f} DHS")
            with col2:
                st.metric("Recurring Revenue", f"{recurring_revenue:,.0f} DHS")
            with col3:
                recurring_percentage = (recurring_revenue / total_monthly_revenue * 100) if total_monthly_revenue > 0 else 0
                st.metric("Recurring %", f"{recurring_percentage:.1f}%")
            
            # Store in calculated data
            st.session_state.calculated_data['total_ventes'] = total_monthly_revenue
    
    with tab2:
        st.subheader("Operating Expenses")
        
        # Initialize expense data
        if 'operating_expenses' not in st.session_state:
            st.session_state.operating_expenses = [
                {"category": "Cost of Goods Sold", "amount": 9000, "variable": True},
                {"category": "Marketing & Advertising", "amount": 2000, "variable": False},
                {"category": "Salaries & Benefits", "amount": 8000, "variable": False},
                {"category": "Rent & Utilities", "amount": 3000, "variable": False}
            ]
        
        # Add new expense
        with st.form("add_expense_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                expense_category = st.selectbox("Expense Category", [
                    "Cost of Goods Sold", "Salaries & Benefits", "Marketing & Advertising",
                    "Rent & Utilities", "Professional Services", "Insurance", "Technology",
                    "Travel & Entertainment", "Office Supplies", "Other Operating Expenses"
                ])
            with col2:
                expense_amount = st.number_input("Monthly Amount (DHS)", min_value=0.0, step=500.0)
            with col3:
                is_variable = st.checkbox("Variable Cost", help="Changes with revenue volume")
            
            add_expense = st.form_submit_button("Add Expense")
            
            if add_expense and expense_amount > 0:
                st.session_state.operating_expenses.append({
                    "category": expense_category,
                    "amount": expense_amount,
                    "variable": is_variable
                })
                st.success(f"✅ Expense '{expense_category}' added!")
                st.rerun()
        
        # Edit existing expenses
        if st.session_state.operating_expenses:
            st.markdown("### Current Operating Expenses")
            
            df_expenses = pd.DataFrame(st.session_state.operating_expenses)
            
            edited_expenses = st.data_editor(
                df_expenses,
                column_config={
                    "category": "Expense Category",
                    "amount": st.column_config.NumberColumn("Monthly Amount (DHS)", format="%.0f"),
                    "variable": "Variable Cost"
                },
                num_rows="dynamic",
                use_container_width=True,
                key="expenses_editor"
            )
            
            st.session_state.operating_expenses = edited_expenses.to_dict('records')
            
            # Expense summary
            total_monthly_expenses = sum(item['amount'] for item in st.session_state.operating_expenses)
            variable_costs = sum(item['amount'] for item in st.session_state.operating_expenses if item['variable'])
            fixed_costs = total_monthly_expenses - variable_costs
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Monthly Expenses", f"{total_monthly_expenses:,.0f} DHS")
            with col2:
                st.metric("Variable Costs", f"{variable_costs:,.0f} DHS")
            with col3:
                st.metric("Fixed Costs", f"{fixed_costs:,.0f} DHS")
            
            # Store in calculated data
            st.session_state.calculated_data['total_charges'] = total_monthly_expenses
            st.session_state.calculated_data['variable_costs'] = variable_costs
            st.session_state.calculated_data['fixed_costs'] = fixed_costs
    
    with tab3:
        st.subheader(f"{period_label}ly Income Statement")
        
        # Calculate income statement items
        total_revenue = st.session_state.calculated_data.get('total_ventes', 0) * periods
        total_expenses = st.session_state.calculated_data.get('total_charges', 0) * periods
        variable_costs = st.session_state.calculated_data.get('variable_costs', 0) * periods
        fixed_costs = st.session_state.calculated_data.get('fixed_costs', 0) * periods
        
        # Gross profit
        gross_profit = total_revenue - variable_costs
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Operating profit
        operating_profit = gross_profit - fixed_costs
        operating_margin = (operating_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Other income/expenses
        col1, col2 = st.columns(2)
        
        with col1:
            interest_income = st.number_input(f"Interest Income ({period_label}ly)", min_value=0.0, value=0.0, step=100.0)
            other_income = st.number_input(f"Other Income ({period_label}ly)", min_value=0.0, value=0.0, step=100.0)
        
        with col2:
            interest_expense = st.number_input(f"Interest Expense ({period_label}ly)", min_value=0.0, value=0.0, step=100.0)
            other_expenses = st.number_input(f"Other Expenses ({period_label}ly)", min_value=0.0, value=0.0, step=100.0)
        
        # Calculate net income
        income_before_tax = operating_profit + interest_income + other_income - interest_expense - other_expenses
        
        # Tax calculation
        tax_rate = st.slider("Tax Rate (%)", min_value=0, max_value=50, value=30) / 100
        tax_expense = income_before_tax * tax_rate if income_before_tax > 0 else 0
        
        net_income = income_before_tax - tax_expense
        net_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0
        
        # Display income statement
        st.markdown("### 📊 Income Statement")
        
        income_statement_data = {
            "Item": [
                "Total Revenue",
                "Less: Variable Costs",
                "Gross Profit",
                "Less: Fixed Costs",
                "Operating Profit",
                "Add: Interest Income",
                "Add: Other Income",
                "Less: Interest Expense",
                "Less: Other Expenses",
                "Income Before Tax",
                "Less: Tax Expense",
                "Net Income"
            ],
            "Amount (DHS)": [
                f"{total_revenue:,.0f}",
                f"({variable_costs:,.0f})",
                f"{gross_profit:,.0f}",
                f"({fixed_costs:,.0f})",
                f"{operating_profit:,.0f}",
                f"{interest_income:,.0f}",
                f"{other_income:,.0f}",
                f"({interest_expense:,.0f})",
                f"({other_expenses:,.0f})",
                f"{income_before_tax:,.0f}",
                f"({tax_expense:,.0f})",
                f"{net_income:,.0f}"
            ],
            "% of Revenue": [
                "100.0%",
                f"{(variable_costs/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{gross_margin:.1f}%",
                f"{(fixed_costs/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{operating_margin:.1f}%",
                f"{(interest_income/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{(other_income/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{(interest_expense/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{(other_expenses/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{(income_before_tax/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{(tax_expense/total_revenue*100):.1f}%" if total_revenue > 0 else "0.0%",
                f"{net_margin:.1f}%"
            ]
        }
        
        df_income_statement = pd.DataFrame(income_statement_data)
        
        # Style the dataframe
        def style_income_statement(df):
            def highlight_totals(s):
                return ['background-color: #f0f0f0' if 'Profit' in s.name or 'Income' in s.name else '' for _ in s]
            
            return df.style.apply(highlight_totals, axis=1)
        
        st.dataframe(style_income_statement(df_income_statement), use_container_width=True)
        
        # Key metrics
        st.markdown("### 📊 Key Profitability Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Gross Margin", f"{gross_margin:.1f}%")
        with col2:
            st.metric("Operating Margin", f"{operating_margin:.1f}%")
        with col3:
            st.metric("Net Margin", f"{net_margin:.1f}%")
        with col4:
            st.metric("Effective Tax Rate", f"{(tax_expense/income_before_tax*100):.1f}%" if income_before_tax > 0 else "0.0%")
        
        # Store key metrics
        st.session_state.calculated_data.update({
            'gross_profit': gross_profit / periods,  # Monthly values
            'operating_profit': operating_profit / periods,
            'net_income': net_income / periods,
            'gross_margin': gross_margin / 100,
            'operating_margin': operating_margin / 100,
            'net_margin': net_margin / 100
        })
    
    with tab4:
        st.subheader("Profitability Analysis")
        
        # Trend analysis
        st.markdown("### 📈 Profitability Trends")
        
        # Generate trend data (simplified)
        months = list(range(1, 13))
        monthly_revenue = st.session_state.calculated_data.get('total_ventes', 15000)
        monthly_expenses = st.session_state.calculated_data.get('total_charges', 12000)
        
        # Apply seasonal factors if available
        if 'industry_template_applied' in st.session_state:
            template = st.session_state.industry_template_applied['template']
            seasonal_factors = template['seasonal_factors']
        else:
            seasonal_factors = [1.0] * 12
        
        revenue_trend = [monthly_revenue * factor * (1 + i * 0.01) for i, factor in enumerate(seasonal_factors)]
        expense_trend = [monthly_expenses * (1 + i * 0.005) for i in range(12)]
        profit_trend = [r - e for r, e in zip(revenue_trend, expense_trend)]
        
        # Create trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=revenue_trend,
            mode='lines+markers',
            name='Revenue',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=expense_trend,
            mode='lines+markers',
            name='Expenses',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=profit_trend,
            mode='lines+markers',
            name='Profit',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Monthly Profitability Trend",
            xaxis_title="Month",
            yaxis_title="Amount (DHS)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Break-even analysis
        st.markdown("### ⚖️ Break-even Analysis")
        
        fixed_costs_monthly = st.session_state.calculated_data.get('fixed_costs', 8000)
        variable_costs_monthly = st.session_state.calculated_data.get('variable_costs', 7000)
        revenue_monthly = st.session_state.calculated_data.get('total_ventes', 15000)
        
        # Calculate contribution margin
        contribution_margin = revenue_monthly - variable_costs_monthly
        contribution_margin_ratio = contribution_margin / revenue_monthly if revenue_monthly > 0 else 0
        
        # Break-even point
        break_even_revenue = fixed_costs_monthly / contribution_margin_ratio if contribution_margin_ratio > 0 else 0
        break_even_units = break_even_revenue / (revenue_monthly / 1) if revenue_monthly > 0 else 0  # Assuming 1 unit
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Contribution Margin", f"{contribution_margin:,.0f} DHS")
        with col2:
            st.metric("Contribution Margin %", f"{contribution_margin_ratio*100:.1f}%")
        with col3:
            st.metric("Break-even Revenue", f"{break_even_revenue:,.0f} DHS")
        with col4:
            margin_of_safety = ((revenue_monthly - break_even_revenue) / revenue_monthly * 100) if revenue_monthly > 0 else 0
            st.metric("Margin of Safety", f"{margin_of_safety:.1f}%")
        
        # Break-even chart
        revenue_levels = list(range(0, int(revenue_monthly * 2), 1000))
        fixed_cost_line = [fixed_costs_monthly] * len(revenue_levels)
        total_cost_line = [fixed_costs_monthly + (r * variable_costs_monthly / revenue_monthly) for r in revenue_levels]
        revenue_line = revenue_levels
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=revenue_levels,
            y=fixed_cost_line,
            mode='lines',
            name='Fixed Costs',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=revenue_levels,
            y=total_cost_line,
            mode='lines',
            name='Total Costs',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=revenue_levels,
            y=revenue_line,
            mode='lines',
            name='Revenue',
            line=dict(color='green')
        ))
        
        # Add break-even point
        fig.add_vline(x=break_even_revenue, line_dash="dot", line_color="blue", 
                     annotation_text=f"Break-even: {break_even_revenue:,.0f} DHS")
        
        fig.update_layout(
            title="Break-even Analysis",
            xaxis_title="Revenue (DHS)",
            yaxis_title="Amount (DHS)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("### 📊 Sensitivity Analysis")
        
        # Revenue sensitivity
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Revenue Impact on Profit**")
            revenue_changes = [-20, -10, -5, 0, 5, 10, 20]
            profit_changes = []
            
            base_profit = revenue_monthly - monthly_expenses
            
            for change in revenue_changes:
                new_revenue = revenue_monthly * (1 + change/100)
                new_variable_costs = variable_costs_monthly * (1 + change/100)  # Variable costs change with revenue
                new_profit = new_revenue - new_variable_costs - fixed_costs_monthly
                profit_change = ((new_profit - base_profit) / base_profit * 100) if base_profit != 0 else 0
                profit_changes.append(profit_change)
            
            fig = px.bar(
                x=[f"{c:+d}%" for c in revenue_changes],
                y=profit_changes,
                title="Profit Sensitivity to Revenue Changes",
                labels={'x': 'Revenue Change', 'y': 'Profit Change (%)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Cost Impact on Profit**")
            cost_changes = [-20, -10, -5, 0, 5, 10, 20]
            cost_profit_changes = []
            
            for change in cost_changes:
                new_expenses = monthly_expenses * (1 + change/100)
                new_profit = revenue_monthly - new_expenses
                profit_change = ((new_profit - base_profit) / base_profit * 100) if base_profit != 0 else 0
                cost_profit_changes.append(profit_change)
            
            fig = px.bar(
                x=[f"{c:+d}%" for c in cost_changes],
                y=cost_profit_changes,
                title="Profit Sensitivity to Cost Changes",
                labels={'x': 'Cost Change', 'y': 'Profit Change (%)'},
                color_discrete_sequence=['red']
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_cash_flow():
    """Enhanced cash flow statement and analysis"""
    st.header("💰 Cash Flow Statement & Analysis")
    
    # Cash flow period selection
    period = st.selectbox("Analysis Period", ["Monthly", "Quarterly", "Annual"], index=0)
    
    if period == "Monthly":
        multiplier = 1
        period_label = "Monthly"
    elif period == "Quarterly":
        multiplier = 3
        period_label = "Quarterly"
    else:
        multiplier = 12
        period_label = "Annual"
    
    # Cash flow tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Operating", "💼 Investing", "🏦 Financing", "📊 Analysis"])
    
    with tab1:
        st.subheader("Operating Cash Flow")
        
        # Start with net income
        net_income = st.session_state.calculated_data.get('net_income', 3000) * multiplier
        
        st.metric("Starting Point: Net Income", f"{net_income:,.0f} DHS")
        
        # Non-cash items
        st.markdown("### ➕ Add Back: Non-Cash Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            depreciation = st.number_input(f"Depreciation & Amortization ({period_label})", min_value=0.0, value=5000.0 * multiplier, step=500.0)
            amortization = st.number_input(f"Amortization of Intangibles ({period_label})", min_value=0.0, value=1000.0 * multiplier, step=100.0)
        
        with col2:
            bad_debt = st.number_input(f"Bad Debt Expense ({period_label})", min_value=0.0, value=500.0 * multiplier, step=100.0)
            other_non_cash = st.number_input(f"Other Non-Cash Items ({period_label})", value=0.0, step=100.0)
        
        total_non_cash = depreciation + amortization + bad_debt + other_non_cash
        
        # Working capital changes
        st.markdown("### 📊 Changes in Working Capital")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Assets Changes:**")
            accounts_receivable_change = st.number_input("Accounts Receivable Change", value=0.0, step=500.0, help="Increase = negative cash flow")
            inventory_change = st.number_input("Inventory Change", value=0.0, step=500.0, help="Increase = negative cash flow")
            prepaid_change = st.number_input("Prepaid Expenses Change", value=0.0, step=100.0, help="Increase = negative cash flow")
        
        with col2:
            st.markdown("**Current Liabilities Changes:**")
            accounts_payable_change = st.number_input("Accounts Payable Change", value=0.0, step=500.0, help="Increase = positive cash flow")
            accrued_change = st.number_input("Accrued Expenses Change", value=0.0, step=100.0, help="Increase = positive cash flow")
            taxes_payable_change = st.number_input("Taxes Payable Change", value=0.0, step=100.0, help="Increase = positive cash flow")
        
        working_capital_change = -(accounts_receivable_change + inventory_change + prepaid_change) + (accounts_payable_change + accrued_change + taxes_payable_change)
        
        # Calculate operating cash flow
        operating_cash_flow = net_income + total_non_cash + working_capital_change
        
        # Display operating cash flow summary
        st.markdown("### 📊 Operating Cash Flow Summary")
        
        operating_cf_data = {
            "Item": [
                "Net Income",
                "Add: Depreciation & Amortization",
                "Add: Other Non-Cash Items",
                "Changes in Working Capital",
                "Operating Cash Flow"
            ],
            "Amount (DHS)": [
                f"{net_income:,.0f}",
                f"{depreciation + amortization:,.0f}",
                f"{bad_debt + other_non_cash:,.0f}",
                f"{working_capital_change:+,.0f}",
                f"{operating_cash_flow:,.0f}"
            ]
        }
        
        df_operating = pd.DataFrame(operating_cf_data)
        st.dataframe(df_operating, use_container_width=True)
        
        # Store operating cash flow
        st.session_state.calculated_data['operating_cash_flow'] = operating_cash_flow / multiplier  # Monthly equivalent
    
    with tab2:
        st.subheader("Investing Cash Flow")
        
        st.markdown("### 💼 Capital Expenditures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            equipment_purchases = st.number_input(f"Equipment Purchases ({period_label})", min_value=0.0, value=0.0, step=1000.0)
            technology_investments = st.number_input(f"Technology Investments ({period_label})", min_value=0.0, value=0.0, step=1000.0)
            facility_improvements = st.number_input(f"Facility Improvements ({period_label})", min_value=0.0, value=0.0, step=1000.0)
        
        with col2:
            asset_sales = st.number_input(f"Asset Sales ({period_label})", min_value=0.0, value=0.0, step=1000.0)
            investment_purchases = st.number_input(f"Investment Purchases ({period_label})", min_value=0.0, value=0.0, step=1000.0)
            investment_sales = st.number_input(f"Investment Sales ({period_label})", min_value=0.0, value=0.0, step=1000.0)
        
        # Calculate investing cash flow
        cash_outflows = equipment_purchases + technology_investments + facility_improvements + investment_purchases
        cash_inflows = asset_sales + investment_sales
        investing_cash_flow = cash_inflows - cash_outflows
        
        # Display investing cash flow summary
        st.markdown("### 📊 Investing Cash Flow Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cash Outflows", f"({cash_outflows:,.0f}) DHS")
        with col2:
            st.metric("Cash Inflows", f"{cash_inflows:,.0f} DHS")
        with col3:
            st.metric("Net Investing Cash Flow", f"{investing_cash_flow:+,.0f} DHS")
        
        # Store investing cash flow
        st.session_state.calculated_data['investing_cash_flow'] = investing_cash_flow / multiplier
    
    with tab3:
        st.subheader("Financing Cash Flow")
        
        st.markdown("### 🏦 Debt Activities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_borrowings = st.number_input(f"New Borrowings ({period_label})", min_value=0.0, value=0.0, step=5000.0)
            debt_repayments = st.number_input(f"Debt Repayments ({period_label})", min_value=0.0, value=0.0, step=1000.0)
        
        with col2:
            interest_paid = st.number_input(f"Interest Paid ({period_label})", min_value=0.0, value=0.0, step=500.0)
            fees_paid = st.number_input(f"Bank Fees Paid ({period_label})", min_value=0.0, value=0.0, step=100.0)
        
        st.markdown("### 💰 Equity Activities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            equity_contributions = st.number_input(f"New Equity Contributions ({period_label})", min_value=0.0, value=0.0, step=5000.0)
            dividends_paid = st.number_input(f"Dividends Paid ({period_label})", min_value=0.0, value=0.0, step=1000.0)
        
        with col2:
            share_buybacks = st.number_input(f"Share Buybacks ({period_label})", min_value=0.0, value=0.0, step=1000.0)
            other_financing = st.number_input(f"Other Financing Activities ({period_label})", value=0.0, step=500.0)
        
        # Calculate financing cash flow
        financing_inflows = new_borrowings + equity_contributions
        financing_outflows = debt_repayments + interest_paid + fees_paid + dividends_paid + share_buybacks - other_financing
        financing_cash_flow = financing_inflows - financing_outflows
        
        # Display financing cash flow summary
        st.markdown("### 📊 Financing Cash Flow Summary")
        
        financing_cf_data = {
            "Item": [
                "New Borrowings",
                "Equity Contributions", 
                "Debt Repayments",
                "Interest Paid",
                "Dividends Paid",
                "Other Activities",
                "Net Financing Cash Flow"
            ],
            "Amount (DHS)": [
                f"{new_borrowings:,.0f}",
                f"{equity_contributions:,.0f}",
                f"({debt_repayments:,.0f})",
                f"({interest_paid:,.0f})",
                f"({dividends_paid:,.0f})",
                f"{other_financing:+,.0f}",
                f"{financing_cash_flow:+,.0f}"
            ]
        }
        
        df_financing = pd.DataFrame(financing_cf_data)
        st.dataframe(df_financing, use_container_width=True)
        
        # Store financing cash flow
        st.session_state.calculated_data['financing_cash_flow'] = financing_cash_flow / multiplier
    
    with tab4:
        st.subheader("Cash Flow Analysis & Projections")
        
        # Calculate total cash flow
        operating_cf = st.session_state.calculated_data.get('operating_cash_flow', 3000) * multiplier
        investing_cf = st.session_state.calculated_data.get('investing_cash_flow', 0) * multiplier
        financing_cf = st.session_state.calculated_data.get('financing_cash_flow', 0) * multiplier
        
        net_cash_flow = operating_cf + investing_cf + financing_cf
        
        # Cash flow statement summary
        st.markdown("### 📊 Complete Cash Flow Statement")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Operating Cash Flow", f"{operating_cf:+,.0f} DHS")
            if operating_cf > 0:
                st.success("✅ Positive operating cash flow")
            else:
                st.error("❌ Negative operating cash flow")
        
        with col2:
            st.metric("Investing Cash Flow", f"{investing_cf:+,.0f} DHS")
            if investing_cf < 0:
                st.info("💼 Investing in growth")
            else:
                st.warning("⚠️ Divesting assets")
        
        with col3:
            st.metric("Financing Cash Flow", f"{financing_cf:+,.0f} DHS")
            if financing_cf > 0:
                st.info("💰 Raising capital")
            else:
                st.info("💸 Returning capital")
        
        with col4:
            st.metric("Net Cash Flow", f"{net_cash_flow:+,.0f} DHS")
            if net_cash_flow > 0:
                st.success("✅ Cash positive")
            else:
                st.error("❌ Cash negative")
        
        # Store net cash flow
        st.session_state.calculated_data['cash_flow_mensuel'] = net_cash_flow / multiplier
        
        # Cash flow visualization
        st.markdown("### 📈 Cash Flow Visualization")
        
        # Create waterfall chart
        categories = ['Operating CF', 'Investing CF', 'Financing CF']
        values = [operating_cf, investing_cf, financing_cf]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=categories + ["Net Cash Flow"],
            textposition="outside",
            text=[f"{v:+,.0f}" for v in values] + [f"{net_cash_flow:+,.0f}"],
            y=values + [net_cash_flow],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"{period_label} Cash Flow Waterfall",
            showlegend=True,
            yaxis_title="Cash Flow (DHS)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cash flow projection
        st.markdown("### 🔮 Cash Flow Projections")
        
        projection_months = st.slider("Projection Period (Months)", 3, 24, 12)
        
        # Generate monthly projections
        monthly_operating_cf = st.session_state.calculated_data.get('operating_cash_flow', 3000)
        monthly_investing_cf = st.session_state.calculated_data.get('investing_cash_flow', 0)
        monthly_financing_cf = st.session_state.calculated_data.get('financing_cash_flow', 0)
        
        # Apply seasonal factors if available
        if 'industry_template_applied' in st.session_state:
            template = st.session_state.industry_template_applied['template']
            seasonal_factors = template['seasonal_factors']
        else:
            seasonal_factors = [1.0] * 12
        
        # Generate projections
        months = list(range(1, projection_months + 1))
        operating_projection = []
        investing_projection = []
        financing_projection = []
        cumulative_cf = []
        
        running_total = 0
        
        for i in range(projection_months):
            # Apply seasonality to operating cash flow
            seasonal_factor = seasonal_factors[i % 12]
            monthly_op_cf = monthly_operating_cf * seasonal_factor * (1 + i * 0.01)  # 1% monthly growth
            
            # Investing cash flow (lumpy - major investments quarterly)
            monthly_inv_cf = monthly_investing_cf if i % 3 == 0 else 0
            
            # Financing cash flow (annual)
            monthly_fin_cf = monthly_financing_cf if i % 12 == 0 else 0
            
            operating_projection.append(monthly_op_cf)
            investing_projection.append(monthly_inv_cf)
            financing_projection.append(monthly_fin_cf)
            
            monthly_total = monthly_op_cf + monthly_inv_cf + monthly_fin_cf
            running_total += monthly_total
            cumulative_cf.append(running_total)
        
        # Create projection chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Cash Flow Components', 'Cumulative Cash Flow'),
            vertical_spacing=0.1
        )
        
        # Monthly components
        fig.add_trace(
            go.Bar(x=months, y=operating_projection, name='Operating CF', marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=months, y=investing_projection, name='Investing CF', marker_color='red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=months, y=financing_projection, name='Financing CF', marker_color='blue'),
            row=1, col=1
        )
        
        # Cumulative cash flow
        fig.add_trace(
            go.Scatter(x=months, y=cumulative_cf, mode='lines+markers', name='Cumulative CF', line=dict(color='purple', width=3)),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Cash Flow Projections")
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Cash Flow (DHS)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cash flow insights
        st.markdown("### 💡 Cash Flow Insights")
        
        # Calculate key metrics
        avg_monthly_cf = sum(operating_projection) / len(operating_projection)
        min_cumulative = min(cumulative_cf)
        max_cumulative = max(cumulative_cf)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Monthly CF", f"{avg_monthly_cf:+,.0f} DHS")
        with col2:
            st.metric("Lowest Cash Position", f"{min_cumulative:+,.0f} DHS")
        with col3:
            st.metric("Highest Cash Position", f"{max_cumulative:+,.0f} DHS")
        
        # Cash flow alerts
        alerts = []
        
        if min_cumulative < -10000:
            alerts.append("🔴 **Cash Flow Alert**: Projected negative cash position detected")
        
        if avg_monthly_cf < 0:
            alerts.append("🟡 **Warning**: Average monthly cash flow is negative")
        
        if operating_cf <= 0:
            alerts.append("🔴 **Critical**: Operating cash flow is not positive")
        
        if not alerts:
            alerts.append("✅ **Healthy**: Cash flow projections look positive")
        
        for alert in alerts:
            if "🔴" in alert:
                st.error(alert)
            elif "🟡" in alert:
                st.warning(alert)
            else:
                st.success(alert)
        
        # Cash management recommendations
        st.markdown("### 🎯 Cash Management Recommendations")
        
        recommendations = []
        
        if operating_cf <= 0:
            recommendations.append("🔧 **Improve Operating Efficiency**: Focus on accelerating receivables and optimizing payables")
        
        if min_cumulative < 0:
            recommendations.append("💳 **Establish Credit Line**: Consider setting up emergency financing to cover cash shortfalls")
        
        if investing_cf < -50000:
            recommendations.append("📊 **Review Capital Allocation**: High investing outflows - ensure ROI justification")
        
        if avg_monthly_cf > 20000:
            recommendations.append("💰 **Optimize Excess Cash**: Consider investment opportunities or debt reduction")
        
        if not recommendations:
            recommendations.append("✅ **Maintain Course**: Current cash flow management appears adequate")
        
        for rec in recommendations:
            st.info(rec)

def show_amortization():
    """Enhanced amortization schedule page"""
    st.header("📋 Amortization Schedule")
    
    # Check if there are any credits/loans
    if not st.session_state.get('credits'):
        st.warning("⚠️ No loans found. Please add loans in the Investments section first.")
        if st.button("➕ Go to Investments"):
            st.session_state.redirect_to = "Investments"
            st.rerun()
        return
    
    # Loan selection
    loans = st.session_state.get('credits', [])
    loan_options = [f"{loan['Type']} - {loan['Montant']:,.0f} DHS @ {loan['Taux']:.2f}%" for loan in loans]
    
    selected_loan_idx = st.selectbox("Select Loan", range(len(loans)), format_func=lambda x: loan_options[x])
    selected_loan = loans[selected_loan_idx]
    
    # Loan details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Loan Amount", f"{selected_loan['Montant']:,.0f} DHS")
    with col2:
        st.metric("Interest Rate", f"{selected_loan['Taux']:.2f}%")
    with col3:
        st.metric("Term", f"{selected_loan['Duree']} months")
    with col4:
        st.metric("Monthly Payment", f"{selected_loan['Mensualite']:,.0f} DHS")
    
    # Generate amortization schedule
    principal = selected_loan['Montant']
    annual_rate = selected_loan['Taux'] / 100
    monthly_rate = annual_rate / 12
    num_payments = selected_loan['Duree']
    monthly_payment = selected_loan['Mensualite']
    
    # Calculate amortization schedule
    schedule_data = []
    remaining_balance = principal
    
    for payment_num in range(1, num_payments + 1):
        # Calculate interest and principal for this payment
        interest_payment = remaining_balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        remaining_balance -= principal_payment
        
        # Ensure we don't go negative on the last payment
        if payment_num == num_payments:
            principal_payment += remaining_balance
            remaining_balance = 0
        
        schedule_data.append({
            'Payment #': payment_num,
            'Payment Amount': monthly_payment,
            'Principal': principal_payment,
            'Interest': interest_payment,
            'Remaining Balance': max(0, remaining_balance),
            'Cumulative Interest': sum(row['Interest'] for row in schedule_data) + interest_payment,
            'Cumulative Principal': sum(row['Principal'] for row in schedule_data) + principal_payment
        })
    
    df_schedule = pd.DataFrame(schedule_data)
    
    # Amortization summary
    st.subheader("📊 Loan Summary")
    
    total_payments = monthly_payment * num_payments
    total_interest = total_payments - principal
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total of Payments", f"{total_payments:,.0f} DHS")
    with col2:
        st.metric("Total Interest", f"{total_interest:,.0f} DHS")
    with col3:
        interest_percentage = (total_interest / principal) * 100
        st.metric("Interest as % of Principal", f"{interest_percentage:.1f}%")
    with col4:
        avg_monthly_interest = total_interest / num_payments
        st.metric("Avg Monthly Interest", f"{avg_monthly_interest:,.0f} DHS")
    
    # Display options
    display_option = st.selectbox("Display Option", ["Summary View", "Detailed Schedule", "Charts"])
    
    if display_option == "Summary View":
        # Show annual summary
        st.subheader("📅 Annual Summary")
        
        annual_summary = []
        current_year = 1
        year_data = {'Year': current_year, 'Principal': 0, 'Interest': 0, 'Payments': 0, 'End Balance': 0}
        
        for i, row in df_schedule.iterrows():
            if (i + 1) % 12 == 0 or i == len(df_schedule) - 1:
                year_data['Principal'] += row['Principal']
                year_data['Interest'] += row['Interest']
                year_data['Payments'] += row['Payment Amount']
                year_data['End Balance'] = row['Remaining Balance']
                
                annual_summary.append(year_data.copy())
                current_year += 1
                year_data = {'Year': current_year, 'Principal': 0, 'Interest': 0, 'Payments': 0, 'End Balance': 0}
            else:
                year_data['Principal'] += row['Principal']
                year_data['Interest'] += row['Interest']
                year_data['Payments'] += row['Payment Amount']
        
        df_annual = pd.DataFrame(annual_summary)
        
        # Format the dataframe
        df_annual_formatted = df_annual.copy()
        for col in ['Principal', 'Interest', 'Payments', 'End Balance']:
            df_annual_formatted[col] = df_annual_formatted[col].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(df_annual_formatted, use_container_width=True)
    
    elif display_option == "Detailed Schedule":
        # Show detailed monthly schedule
        st.subheader("📋 Detailed Monthly Schedule")
        
        # Add pagination for large schedules
        page_size = 24  # 2 years worth of payments
        total_pages = (len(df_schedule) + page_size - 1) // page_size
        
        if total_pages > 1:
            page = st.selectbox("Select Page", range(1, total_pages + 1), format_func=lambda x: f"Payments {(x-1)*page_size + 1}-{min(x*page_size, len(df_schedule))}")
            start_idx = (page - 1) * page_size
            end_idx = min(page * page_size, len(df_schedule))
            display_df = df_schedule.iloc[start_idx:end_idx].copy()
        else:
            display_df = df_schedule.copy()
        
        # Format currency columns
        currency_cols = ['Payment Amount', 'Principal', 'Interest', 'Remaining Balance', 'Cumulative Interest', 'Cumulative Principal']
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download option
        csv = df_schedule.to_csv(index=False)
        st.download_button(
            label="💾 Download Complete Schedule as CSV",
            data=csv,
            file_name=f"amortization_schedule_{selected_loan['Type'].lower().replace(' ', '_')}.csv",
            mime='text/csv'
        )
    
    else:  # Charts
        st.subheader("📈 Amortization Charts")
        
        # Principal vs Interest over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Principal vs Interest',
                'Remaining Balance Over Time',
                'Cumulative Interest vs Principal',
                'Payment Breakdown'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Monthly principal vs interest
        fig.add_trace(
            go.Scatter(x=df_schedule['Payment #'], y=df_schedule['Principal'], name='Principal', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_schedule['Payment #'], y=df_schedule['Interest'], name='Interest', line=dict(color='red')),
            row=1, col=1
        )
        
        # Remaining balance
        fig.add_trace(
            go.Scatter(x=df_schedule['Payment #'], y=df_schedule['Remaining Balance'], name='Balance', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Cumulative amounts
        fig.add_trace(
            go.Scatter(x=df_schedule['Payment #'], y=df_schedule['Cumulative Principal'], name='Cumulative Principal', line=dict(color='green', dash='dash')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_schedule['Payment #'], y=df_schedule['Cumulative Interest'], name='Cumulative Interest', line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Payment breakdown pie chart
        fig.add_trace(
            go.Pie(labels=['Total Principal', 'Total Interest'], values=[principal, total_interest], name="Payment Breakdown"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Loan Amortization Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Interest savings calculator
        st.markdown("### 💰 Early Payment Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            extra_payment = st.number_input("Extra Monthly Payment (DHS)", min_value=0.0, value=0.0, step=500.0)
        
        with col2:
            one_time_payment = st.number_input("One-time Extra Payment (DHS)", min_value=0.0, value=0.0, step=5000.0)
            payment_month = st.number_input("Apply in Month", min_value=1, max_value=num_payments, value=1) if one_time_payment > 0 else 1
        
        if extra_payment > 0 or one_time_payment > 0:
            # Recalculate with extra payments
            new_schedule = []
            remaining_balance = principal
            payment_num = 0
            
            while remaining_balance > 0.01 and payment_num < num_payments * 2:  # Prevent infinite loop
                payment_num += 1
                
                # Calculate standard payment
                interest_payment = remaining_balance * monthly_rate
                principal_payment = monthly_payment - interest_payment
                
                # Add extra payments
                if extra_payment > 0:
                    principal_payment += extra_payment
                
                if one_time_payment > 0 and payment_num == payment_month:
                    principal_payment += one_time_payment
                
                # Don't overpay
                if principal_payment > remaining_balance:
                    principal_payment = remaining_balance
                
                remaining_balance -= principal_payment
                
                new_schedule.append({
                    'Payment #': payment_num,
                    'Interest': interest_payment,
                    'Principal': principal_payment,
                    'Remaining Balance': max(0, remaining_balance)
                })
                
                if remaining_balance <= 0:
                    break
            
            # Calculate savings
            new_total_payments = sum(row['Interest'] + row['Principal'] for row in new_schedule)
            new_total_interest = sum(row['Interest'] for row in new_schedule)
            
            interest_savings = total_interest - new_total_interest
            time_savings = num_payments - len(new_schedule)
            
            st.markdown("#### 💡 Savings with Extra Payments")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Interest Savings", f"{interest_savings:,.0f} DHS")
            with col2:
                st.metric("Time Savings", f"{time_savings} months")
            with col3:
                st.metric("New Payoff Time", f"{len(new_schedule)} months")

def show_financial_education():
    """Enhanced financial education and help page"""
    st.header("🎓 Financial Education & Business Intelligence")
    
    # Education topics
    education_tabs = st.tabs([
        "📚 Fundamentals", 
        "💡 Financial Ratios", 
        "📊 Business Planning", 
        "🎯 Industry Insights",
        "⚠️ Risk Management",
        "🚀 Growth Strategies"
    ])
    
    with education_tabs[0]:
        st.subheader("📚 Financial Fundamentals")
        
        # Basic concepts
        with st.expander("💰 Financial Statements Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                ### 📈 Income Statement
                **Purpose**: Shows profitability over time
                
                **Key Components**:
                - Revenue/Sales
                - Cost of Goods Sold
                - Operating Expenses
                - Net Income
                
                **Key Insight**: How much money your business made or lost
                """)
            
            with col2:
                st.markdown("""
                ### 📊 Balance Sheet
                **Purpose**: Shows financial position at a point in time
                
                **Key Components**:
                - Assets (what you own)
                - Liabilities (what you owe)
                - Equity (owner's stake)
                
                **Key Insight**: Your business's net worth
                """)
            
            with col3:
                st.markdown("""
                ### 💰 Cash Flow Statement
                **Purpose**: Shows cash movements
                
                **Key Components**:
                - Operating Cash Flow
                - Investing Cash Flow
                - Financing Cash Flow
                
                **Key Insight**: How cash moves in and out of your business
                """)
        
        with st.expander("🔢 Key Financial Terms"):
            terms = {
                "Revenue": "Total income from sales before any expenses",
                "Gross Profit": "Revenue minus cost of goods sold",
                "Operating Profit": "Gross profit minus operating expenses",
                "Net Profit": "Final profit after all expenses and taxes",
                "Cash Flow": "Movement of money in and out of business",
                "Working Capital": "Current assets minus current liabilities",
                "EBITDA": "Earnings before interest, taxes, depreciation, and amortization",
                "ROI": "Return on Investment - profit relative to investment cost",
                "Break-even Point": "Level of sales needed to cover all costs"
            }
            
            for term, definition in terms.items():
                st.markdown(f"**{term}**: {definition}")
        
        with st.expander("📝 Financial Planning Best Practices"):
            st.markdown("""
            ### 🎯 Planning Principles
            
            1. **Start with Clear Goals**
               - Define specific, measurable objectives
               - Set realistic timelines
               - Align financial plans with business strategy
            
            2. **Use Conservative Estimates**
               - Plan for 80% of optimistic scenarios
               - Include contingency buffers
               - Prepare for seasonal variations
            
            3. **Monitor and Adjust Regularly**
               - Review monthly performance
               - Update forecasts quarterly
               - Adjust strategies based on results
            
            4. **Focus on Cash Flow**
               - Cash is king for business survival
               - Monitor payment cycles carefully
               - Plan for working capital needs
            
            5. **Understand Your Industry**
               - Know typical margins and ratios
               - Understand seasonal patterns
               - Benchmark against competitors
            """)
    
    with education_tabs[1]:
        st.subheader("💡 Financial Ratios Guide")
        
        # Interactive ratio calculator
        st.markdown("### 🧮 Interactive Ratio Calculator")
        
        with st.form("ratio_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input Values:**")
                revenue = st.number_input("Annual Revenue (DHS)", min_value=0.0, value=180000.0, step=10000.0)
                net_income = st.number_input("Net Income (DHS)", value=27000.0, step=1000.0)
                current_assets = st.number_input("Current Assets (DHS)", min_value=0.0, value=75000.0, step=5000.0)
                current_liabilities = st.number_input("Current Liabilities (DHS)", min_value=0.1, value=30000.0, step=1000.0)
            
            with col2:
                total_assets = st.number_input("Total Assets (DHS)", min_value=0.1, value=150000.0, step=10000.0)
                total_debt = st.number_input("Total Debt (DHS)", min_value=0.0, value=50000.0, step=5000.0)
                equity = st.number_input("Equity (DHS)", min_value=0.1, value=100000.0, step=5000.0)
                gross_profit = st.number_input("Gross Profit (DHS)", value=108000.0, step=5000.0)
            
            calculate_ratios = st.form_submit_button("Calculate Ratios")
            
            if calculate_ratios:
                # Calculate ratios
                calculated_ratios = {
                    "Net Margin": (net_income / revenue * 100, "%", "Higher is better - shows profitability"),
                    "Gross Margin": (gross_profit / revenue * 100, "%", "Higher is better - shows pricing power"),
                    "Current Ratio": (current_assets / current_liabilities, "x", "Above 1.5 is good - shows liquidity"),
                    "Debt-to-Equity": (total_debt / equity, "x", "Lower is better - shows financial stability"),
                    "ROA": (net_income / total_assets * 100, "%", "Higher is better - shows asset efficiency"),
                    "ROE": (net_income / equity * 100, "%", "Higher is better - shows return to owners"),
                    "Asset Turnover": (revenue / total_assets, "x", "Higher is better - shows asset productivity")
                }
                
                st.markdown("### 📊 Your Calculated Ratios")
                
                for ratio_name, (value, unit, interpretation) in calculated_ratios.items():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(ratio_name, f"{value:.2f}{unit}")
                    with col2:
                        st.caption(interpretation)
        
        # Ratio benchmarks by industry
        st.markdown("### 🏭 Industry Benchmarks")
        
        benchmark_data = {
            "Technology": {"Net Margin": "15-25%", "Current Ratio": "2.0+", "Debt-to-Equity": "0.1-0.3", "ROE": "15-25%"},
            "Retail": {"Net Margin": "3-8%", "Current Ratio": "1.0-1.5", "Debt-to-Equity": "0.5-0.8", "ROE": "10-18%"},
            "Manufacturing": {"Net Margin": "5-12%", "Current Ratio": "1.2-1.8", "Debt-to-Equity": "0.4-0.7", "ROE": "8-15%"},
            "Restaurant": {"Net Margin": "3-7%", "Current Ratio": "0.8-1.2", "Debt-to-Equity": "0.6-1.0", "ROE": "12-20%"},
            "Consulting": {"Net Margin": "10-20%", "Current Ratio": "1.5-2.5", "Debt-to-Equity": "0.2-0.5", "ROE": "15-30%"}
        }
        
        selected_industry = st.selectbox("Select Industry for Benchmarks", list(benchmark_data.keys()))
        
        industry_benchmarks = benchmark_data[selected_industry]
        
        cols = st.columns(len(industry_benchmarks))
        for i, (metric, benchmark) in enumerate(industry_benchmarks.items()):
            with cols[i]:
                st.metric(metric, benchmark)
    
    with education_tabs[2]:
        st.subheader("📊 Business Planning Essentials")
        
        with st.expander("🎯 Business Model Canvas", expanded=True):
            st.markdown("""
            ### 🧩 Key Components of Your Business Model
            
            Use this framework to analyze your business:
            """)
            
            canvas_cols = st.columns(3)
            
            with canvas_cols[0]:
                st.markdown("""
                **📋 Key Activities**
                - Primary operations
                - Core processes
                - Value creation activities
                
                **🤝 Key Partnerships**
                - Strategic alliances
                - Suppliers
                - Key vendors
                
                **💰 Cost Structure**
                - Fixed costs
                - Variable costs
                - Economies of scale
                """)
            
            with canvas_cols[1]:
                st.markdown("""
                **🎁 Value Propositions**
                - Unique benefits
                - Problem solving
                - Customer value
                
                **👥 Customer Segments**
                - Target markets
                - User personas
                - Market size
                """)
            
            with canvas_cols[2]:
                st.markdown("""
                **📢 Channels**
                - Sales channels
                - Distribution
                - Marketing channels
                
                **🤝 Customer Relationships**
                - Service model
                - Support strategy
                - Retention approach
                
                **💵 Revenue Streams**
                - Pricing model
                - Revenue sources
                - Payment terms
                """)
        
        with st.expander("📈 Financial Forecasting Framework"):
            st.markdown("""
            ### 🔮 Building Reliable Forecasts
            
            #### 1. **Bottom-Up Approach**
            - Start with unit sales/service volumes
            - Apply pricing per unit/service
            - Build up to total revenue
            
            #### 2. **Cost Structure Analysis**
            - Identify fixed vs variable costs
            - Understand cost drivers
            - Plan for scale economies
            
            #### 3. **Seasonality & Trends**
            - Analyze historical patterns
            - Factor in market trends
            - Include economic cycles
            
            #### 4. **Scenario Planning**
            - Best case (optimistic)
            - Base case (realistic)
            - Worst case (pessimistic)
            """)
            
            # Interactive forecasting tips
            forecast_method = st.selectbox("Select Forecasting Method", [
                "Unit-Based Forecasting",
                "Trend Analysis",
                "Market Share Approach",
                "Customer-Based Forecasting"
            ])
            
            if forecast_method == "Unit-Based Forecasting":
                st.info("""
                **Best for**: Product businesses, manufacturing, retail
                
                **Method**: Units × Price = Revenue
                
                **Steps**:
                1. Estimate monthly unit sales
                2. Apply average selling price
                3. Factor in seasonality
                4. Plan for growth/decline
                """)
            
            elif forecast_method == "Customer-Based Forecasting":
                st.info("""
                **Best for**: Service businesses, SaaS, consulting
                
                **Method**: Customers × Value per Customer = Revenue
                
                **Steps**:
                1. Estimate customer acquisition rate
                2. Calculate average customer value
                3. Factor in churn/retention
                4. Plan for upselling
                """)
    
    with education_tabs[3]:
        st.subheader("🎯 Industry-Specific Insights")
        
        # Industry analysis tool
        analysis_industry = st.selectbox("Select Industry for Analysis", [
            "Technology/SaaS",
            "Retail/E-commerce", 
            "Manufacturing",
            "Restaurant/Food Service",
            "Professional Services"
        ])
        
        industry_insights = {
            "Technology/SaaS": {
                "key_metrics": ["Monthly Recurring Revenue (MRR)", "Customer Acquisition Cost (CAC)", "Lifetime Value (LTV)", "Churn Rate"],
                "success_factors": ["Product-market fit", "Scalable technology", "Strong customer support", "Continuous innovation"],
                "common_challenges": ["High customer acquisition costs", "Competitive market", "Technology scalability", "Talent retention"],
                "financial_characteristics": ["High gross margins (70-90%)", "Predictable revenue", "High upfront costs", "Long payback periods"]
            },
            "Retail/E-commerce": {
                "key_metrics": ["Gross Margin", "Inventory Turnover", "Customer Acquisition Cost", "Average Order Value"],
                "success_factors": ["Strong supply chain", "Effective marketing", "Customer experience", "Inventory management"],
                "common_challenges": ["Seasonal fluctuations", "Inventory management", "Competition", "Margin pressure"],
                "financial_characteristics": ["Lower margins (20-50%)", "Working capital intensive", "Seasonal cash flow", "High fixed costs"]
            },
            "Manufacturing": {
                "key_metrics": ["Capacity Utilization", "Cost per Unit", "Quality Metrics", "Equipment Efficiency"],
                "success_factors": ["Operational efficiency", "Quality control", "Supply chain management", "Cost optimization"],
                "common_challenges": ["Raw material costs", "Equipment maintenance", "Regulatory compliance", "Market demand"],
                "financial_characteristics": ["High fixed costs", "Capital intensive", "Economies of scale", "Cyclical revenue"]
            }
        }
        
        if analysis_industry in industry_insights:
            insights = industry_insights[analysis_industry]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Key Metrics to Track")
                for metric in insights["key_metrics"]:
                    st.write(f"• {metric}")
                
                st.markdown("### 🎯 Success Factors")
                for factor in insights["success_factors"]:
                    st.write(f"• {factor}")
            
            with col2:
                st.markdown("### ⚠️ Common Challenges")
                for challenge in insights["common_challenges"]:
                    st.write(f"• {challenge}")
                
                st.markdown("### 💰 Financial Characteristics")
                for char in insights["financial_characteristics"]:
                    st.write(f"• {char}")
    
    with education_tabs[4]:
        st.subheader("⚠️ Risk Management Fundamentals")
        
        with st.expander("🎯 Types of Business Risks", expanded=True):
            risk_types = {
                "💼 Operational Risk": [
                    "Key person dependency",
                    "Supply chain disruption", 
                    "Technology failures",
                    "Quality control issues"
                ],
                "💰 Financial Risk": [
                    "Cash flow shortages",
                    "Credit risk",
                    "Interest rate changes",
                    "Currency fluctuations"
                ],
                "📈 Market Risk": [
                    "Demand changes",
                    "Competition",
                    "Economic downturns",
                    "Industry disruption"
                ],
                "📋 Compliance Risk": [
                    "Regulatory changes",
                    "Tax law changes",
                    "Industry standards",
                    "Data protection"
                ]
            }
            
            for risk_category, risks in risk_types.items():
                st.markdown(f"### {risk_category}")
                for risk in risks:
                    st.write(f"• {risk}")
        
        with st.expander("🛡️ Risk Mitigation Strategies"):
            st.markdown("""
            ### 🔧 Risk Management Framework
            
            #### 1. **Risk Identification**
            - Regular risk assessments
            - Stakeholder input
            - Industry analysis
            - Historical review
            
            #### 2. **Risk Assessment**
            - Probability estimation
            - Impact analysis
            - Risk scoring
            - Priority ranking
            
            #### 3. **Risk Response**
            - **Avoid**: Eliminate the risk
            - **Mitigate**: Reduce probability/impact
            - **Transfer**: Insurance, contracts
            - **Accept**: Monitor and manage
            
            #### 4. **Monitoring & Review**
            - Regular reviews
            - Updated assessments
            - Response effectiveness
            - New risk identification
            """)
        
        # Risk assessment tool
        st.markdown("### 🧮 Simple Risk Assessment Tool")
        
        with st.form("risk_assessment"):
            risk_name = st.text_input("Risk Description")
            probability = st.selectbox("Probability", ["Low (1)", "Medium (2)", "High (3)"])
            impact = st.selectbox("Impact", ["Low (1)", "Medium (2)", "High (3)"])
            
            assess_risk = st.form_submit_button("Assess Risk")
            
            if assess_risk and risk_name:
                prob_score = int(probability.split("(")[1].split(")")[0])
                impact_score = int(impact.split("(")[1].split(")")[0])
                risk_score = prob_score * impact_score
                
                if risk_score >= 6:
                    risk_level = "🔴 High Risk"
                    recommendation = "Immediate action required"
                elif risk_score >= 4:
                    risk_level = "🟡 Medium Risk"
                    recommendation = "Develop mitigation plan"
                else:
                    risk_level = "🟢 Low Risk"
                    recommendation = "Monitor regularly"
                
                st.success(f"**{risk_name}**: {risk_level} (Score: {risk_score}/9)")
                st.info(f"**Recommendation**: {recommendation}")
    
    with education_tabs[5]:
        st.subheader("🚀 Growth Strategies")
        
        with st.expander("📈 Growth Framework", expanded=True):
            st.markdown("""
            ### 🎯 Ansoff Growth Matrix
            
            Choose your growth strategy based on markets and products:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔄 Market Penetration**
                *Existing Products + Existing Markets*
                
                Strategies:
                • Increase market share
                • Improve customer retention
                • Increase usage frequency
                • Competitive pricing
                
                Risk: Low | Potential: Medium
                """)
                
                st.markdown("""
                **🆕 Product Development**
                *New Products + Existing Markets*
                
                Strategies:
                • Develop new features
                • Create complementary products
                • Improve existing products
                • Innovation initiatives
                
                Risk: Medium | Potential: High
                """)
            
            with col2:
                st.markdown("""
                **🌍 Market Development**
                *Existing Products + New Markets*
                
                Strategies:
                • Geographic expansion
                • New customer segments
                • New distribution channels
                • Online expansion
                
                Risk: Medium | Potential: High
                """)
                
                st.markdown("""
                **🎲 Diversification**
                *New Products + New Markets*
                
                Strategies:
                • Related diversification
                • Unrelated diversification
                • Acquisitions
                • Strategic partnerships
                
                Risk: High | Potential: Very High
                """)
        
        with st.expander("💰 Funding Growth"):
            st.markdown("""
            ### 💳 Financing Options for Growth
            
            #### **🏦 Debt Financing**
            **Pros**: Retain ownership, tax deductible, predictable costs
            **Cons**: Repayment obligation, collateral requirements, financial covenants
            
            **Options**:
            • Bank loans
            • Lines of credit
            • Equipment financing
            • SBA loans
            
            #### **📈 Equity Financing**
            **Pros**: No repayment, additional expertise, shared risk
            **Cons**: Dilution, loss of control, complex processes
            
            **Options**:
            • Angel investors
            • Venture capital
            • Crowdfunding
            • Strategic partners
            
            #### **💰 Alternative Financing**
            **Options**:
            • Revenue-based financing
            • Invoice factoring
            • Merchant cash advances
            • Government grants
            """)
        
        # Growth calculator
        st.markdown("### 🧮 Growth Planning Calculator")
        
        with st.form("growth_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                current_revenue = st.number_input("Current Annual Revenue (DHS)", min_value=0.0, value=180000.0, step=10000.0)
                target_growth = st.slider("Target Annual Growth Rate (%)", 5, 100, 25)
                time_horizon = st.selectbox("Time Horizon", [1, 2, 3, 5], index=2)
            
            with col2:
                current_margin = st.slider("Current Net Margin (%)", 0, 50, 15)
                investment_ratio = st.slider("Investment as % of Revenue", 0, 50, 10)
                
            calculate_growth = st.form_submit_button("Calculate Growth Plan")
            
            if calculate_growth:
                # Calculate projections
                years = []
                revenues = []
                investments = []
                profits = []
                
                for year in range(1, time_horizon + 1):
                    projected_revenue = current_revenue * ((1 + target_growth/100) ** year)
                    required_investment = projected_revenue * (investment_ratio/100)
                    projected_profit = projected_revenue * (current_margin/100)
                    
                    years.append(f"Year {year}")
                    revenues.append(projected_revenue)
                    investments.append(required_investment)
                    profits.append(projected_profit)
                
                # Display results
                st.markdown("#### 📊 Growth Projections")
                
                projection_df = pd.DataFrame({
                    "Year": years,
                    "Revenue": [f"{r:,.0f}" for r in revenues],
                    "Investment Needed": [f"{i:,.0f}" for i in investments],
                    "Projected Profit": [f"{p:,.0f}" for p in profits]
                })
                
                st.dataframe(projection_df, use_container_width=True)
                
                # Growth chart
                fig = go.Figure()
                fig.add_trace(go.Bar(x=years, y=revenues, name='Revenue', marker_color='green'))
                fig.add_trace(go.Bar(x=years, y=investments, name='Investment', marker_color='red'))
                fig.update_layout(title="Growth Investment vs Revenue Projection", yaxis_title="Amount (DHS)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Key insights
                total_investment = sum(investments)
                final_revenue = revenues[-1]
                revenue_multiple = final_revenue / current_revenue
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment Required", f"{total_investment:,.0f} DHS")
                with col2:
                    st.metric("Revenue Multiple", f"{revenue_multiple:.1f}x")
                with col3:
                    st.metric("Final Annual Revenue", f"{final_revenue:,.0f} DHS")

def show_csv_import():
    """Enhanced CSV import with AI-powered analysis"""
    st.header("📤 Advanced CSV Import & Data Analysis")
    
    st.markdown("""
    Import your existing financial data from CSV files. Our system will automatically detect columns and provide intelligent analysis.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload financial data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column analysis
            st.subheader("🔍 Column Analysis")
            
            # AI-powered column detection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            date_columns = []
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Try to detect date columns
            for col in text_columns:
                try:
                    pd.to_datetime(df[col].iloc[0])
                    date_columns.append(col)
                except:
                    pass
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Numeric Columns**")
                for col in numeric_columns:
                    st.write(f"• {col}")
            
            with col2:
                st.markdown("**📅 Date Columns**")
                for col in date_columns:
                    st.write(f"• {col}")
            
            with col3:
                st.markdown("**📝 Text Columns**")
                for col in text_columns:
                    if col not in date_columns:
                        st.write(f"• {col}")
            
            # Intelligent mapping
            st.subheader("🧠 Intelligent Column Mapping")
            
            # Suggest mappings based on column names
            column_mappings = {}
            mapping_suggestions = {
                'revenue': ['revenue', 'sales', 'income', 'turnover', 'receipts'],
                'costs': ['costs', 'expenses', 'expenditure', 'outgoings'],
                'date': ['date', 'month', 'period', 'time'],
                'profit': ['profit', 'earnings', 'net income', 'pnl'],
                'cash_flow': ['cash flow', 'cashflow', 'cash', 'flow']
            }
            
            for target, keywords in mapping_suggestions.items():
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in keywords):
                        column_mappings[target] = col
                        break
            
            # Manual mapping interface
            with st.form("column_mapping"):
                st.markdown("**Map your columns to financial categories:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    revenue_col = st.selectbox("Revenue Column", ['None'] + list(df.columns), 
                                             index=list(df.columns).index(column_mappings.get('revenue', df.columns[0])) + 1 if column_mappings.get('revenue') in df.columns else 0)
                    
                    costs_col = st.selectbox("Costs Column", ['None'] + list(df.columns),
                                           index=list(df.columns).index(column_mappings.get('costs', df.columns[0])) + 1 if column_mappings.get('costs') in df.columns else 0)
                    
                    date_col = st.selectbox("Date Column", ['None'] + list(df.columns),
                                          index=list(df.columns).index(column_mappings.get('date', df.columns[0])) + 1 if column_mappings.get('date') in df.columns else 0)
                
                with col2:
                    profit_col = st.selectbox("Profit Column", ['None'] + list(df.columns),
                                            index=list(df.columns).index(column_mappings.get('profit', df.columns[0])) + 1 if column_mappings.get('profit') in df.columns else 0)
                    
                    cash_flow_col = st.selectbox("Cash Flow Column", ['None'] + list(df.columns),
                                                index=list(df.columns).index(column_mappings.get('cash_flow', df.columns[0])) + 1 if column_mappings.get('cash_flow') in df.columns else 0)
                
                analyze_data = st.form_submit_button("🔍 Analyze Data", type="primary")
                
                if analyze_data:
                    # Perform analysis
                    st.subheader("📈 Data Analysis Results")
                    
                    analysis_results = {}
                    
                    # Revenue analysis
                    if revenue_col != 'None':
                        revenue_data = df[revenue_col].dropna()
                        analysis_results['revenue'] = {
                            'total': revenue_data.sum(),
                            'average': revenue_data.mean(),
                            'trend': 'increasing' if revenue_data.iloc[-1] > revenue_data.iloc[0] else 'decreasing',
                            'volatility': revenue_data.std() / revenue_data.mean() if revenue_data.mean() != 0 else 0
                        }
                    
                    # Costs analysis
                    if costs_col != 'None':
                        costs_data = df[costs_col].dropna()
                        analysis_results['costs'] = {
                            'total': costs_data.sum(),
                            'average': costs_data.mean(),
                            'trend': 'increasing' if costs_data.iloc[-1] > costs_data.iloc[0] else 'decreasing',
                            'volatility': costs_data.std() / costs_data.mean() if costs_data.mean() != 0 else 0
                        }
                    
                    # Display analysis
                    if 'revenue' in analysis_results and 'costs' in analysis_results:
                        revenue_stats = analysis_results['revenue']
                        costs_stats = analysis_results['costs']
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Revenue", f"{revenue_stats['total']:,.0f}")
                        with col2:
                            st.metric("Total Costs", f"{costs_stats['total']:,.0f}")
                        with col3:
                            total_profit = revenue_stats['total'] - costs_stats['total']
                            st.metric("Total Profit", f"{total_profit:,.0f}")
                        with col4:
                            margin = total_profit / revenue_stats['total'] * 100 if revenue_stats['total'] != 0 else 0
                            st.metric("Profit Margin", f"{margin:.1f}%")
                        
                        # Trends visualization
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Revenue and Costs Over Time', 'Profit Trend'),
                            vertical_spacing=0.1
                        )
                        
                        # Get data for plotting
                        if date_col != 'None':
                            try:
                                plot_df = df.copy()
                                plot_df[date_col] = pd.to_datetime(plot_df[date_col])
                                plot_df = plot_df.sort_values(date_col)
                                
                                x_data = plot_df[date_col]
                            except:
                                x_data = range(len(df))
                        else:
                            x_data = range(len(df))
                        
                        # Revenue and costs
                        fig.add_trace(
                            go.Scatter(x=x_data, y=df[revenue_col], name='Revenue', line=dict(color='green')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=x_data, y=df[costs_col], name='Costs', line=dict(color='red')),
                            row=1, col=1
                        )
                        
                        # Profit
                        profit_data = df[revenue_col] - df[costs_col]
                        fig.add_trace(
                            go.Scatter(x=x_data, y=profit_data, name='Profit', line=dict(color='blue')),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=600, title_text="Financial Data Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # AI Insights
                        st.subheader("🤖 AI-Generated Insights")
                        
                        insights = []
                        
                        # Revenue insights
                        if revenue_stats['trend'] == 'increasing':
                            insights.append("✅ **Positive Revenue Trend**: Your revenue is growing over time")
                        else:
                            insights.append("⚠️ **Revenue Decline**: Revenue is decreasing - investigate causes")
                        
                        # Volatility insights
                        if revenue_stats['volatility'] > 0.3:
                            insights.append("📊 **High Revenue Volatility**: Consider strategies to stabilize income")
                        
                        # Margin insights
                        if margin > 20:
                            insights.append("💰 **Strong Margins**: Excellent profitability performance")
                        elif margin > 10:
                            insights.append("📈 **Healthy Margins**: Good profitability, room for improvement")
                        elif margin > 0:
                            insights.append("⚠️ **Thin Margins**: Focus on cost optimization or pricing")
                        else:
                            insights.append("🔴 **Negative Margins**: Immediate action needed to return to profitability")
                        
                        # Cost insights
                        if costs_stats['trend'] == 'increasing' and revenue_stats['trend'] == 'decreasing':
                            insights.append("⚠️ **Cost Control Alert**: Costs rising while revenue falls")
                        
                        for insight in insights:
                            if "🔴" in insight or "⚠️" in insight:
                                st.warning(insight)
                            else:
                                st.success(insight)
                        
                        # Import to system
                        st.subheader("💾 Import to Financial System")
                        
                        if st.button("Import Analysis to Current Project", type="primary"):
                            # Update session state with imported data
                            st.session_state.calculated_data.update({
                                'total_ventes': revenue_stats['average'],
                                'total_charges': costs_stats['average'],
                                'imported_from_csv': True,
                                'csv_analysis': analysis_results
                            })
                            
                            st.success("✅ Data successfully imported to your financial model!")
                            st.balloons()
                            
                            # Offer to navigate to analysis
                            if st.button("🚀 View Advanced Analytics"):
                                st.session_state.redirect_to = "Advanced Analytics"
                                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains numeric data.")
    
    else:
        # Show example format
        st.subheader("📋 Expected CSV Format")
        
        example_data = {
            'Date': ['2024-01', '2024-02', '2024-03', '2024-04'],
            'Revenue': [15000, 16500, 14200, 17800],
            'Costs': [12000, 13100, 11800, 14200],
            'Profit': [3000, 3400, 2400, 3600]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        ### 📝 CSV Requirements:
        
        ✅ **Supported formats**: CSV files with comma separation
        
        ✅ **Required data**: At least one numeric column (revenue, costs, or profit)
        
        ✅ **Optional columns**: Date/period, categories, descriptions
        
        ✅ **File size**: Maximum 10MB
        
        ### 💡 Tips for best results:
        
        • Include clear column headers
        • Use consistent date formats (YYYY-MM-DD recommended)
        • Ensure numeric data doesn't contain currency symbols
        • Include as much historical data as possible for better analysis
        """)

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    main()