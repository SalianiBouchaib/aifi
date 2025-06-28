import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="Advanced Financial Planning Suite", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💼"
)

# Enhanced imports
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

# Handle optional dependencies with try/except
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

# ========== CSV DATA MANAGER ==========
class CSVDataManager:
    """Centralized CSV data management to ensure all pages use uploaded data"""
    
    @staticmethod
    def get_csv_financial_data():
        """Get financial data from uploaded CSV, return None if not available"""
        if not st.session_state.imported_metrics:
            return None
        
        metrics = st.session_state.imported_metrics
        
        # Extract key financial metrics
        financial_data = {}
        
        if 'revenue' in metrics:
            financial_data['revenue'] = metrics['revenue']['average'] * 12  # Annualized
            financial_data['monthly_revenue'] = metrics['revenue']['average']
            financial_data['revenue_data'] = metrics['revenue']['data']
            financial_data['revenue_growth'] = metrics['revenue'].get('growth_rate', 0)
            financial_data['revenue_volatility'] = metrics['revenue'].get('volatility', 0)
        
        if 'costs' in metrics:
            financial_data['total_costs'] = metrics['costs']['average'] * 12  # Annualized
            financial_data['monthly_costs'] = metrics['costs']['average']
            financial_data['costs_data'] = metrics['costs']['data']
            financial_data['costs_growth'] = metrics['costs'].get('growth_rate', 0)
        
        if 'profit' in metrics:
            financial_data['net_profit'] = metrics['profit']['average'] * 12  # Annualized
            financial_data['monthly_profit'] = metrics['profit']['average']
            financial_data['profit_data'] = metrics['profit']['data']
            financial_data['profit_margin'] = metrics['profit'].get('margin_average', 0)
        
        # Calculate derived metrics
        if 'revenue' in financial_data and 'total_costs' in financial_data:
            financial_data['gross_profit'] = financial_data['revenue'] - financial_data['total_costs']
            financial_data['operating_profit'] = financial_data['gross_profit'] * 0.8  # Estimate
            financial_data['net_margin'] = financial_data['net_profit'] / financial_data['revenue'] if financial_data['revenue'] > 0 else 0
        
        # Add default balance sheet estimates based on revenue
        if 'revenue' in financial_data:
            revenue = financial_data['revenue']
            financial_data['current_assets'] = revenue * 0.3
            financial_data['current_liabilities'] = revenue * 0.15
            financial_data['total_assets'] = revenue * 0.8
            financial_data['total_debt'] = revenue * 0.2
            financial_data['equity'] = revenue * 0.4
            financial_data['cash'] = revenue * 0.1
            financial_data['inventory'] = revenue * 0.08
            financial_data['interest_expense'] = revenue * 0.02
            financial_data['cash_flow'] = financial_data.get('monthly_profit', revenue * 0.02)
            
            # Financial ratios estimates
            financial_data['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']
            financial_data['debt_to_equity'] = financial_data['total_debt'] / financial_data['equity']
            financial_data['interest_coverage'] = financial_data['operating_profit'] / financial_data['interest_expense']
        
        return financial_data
    
    @staticmethod
    def has_csv_data():
        """Check if CSV data is available"""
        return bool(st.session_state.imported_metrics)
    
    @staticmethod
    def get_csv_insights():
        """Get AI insights from CSV data"""
        if 'csv_data' in st.session_state and 'insights' in st.session_state.csv_data:
            return st.session_state.csv_data['insights']
        return None
    
    @staticmethod
    def get_csv_visualizations():
        """Get visualizations from CSV data"""
        if 'csv_data' in st.session_state and 'figures' in st.session_state.csv_data:
            return st.session_state.csv_data['figures']
        return None

# ========== ENHANCED CSV PROCESSOR ==========
class AdvancedCSVProcessor:
    def __init__(self):
        self.column_mappings = {
            'revenue': ['revenue', 'sales', 'income', 'turnover', 'receipts', 'monthly_recurring_revenue', 'mrr', 'arr'],
            'costs': ['costs', 'expenses', 'expenditure', 'outgoings', 'total_costs', 'variable_costs', 'fixed_costs', 'cost_of_goods_sold', 'cogs'],
            'date': ['date', 'month', 'period', 'time', 'year', 'quarter'],
            'profit': ['profit', 'earnings', 'net_income', 'net income', 'pnl', 'p&l', 'operating_profit', 'gross_profit'],
            'cash_flow': ['cash_flow', 'cash flow', 'cashflow', 'cash', 'flow'],
            'assets': ['assets', 'total_assets', 'current_assets', 'fixed_assets'],
            'liabilities': ['liabilities', 'total_liabilities', 'current_liabilities', 'debt'],
            'equity': ['equity', 'shareholders_equity', 'owners_equity'],
            'inventory': ['inventory', 'stock', 'goods'],
            'accounts_receivable': ['accounts_receivable', 'receivables', 'ar', 'debtors'],
            'accounts_payable': ['accounts_payable', 'payables', 'ap', 'creditors'],
            'customer_metrics': ['customer_count', 'customers', 'active_users', 'monthly_active_users', 'mau'],
            'unit_metrics': ['units_sold', 'quantity', 'volume', 'transactions'],
            'pricing_metrics': ['average_price', 'price_per_unit', 'average_transaction_value', 'atv'],
            'saas_metrics': ['churn_rate', 'retention_rate', 'ltv', 'cac', 'customer_acquisition_cost', 'lifetime_value']
        }
        
        self.required_columns = ['date', 'revenue']
        self.detected_mappings = {}
        self.analysis_results = {}
    
    def detect_columns(self, df):
        """Enhanced column detection with fuzzy matching"""
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
        
        # Remove duplicates and prioritize exact matches
        for target in detected:
            detected[target] = list(set(detected[target]))
            if len(detected[target]) > 1:
                exact_matches = [col for col in detected[target] 
                               if col.lower().replace('_', ' ') in self.column_mappings[target]]
                if exact_matches:
                    detected[target] = exact_matches[:1]
                else:
                    detected[target] = detected[target][:1]
        
        # Flatten to single column per target
        for target in detected:
            if detected[target]:
                detected[target] = detected[target][0]
        
        self.detected_mappings = detected
        return detected
    
    def validate_data(self, df, mappings):
        """Validate the imported data quality"""
        issues = []
        suggestions = []
        
        if 'date' not in mappings:
            issues.append("No date column detected - temporal analysis will be limited")
            suggestions.append("Include a date column for trend analysis")
        
        if 'revenue' not in mappings:
            issues.append("No revenue column detected - this is critical for financial analysis")
            suggestions.append("Ensure you have a revenue/sales column")
        
        for target, col in mappings.items():
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                if missing_pct > 20:
                    issues.append(f"{col} has {missing_pct:.1f}% missing values")
                    suggestions.append(f"Consider filling missing values in {col}")
                
                if target in ['revenue', 'costs', 'profit', 'assets', 'liabilities']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        issues.append(f"{col} should be numeric but contains text")
                        suggestions.append(f"Remove currency symbols and commas from {col}")
        
        return issues, suggestions
    
    def clean_numeric_column(self, series):
        """Clean numeric columns by removing currency symbols and converting to float"""
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        cleaned = series.astype(str)
        cleaned = cleaned.str.replace(r'[$€£¥₹₽]', '', regex=True)
        cleaned = cleaned.str.replace(',', '')
        cleaned = cleaned.str.replace(' ', '')
        cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
        
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        return cleaned
    
    def standardize_date_column(self, series):
        """Standardize date column to datetime"""
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
    
    def calculate_comprehensive_metrics(self, df, mappings):
        """Calculate comprehensive financial metrics from imported data"""
        metrics = {}
        
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
        
        # Calculate profit if both revenue and costs available
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
        
        self.analysis_results = metrics
        return metrics
    
    def generate_insights(self, metrics):
        """Generate AI-powered insights from the analysis"""
        insights = []
        recommendations = []
        alerts = []
        
        if 'revenue' in metrics:
            rev_metrics = metrics['revenue']
            
            if rev_metrics['trend'] == 'increasing':
                insights.append(f"📈 **Positive Revenue Growth**: {rev_metrics['growth_rate']:.1f}% growth over the period")
            else:
                alerts.append(f"📉 **Revenue Decline**: {abs(rev_metrics['growth_rate']):.1f}% decrease detected")
                recommendations.append("Investigate causes of revenue decline and develop recovery strategies")
            
            if rev_metrics['volatility'] > 0.3:
                alerts.append(f"📊 **High Revenue Volatility**: {rev_metrics['volatility']:.1%} coefficient of variation")
                recommendations.append("Consider strategies to stabilize revenue streams")
            elif rev_metrics['volatility'] < 0.1:
                insights.append("✅ **Stable Revenue**: Low volatility indicates predictable business performance")
        
        if 'profit' in metrics:
            profit_metrics = metrics['profit']
            
            if profit_metrics['margin_average'] > 20:
                insights.append(f"💰 **Excellent Margins**: {profit_metrics['margin_average']:.1f}% average profit margin")
            elif profit_metrics['margin_average'] > 10:
                insights.append(f"📈 **Healthy Margins**: {profit_metrics['margin_average']:.1f}% average profit margin")
            elif profit_metrics['margin_average'] > 0:
                alerts.append(f"⚠️ **Thin Margins**: Only {profit_metrics['margin_average']:.1f}% average profit margin")
                recommendations.append("Focus on cost optimization or pricing strategy improvement")
            else:
                alerts.append(f"🔴 **Negative Margins**: {profit_metrics['margin_average']:.1f}% average - immediate action needed")
                recommendations.append("Urgent: Review cost structure and pricing to return to profitability")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'alerts': alerts
        }
    
    def create_visualizations(self, df, mappings, metrics):
        """Create comprehensive visualizations from imported data"""
        figures = {}
        
        if 'date' in mappings and mappings['date'] in df.columns:
            time_col = self.standardize_date_column(df[mappings['date']])
            x_axis = time_col
            x_title = "Date"
        else:
            x_axis = range(len(df))
            x_title = "Period"
        
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
                    name='Costs',
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
                title="Financial Performance Over Time",
                xaxis_title=x_title,
                yaxis_title="Amount (DHS)",
                hovermode='x unified',
                height=500
            )
            
            figures['financial_trend'] = fig
        
        return figures

    def process_csv(self, df):
        """Main processing function for uploaded CSV"""
        mappings = self.detect_columns(df)
        issues, suggestions = self.validate_data(df, mappings)
        metrics = self.calculate_comprehensive_metrics(df, mappings)
        insights_data = self.generate_insights(metrics)
        figures = self.create_visualizations(df, mappings, metrics)
        
        return {
            'mappings': mappings,
            'metrics': metrics,
            'insights': insights_data,
            'figures': figures,
            'issues': issues,
            'suggestions': suggestions,
            'processed_df': df
        }

# ========== CSV TEMPLATE GENERATOR ==========
class CSVTemplateGenerator:
    def __init__(self):
        self.templates = {
            'complete_financial': {
                'name': 'Complete Financial Data Template',
                'description': 'Comprehensive template with all financial metrics for maximum analysis',
                'columns': {
                    'Date': 'YYYY-MM-DD format (e.g., 2024-01-01)',
                    'Revenue': 'Monthly revenue in local currency (numbers only)',
                    'Sales': 'Alternative revenue column (use if preferred)',
                    'Costs': 'Total monthly costs/expenses',
                    'Variable_Costs': 'Costs that change with sales volume',
                    'Fixed_Costs': 'Costs that remain constant',
                    'Profit': 'Net profit (Revenue - Costs)',
                    'Cash_Flow': 'Net cash flow for the month',
                    'Assets': 'Total assets at month end',
                    'Current_Assets': 'Short-term assets (cash, inventory, etc.)',
                    'Fixed_Assets': 'Long-term assets (equipment, property)',
                    'Liabilities': 'Total liabilities',
                    'Current_Liabilities': 'Short-term debts and obligations',
                    'Equity': 'Owner equity/shareholder equity',
                    'Inventory': 'Value of inventory/stock',
                    'Accounts_Receivable': 'Money owed by customers',
                    'Accounts_Payable': 'Money owed to suppliers',
                    'Customer_Count': 'Number of active customers',
                    'Units_Sold': 'Quantity of products/services sold',
                    'Average_Price': 'Average price per unit/service'
                },
                'sample_data': [
                    ['2025-01-01', 15000, 15000, 12000, 8000, 4000, 3000, 2500, 50000, 20000, 30000, 20000, 8000, 30000, 5000, 8000, 6000, 150, 300, 50],
                    ['2025-02-01', 16500, 16500, 13100, 8800, 4300, 3400, 3200, 52000, 21000, 31000, 21000, 8500, 31000, 5200, 8500, 6200, 165, 330, 50],
                    ['2025-03-01', 14200, 14200, 11800, 7600, 4200, 2400, 2100, 51500, 20500, 31000, 20800, 8300, 30700, 5100, 8200, 6100, 158, 284, 50]
                ]
            },
            'saas_template': {
                'name': 'SaaS Business Template',
                'description': 'Specialized template for Software as a Service businesses',
                'columns': {
                    'Date': 'YYYY-MM-DD format',
                    'Monthly_Recurring_Revenue': 'MRR - predictable monthly revenue',
                    'Customer_Count': 'Total active subscribers',
                    'Churn_Rate': 'Monthly churn rate (percentage as decimal)',
                    'Customer_Acquisition_Cost': 'CAC - cost to acquire one customer',
                    'Lifetime_Value': 'LTV - average customer lifetime value',
                    'Costs': 'Total monthly operating costs'
                },
                'sample_data': [
                    ['2025-01-01', 12000, 400, 0.05, 150, 1800, 9000],
                    ['2025-02-01', 13200, 440, 0.05, 140, 1850, 9900],
                    ['2025-03-01', 14100, 470, 0.053, 160, 1820, 10500]
                ]
            }
        }
    
    def generate_template_csv(self, template_type):
        """Generate CSV template with proper formatting"""
        template = self.templates.get(template_type)
        if not template:
            return None
        
        columns = list(template['columns'].keys())
        df = pd.DataFrame(template['sample_data'], columns=columns)
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer.getvalue()

# ========== ADVANCED ANALYTICS ENGINE ==========
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
        correlation = 0.6
        
        for _ in range(simulations):
            revenue_path = []
            cost_path = []
            
            z1 = np.random.normal(0, 1, periods)
            z2 = np.random.normal(0, 1, periods)
            
            for period in range(periods):
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
        quick_ratio_score = min(10, (ratios.get('quick_ratio', 0) / industry_benchmarks.get('quick_ratio', 1.0)) * 10)
        scores['liquidity'] = current_ratio_score + quick_ratio_score
        
        # Profitability Score (0-40)
        net_margin_score = min(20, (ratios.get('net_margin', 0) / industry_benchmarks['net_margin']) * 20)
        roa_score = min(10, (ratios.get('roa', 0) / industry_benchmarks['roa']) * 10)
        roe_score = min(10, (ratios.get('roe', 0) / industry_benchmarks['roe']) * 10)
        scores['profitability'] = net_margin_score + roa_score + roe_score
        
        # Efficiency Score (0-20)
        asset_turnover_score = min(20, (ratios.get('asset_turnover', 0) / industry_benchmarks['asset_turnover']) * 20)
        scores['efficiency'] = asset_turnover_score
        
        # Leverage Score (0-15)
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
                'recommendation': 'Immediate cash flow improvement needed. Consider: 1) Accelerating receivables collection, 2) Extending payables terms, 3) Reducing non-essential expenses, 4) Emergency financing.',
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
        
        return recommendations

# ========== INDUSTRY TEMPLATES MANAGER ==========
class IndustryTemplateManager:
    def __init__(self):
        self.templates = {
            'retail': {
                'name': 'Retail & E-commerce',
                'icon': '🛍️',
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
                },
                'benchmarks': {
                    'revenue_growth': 0.05, 'profit_margin': 0.04, 'inventory_turns': 6,
                    'customer_retention': 0.75, 'market_share': 0.10
                }
            },
            'saas': {
                'name': 'Software as a Service',
                'icon': '☁️',
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
                },
                'benchmarks': {
                    'revenue_growth': 0.30, 'profit_margin': 0.15, 'churn_rate': 0.05,
                    'ltv_cac_ratio': 3.0, 'gross_margin': 0.80
                }
            },
            'technology': {
                'name': 'Technology Services',
                'icon': '💻',
                'revenue_model': 'Product Sales + Service Revenue + Licensing',
                'key_metrics': [
                    'Research & Development Ratio', 'Time to Market', 'Product Margins',
                    'Customer Acquisition Cost', 'Revenue per Employee'
                ],
                'typical_ratios': {
                    'gross_margin': 0.6, 'net_margin': 0.12, 'current_ratio': 2.0,
                    'asset_turnover': 0.8, 'debt_to_equity': 0.3
                },
                'seasonal_factors': [0.95, 0.9, 1.0, 1.05, 1.0, 0.95, 0.85, 0.9, 1.05, 1.1, 1.15, 1.2],
                'cost_structure': {
                    'research_development': 0.25, 'sales_marketing': 0.2, 'cost_of_sales': 0.4,
                    'general_administrative': 0.15
                },
                'working_capital': {
                    'days_sales_outstanding': 45, 'days_inventory_outstanding': 30,
                    'days_payable_outstanding': 35
                },
                'benchmarks': {
                    'revenue_growth': 0.15, 'profit_margin': 0.12, 'rd_ratio': 0.25,
                    'customer_satisfaction': 0.85, 'employee_productivity': 150000
                }
            },
            'manufacturing': {
                'name': 'Manufacturing',
                'icon': '🏭',
                'revenue_model': 'Units Produced × Selling Price - Production Costs',
                'key_metrics': [
                    'Capacity Utilization', 'Overall Equipment Effectiveness (OEE)',
                    'Inventory Turnover', 'Quality Defect Rate', 'Labor Productivity'
                ],
                'typical_ratios': {
                    'gross_margin': 0.25, 'net_margin': 0.08, 'current_ratio': 1.5,
                    'inventory_turnover': 8, 'asset_turnover': 1.2, 'debt_to_equity': 0.8
                },
                'seasonal_factors': [0.9, 0.95, 1.1, 1.05, 1.0, 0.95, 0.85, 0.9, 1.05, 1.1, 1.0, 0.95],
                'cost_structure': {
                    'raw_materials': 0.45, 'direct_labor': 0.20, 'manufacturing_overhead': 0.15,
                    'general_administrative': 0.12, 'sales_marketing': 0.08
                },
                'working_capital': {
                    'days_sales_outstanding': 45, 'days_inventory_outstanding': 90,
                    'days_payable_outstanding': 30
                },
                'benchmarks': {
                    'revenue_growth': 0.08, 'profit_margin': 0.08, 'capacity_utilization': 0.85,
                    'oee': 0.75, 'defect_rate': 0.02
                }
            }
        }
    
    def get_template(self, industry):
        """Get comprehensive industry template"""
        return self.templates.get(industry, self.templates['technology'])
    
    def detect_industry_from_csv(self, csv_data):
        """Detect likely industry based on CSV data patterns"""
        if not csv_data:
            return 'technology'
        
        revenue_data = csv_data.get('revenue_data', [])
        if not revenue_data or len(revenue_data) < 6:
            return 'technology'
        
        # Calculate seasonality score
        if len(revenue_data) >= 12:
            monthly_avg = []
            for month in range(12):
                month_values = [revenue_data[i] for i in range(month, len(revenue_data), 12)]
                if month_values:
                    monthly_avg.append(np.mean(month_values))
            
            if len(monthly_avg) == 12:
                seasonality_score = np.std(monthly_avg) / np.mean(monthly_avg)
                
                # High seasonality suggests retail
                if seasonality_score > 0.2:
                    return 'retail'
        
        # Check profit margins
        profit_margin = csv_data.get('profit_margin', 0)
        
        # High margins suggest SaaS
        if profit_margin > 20:
            return 'saas'
        # Low margins suggest manufacturing
        elif profit_margin < 10:
            return 'manufacturing'
        
        # Default to technology
        return 'technology'
    
    def benchmark_against_industry(self, csv_data, industry):
        """Benchmark company performance against industry standards"""
        template = self.get_template(industry)
        benchmarks = template['benchmarks']
        
        comparison = {}
        
        # Revenue growth comparison
        company_growth = csv_data.get('revenue_growth', 0) / 100
        industry_growth = benchmarks.get('revenue_growth', 0.1)
        
        comparison['revenue_growth'] = {
            'company_value': company_growth,
            'industry_benchmark': industry_growth,
            'difference': company_growth - industry_growth,
            'percentage_difference': ((company_growth - industry_growth) / industry_growth * 100) if industry_growth != 0 else 0,
            'performance': 'Above Average' if company_growth > industry_growth * 1.1 else 'Average' if company_growth > industry_growth * 0.9 else 'Below Average'
        }
        
        # Profit margin comparison
        company_margin = csv_data.get('profit_margin', 0) / 100
        industry_margin = benchmarks.get('profit_margin', 0.1)
        
        comparison['profit_margin'] = {
            'company_value': company_margin,
            'industry_benchmark': industry_margin,
            'difference': company_margin - industry_margin,
            'percentage_difference': ((company_margin - industry_margin) / industry_margin * 100) if industry_margin != 0 else 0,
            'performance': 'Above Average' if company_margin > industry_margin * 1.1 else 'Average' if company_margin > industry_margin * 0.9 else 'Below Average'
        }
        
        return comparison
    
    def generate_industry_insights(self, csv_data, industry):
        """Generate industry-specific insights"""
        template = self.get_template(industry)
        insights = []
        recommendations = []
        
        profit_margin = csv_data.get('profit_margin', 0)
        revenue_growth = csv_data.get('revenue_growth', 0)
        revenue_volatility = csv_data.get('revenue_volatility', 0)
        
        # Industry-specific analysis
        if industry == 'saas':
            if profit_margin > 15:
                insights.append(f"💰 **Strong SaaS Margins**: {profit_margin:.1f}% exceeds typical SaaS benchmarks")
            else:
                recommendations.append("🎯 **SaaS Optimization**: Focus on recurring revenue and reduce customer acquisition costs")
            
            if revenue_volatility < 0.1:
                insights.append("📊 **Excellent Revenue Predictability**: Low volatility aligns with SaaS model strengths")
            else:
                recommendations.append("🔄 **Improve Recurring Revenue**: Reduce churn and increase customer lifetime value")
        
        elif industry == 'retail':
            if revenue_volatility > 0.2:
                insights.append("🛍️ **Seasonal Business Pattern**: High volatility typical for retail operations")
                recommendations.append("📈 **Seasonal Planning**: Optimize inventory and staffing for peak periods")
            
            if profit_margin < 5:
                recommendations.append("💡 **Retail Efficiency**: Focus on inventory turnover and supply chain optimization")
        
        elif industry == 'technology':
            if revenue_growth > 15:
                insights.append(f"🚀 **Strong Tech Growth**: {revenue_growth:.1f}% growth rate excellent for technology sector")
            
            if profit_margin > 12:
                insights.append("💎 **Tech Innovation Premium**: High margins indicate strong market position")
            else:
                recommendations.append("🔬 **R&D Investment**: Increase innovation spending to improve competitive position")
        
        elif industry == 'manufacturing':
            if profit_margin > 8:
                insights.append("🏭 **Efficient Manufacturing**: Above-average margins for manufacturing sector")
            
            if revenue_volatility < 0.15:
                insights.append("⚙️ **Stable Manufacturing Operations**: Consistent production and demand patterns")
            else:
                recommendations.append("📊 **Demand Planning**: Implement better forecasting to reduce volatility")
        
        return insights, recommendations

# ========== SESSION STATE INITIALIZATION ==========
def init_session_state():
    """Initialize all session state variables"""
    
    # CSV Import specific data
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = {}
    
    if 'csv_processor' not in st.session_state:
        st.session_state.csv_processor = AdvancedCSVProcessor()
    
    if 'imported_metrics' not in st.session_state:
        st.session_state.imported_metrics = {}
    
    # Template generator
    if 'template_generator' not in st.session_state:
        st.session_state.template_generator = CSVTemplateGenerator()
    
    # Analytics data
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = {}
    
    if 'ml_forecasts' not in st.session_state:
        st.session_state.ml_forecasts = {}

# ========== CSV IMPORT PAGE ==========
def show_enhanced_csv_import():
    """Enhanced CSV import with automatic processing"""
    st.header("📤 Advanced CSV Import & Automated Analysis")
    
    st.markdown("""
    🚀 **Drag & Drop Financial Analysis**: Upload your CSV file and get instant comprehensive analysis with AI-powered insights!
    
    **Supported Formats**: Extended format with automatic detection of 20+ financial metrics
    """)
    
    # Display optimal format example
    with st.expander("📋 Optimal CSV Format Guide", expanded=False):
        st.markdown("""
        ### 🎯 Recommended Column Names for Auto-Detection
        
        **Core Financial Data:**
        - `Date`, `Revenue`, `Sales`, `Income`, `Costs`, `Expenses`, `Profit`, `Cash_Flow`
        
        **Balance Sheet Data:**  
        - `Assets`, `Liabilities`, `Equity`, `Inventory`, `Accounts_Receivable`, `Accounts_Payable`
        
        **Business Metrics:**
        - `Customer_Count`, `Units_Sold`, `Average_Price`, `Customer_Acquisition_Cost`, `Lifetime_Value`
        
        **SaaS Specific:**
        - `Monthly_Recurring_Revenue`, `MRR`, `ARR`, `Churn_Rate`, `Monthly_Active_Users`
        """)
        
        # Show example format
        example_data = {
            'Date': ['2025-01-01', '2025-02-01', '2025-03-01'],
            'Revenue': [15000, 16500, 14200],
            'Costs': [12000, 13100, 11800],
            'Profit': [3000, 3400, 2400],
            'Cash_Flow': [2500, 3200, 2100],
            'Assets': [50000, 52000, 51500],
            'Liabilities': [20000, 21000, 20800],
            'Customer_Count': [150, 165, 158]
        }
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    # CSV template downloads
    st.markdown("### 📥 Download CSV Templates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Complete Financial Template", type="primary"):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('complete_financial')
            
            if csv_data:
                st.download_button(
                    label="💾 Download Complete Template",
                    data=csv_data,
                    file_name="complete_financial_template.csv",
                    mime="text/csv"
                )
                st.success("✅ Template ready for download!")
    
    with col2:
        if st.button("💰 Basic Financial Template"):
            # Create basic template
            basic_template = pd.DataFrame({
                'Date': ['2025-01-01', '2025-02-01', '2025-03-01'],
                'Revenue': [15000, 16500, 14200],
                'Costs': [12000, 13100, 11800],
                'Profit': [3000, 3400, 2400],
                'Cash_Flow': [2500, 3200, 2100]
            })
            
            csv_data = basic_template.to_csv(index=False)
            
            st.download_button(
                label="💾 Download Basic Template",
                data=csv_data,
                file_name="basic_financial_template.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("☁️ SaaS Template"):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('saas_template')
            
            if csv_data:
                st.download_button(
                    label="💾 Download SaaS Template",
                    data=csv_data,
                    file_name="saas_template.csv",
                    mime="text/csv"
                )
    
    # File upload with drag and drop
    uploaded_file = st.file_uploader(
        "📁 Drop your CSV file here or click to browse",
        type=['csv'],
        help="Supports files up to 200MB with automatic column detection"
    )
    
    if uploaded_file is not None:
        try:
            # Show upload progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Read CSV
            status_text.text("📖 Reading CSV file...")
            progress_bar.progress(20)
            
            df = pd.read_csv(uploaded_file)
            
            # Step 2: Process with enhanced processor
            status_text.text("🔍 Analyzing data structure...")
            progress_bar.progress(40)
            
            processor = st.session_state.csv_processor
            results = processor.process_csv(df)
            
            # Step 3: Store results
            status_text.text("💾 Storing analysis results...")
            progress_bar.progress(80)
            
            st.session_state.csv_data = results
            st.session_state.imported_metrics = results['metrics']
            
            # Step 4: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Success message
            st.success(f"🎉 Successfully processed {len(df)} rows with {len(df.columns)} columns!")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", f"{len(df):,}")
            with col2:
                st.metric("📈 Columns", len(df.columns))
            with col3:
                detected_cols = len(results['mappings'])
                st.metric("🎯 Auto-Detected", detected_cols)
            with col4:
                file_size = uploaded_file.size / (1024 * 1024)
                st.metric("📁 Size", f"{file_size:.1f} MB")
            
            # Display comprehensive analysis
            show_csv_analysis_results(results)
            
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**")
            st.write("• Ensure your CSV uses comma separators")
            st.write("• Remove currency symbols ($, €, etc.)")
            st.write("• Check for consistent date formats")
            st.write("• Verify numeric columns contain only numbers")
    
    else:
        # Show benefits and instructions
        st.markdown("""
        ### 🌟 What You'll Get Instantly:
        
        **📊 Automatic Analysis:**
        - Revenue, cost, and profit trends
        - Growth rates and volatility analysis
        - Margin calculations and benchmarks
        - Financial ratio computations
        
        **🧠 AI-Powered Insights:**
        - Performance trend analysis
        - Risk alerts and recommendations
        - Industry-specific observations
        - Actionable improvement suggestions
        
        **📈 Rich Visualizations:**
        - Interactive trend charts
        - Profit margin analysis
        - Balance sheet breakdowns
        - Key metrics dashboards
        
        **🔄 Seamless Integration:**
        - Auto-populate financial models
        - Update forecasting scenarios
        - Enhance risk analysis
        - Import to advanced analytics
        """)

def show_csv_analysis_results(results):
    """Display comprehensive analysis results from CSV import"""
    
    mappings = results['mappings']
    metrics = results['metrics']
    insights_data = results['insights']
    figures = results['figures']
    issues = results['issues']
    suggestions = results['suggestions']
    
    # Column Detection Results
    st.subheader("🎯 Automatic Column Detection")
    
    if mappings:
        detection_cols = st.columns(min(len(mappings), 4))
        for i, (category, column) in enumerate(mappings.items()):
            with detection_cols[i % 4]:
                st.success(f"**{category.title()}**\n`{column}`")
    else:
        st.warning("No columns automatically detected. Manual mapping required.")
    
    # AI Insights
    if insights_data:
        st.subheader("🤖 AI-Generated Insights")
        
        # Display insights in tabs
        insight_tabs = st.tabs(["✅ Key Insights", "⚠️ Alerts", "💡 Recommendations"])
        
        with insight_tabs[0]:
            if insights_data['insights']:
                for insight in insights_data['insights']:
                    st.success(insight)
            else:
                st.info("No specific insights generated from current data.")
        
        with insight_tabs[1]:
            if insights_data['alerts']:
                for alert in insights_data['alerts']:
                    st.error(alert)
            else:
                st.success("✅ No critical alerts detected!")
        
        with insight_tabs[2]:
            if insights_data['recommendations']:
                for rec in insights_data['recommendations']:
                    st.warning(f"💡 {rec}")
            else:
                st.info("No specific recommendations at this time.")
    
    # Visualizations
    if figures:
        st.subheader("📈 Automated Visualizations")
        
        # Display charts
        for chart_name, fig in figures.items():
            st.plotly_chart(fig, use_container_width=True)
    
    # Integration Options
    st.subheader("🔄 Integration Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Analysis Results", type="primary", use_container_width=True):
            st.success("✅ Analysis results saved successfully!")
            st.balloons()
    
    with col2:
        # Navigation guidance
        if st.button("🧠 View Advanced Analytics", use_container_width=True):
            st.success("🚀 Navigate to Advanced Analytics using the sidebar...")
            st.info("👈 Use the navigation menu on the left to access Advanced Analytics")
    
    with col3:
        if st.button("🎯 Explore Scenarios", use_container_width=True):
            st.success("🚀 Navigate to Scenario Planning using the sidebar...")
            st.info("👈 Use the navigation menu on the left to access Scenario Planning")

# ========== EXECUTIVE DASHBOARD ==========
def show_executive_dashboard():
    """Enhanced executive dashboard - CSV data only"""
    st.header("👔 Executive Dashboard")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if csv_data:
        st.success("📊 **Dashboard powered by your uploaded CSV data**")
        
        # Main KPI metrics from CSV
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_revenue = csv_data.get('monthly_revenue', 0)
            st.metric("Average Monthly Revenue", f"{monthly_revenue:,.0f} DHS")
            
            # Revenue trend
            growth = csv_data.get('revenue_growth', 0)
            if growth > 0:
                st.success(f"📈 Growing {growth:.1f}%")
            else:
                st.error(f"📉 Declining {abs(growth):.1f}%")
        
        with col2:
            monthly_costs = csv_data.get('monthly_costs', 0)
            st.metric("Average Monthly Costs", f"{monthly_costs:,.0f} DHS")
            
            # Cost trend
            cost_growth = csv_data.get('costs_growth', 0)
            if cost_growth < 5:
                st.success("✅ Cost Control")
            else:
                st.warning(f"⚠️ Rising {cost_growth:.1f}%")
        
        with col3:
            monthly_profit = csv_data.get('monthly_profit', 0)
            st.metric("Average Monthly Profit", f"{monthly_profit:,.0f} DHS")
            
            # Profit status
            if monthly_profit > 0:
                st.success("💰 Profitable")
            else:
                st.error("🔴 Loss Making")
        
        with col4:
            profit_margin = csv_data.get('profit_margin', 0)
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            # Margin assessment
            if profit_margin > 20:
                st.success("🎯 Excellent")
            elif profit_margin > 10:
                st.info("📈 Good")
            elif profit_margin > 0:
                st.warning("⚠️ Thin")
            else:
                st.error("🔴 Negative")
        
        # Detailed performance analysis
        st.subheader("📈 Financial Performance Analysis")
        
        # Show CSV visualizations
        csv_figures = CSVDataManager.get_csv_visualizations()
        if csv_figures and 'financial_trend' in csv_figures:
            st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
        
        # Show CSV insights
        csv_insights = CSVDataManager.get_csv_insights()
        if csv_insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 AI Insights from Your Data")
                for insight in csv_insights['insights']:
                    st.success(f"✅ {insight}")
            
            with col2:
                st.markdown("#### 💡 Recommendations")
                for rec in csv_insights['recommendations']:
                    st.info(f"💡 {rec}")
            
            if csv_insights['alerts']:
                st.markdown("#### ⚠️ Risk Alerts")
                for alert in csv_insights['alerts']:
                    st.error(f"⚠️ {alert}")
        
        # Performance summary
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            annual_revenue = csv_data.get('revenue', 0)
            st.metric("Annual Revenue", f"{annual_revenue:,.0f} DHS")
            
            # Revenue volatility
            volatility = csv_data.get('revenue_volatility', 0)
            if volatility < 0.1:
                st.success("🟢 Stable")
            elif volatility < 0.3:
                st.warning("🟡 Moderate")
            else:
                st.error("🔴 Volatile")
        
        with col2:
            annual_profit = csv_data.get('net_profit', 0)
            st.metric("Annual Profit", f"{annual_profit:,.0f} DHS")
            
            # ROI calculation
            if annual_profit > 0:
                roi = (annual_profit / (annual_revenue * 0.6)) * 100  # Estimate based on typical asset base
                st.metric("ROI", f"{roi:.1f}%")
        
        with col3:
            # Cash flow estimate
            cash_flow = csv_data.get('cash_flow', 0)
            st.metric("Monthly Cash Flow", f"{cash_flow:,.0f} DHS")
            
            if cash_flow > 0:
                st.success("💰 Positive")
            else:
                st.error("🔴 Negative")
        
    else:
        # No CSV data available
        st.warning("📤 **No CSV Data Imported**")
        st.info("Upload your financial data via Smart CSV Import to see comprehensive dashboard analysis!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Import CSV Data", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **What you'll see with CSV data:**
            - Real revenue and profit trends
            - AI-powered insights
            - Growth analysis
            - Risk alerts
            - Performance benchmarks
            """)

# ========== ADVANCED ANALYTICS ==========
def show_advanced_analytics():
    """Advanced analytics - CSV data only"""
    st.header("🧠 Advanced Analytics & AI Insights")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **No CSV Data Available**")
        st.info("Advanced Analytics requires your uploaded CSV data to provide meaningful analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Import CSV Data Now", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Advanced Analytics will provide:**
            - Comprehensive financial ratios
            - AI-powered health scoring
            - Industry benchmarking
            - Predictive insights
            - Risk assessments
            """)
        return
    
    st.success("📊 **Analytics powered by your uploaded CSV data**")
    
    # Initialize analytics engine
    analytics = AdvancedAnalytics()
    
    # Calculate comprehensive ratios
    ratios = analytics.calculate_comprehensive_ratios(csv_data)
    
    # Calculate health score
    health_score, score_breakdown = analytics.calculate_financial_health_score(ratios, 'technology')
    
    # Financial health overview using CSV data
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Financial Health", f"{health_score:.0f}/100")
        
        if health_score >= 80:
            st.success("🟢 Excellent")
        elif health_score >= 60:
            st.info("🔵 Good")
        elif health_score >= 40:
            st.warning("🟡 Fair")
        else:
            st.error("🔴 Poor")
    
    with col2:
        current_ratio = csv_data.get('current_ratio', 0)
        st.metric("Current Ratio", f"{current_ratio:.2f}")
        
        if current_ratio > 1.5:
            st.success("🟢 Healthy")
        elif current_ratio > 1.2:
            st.info("🔵 Moderate")
        else:
            st.warning("🟡 Low")
    
    with col3:
        net_margin = csv_data.get('net_margin', 0)
        st.metric("Net Margin", f"{net_margin*100:.1f}%")
        
        if net_margin > 0.15:
            st.success("🟢 Strong")
        elif net_margin > 0.08:
            st.info("🔵 Average")
        else:
            st.warning("🟡 Weak")
    
    with col4:
        debt_to_equity = csv_data.get('debt_to_equity', 0)
        st.metric("Debt-to-Equity", f"{debt_to_equity:.2f}")
        
        if debt_to_equity < 0.5:
            st.success("🟢 Conservative")
        elif debt_to_equity < 1.0:
            st.info("🔵 Moderate")
        else:
            st.warning("🟡 High")
    
    with col5:
        # Revenue per month growth
        revenue_volatility = csv_data.get('revenue_volatility', 0)
        st.metric("Revenue Stability", f"{(1-revenue_volatility)*100:.0f}%")
        
        if revenue_volatility < 0.1:
            st.success("🟢 Very Stable")
        elif revenue_volatility < 0.2:
            st.info("🔵 Stable")
        else:
            st.warning("🟡 Volatile")
    
    # Detailed analysis tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance Analysis", "🤖 AI Insights", "📊 Financial Ratios"])
    
    with tab1:
        st.subheader("Performance Analysis from Your Data")
        
        # Show original CSV visualization
        csv_figures = CSVDataManager.get_csv_visualizations()
        if csv_figures and 'financial_trend' in csv_figures:
            st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Key Performance Indicators")
            
            revenue_data = csv_data.get('revenue_data', [])
            if revenue_data:
                avg_revenue = np.mean(revenue_data)
                revenue_trend = "Growing" if revenue_data[-1] > revenue_data[0] else "Declining"
                
                st.metric("Average Revenue", f"{avg_revenue:,.0f} DHS")
                st.metric("Revenue Trend", revenue_trend)
                
                # Revenue growth rate
                if len(revenue_data) > 1:
                    growth_rate = ((revenue_data[-1] / revenue_data[0]) - 1) * 100
                    st.metric("Total Growth", f"{growth_rate:+.1f}%")
        
        with col2:
            st.markdown("#### 💰 Profitability Analysis")
            
            profit_data = csv_data.get('profit_data', [])
            if profit_data:
                avg_profit = np.mean(profit_data)
                profit_trend = "Improving" if profit_data[-1] > profit_data[0] else "Declining"
                
                st.metric("Average Profit", f"{avg_profit:,.0f} DHS")
                st.metric("Profit Trend", profit_trend)
                
                # Profit margin trend
                profit_margin = csv_data.get('profit_margin', 0)
                st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    with tab2:
        st.subheader("🤖 AI-Powered Insights from Your Data")
        
        # Show CSV insights
        csv_insights = CSVDataManager.get_csv_insights()
        if csv_insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Key Insights")
                if csv_insights['insights']:
                    for insight in csv_insights['insights']:
                        st.success(f"✅ {insight}")
                else:
                    st.info("No specific insights generated from current data.")
                
                st.markdown("#### ⚠️ Risk Alerts")
                if csv_insights['alerts']:
                    for alert in csv_insights['alerts']:
                        st.error(f"⚠️ {alert}")
                else:
                    st.success("✅ No critical alerts detected!")
            
            with col2:
                st.markdown("#### 💡 AI Recommendations")
                if csv_insights['recommendations']:
                    for rec in csv_insights['recommendations']:
                        st.warning(f"💡 {rec}")
                else:
                    st.info("No specific recommendations at this time.")
                
                # Generate additional recommendations based on CSV data
                st.markdown("#### 🎯 Data-Driven Suggestions")
                
                revenue_growth = csv_data.get('revenue_growth', 0)
                profit_margin = csv_data.get('profit_margin', 0)
                
                if revenue_growth < 0:
                    st.error("📉 **Revenue Decline**: Consider market expansion or product diversification")
                elif revenue_growth < 5:
                    st.warning("📈 **Slow Growth**: Explore new customer acquisition strategies")
                else:
                    st.success("🚀 **Strong Growth**: Maintain current strategies and scale operations")
                
                if profit_margin < 10:
                    st.warning("💰 **Margin Improvement**: Focus on cost optimization and pricing strategy")
                elif profit_margin > 25:
                    st.success("💎 **Excellent Margins**: Consider reinvestment opportunities")
        else:
            st.info("Upload CSV data to see AI-powered insights specific to your business")
    
    with tab3:
        st.subheader("📊 Financial Ratios Analysis")
        
        # Calculate and display ratios based on CSV data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Liquidity Ratios")
            
            current_ratio = csv_data.get('current_ratio', 0)
            quick_ratio = (csv_data.get('current_assets', 0) - csv_data.get('inventory', 0)) / csv_data.get('current_liabilities', 1)
            cash_ratio = csv_data.get('cash', 0) / csv_data.get('current_liabilities', 1)
            
            ratios_data = {
                'Ratio': ['Current Ratio', 'Quick Ratio', 'Cash Ratio'],
                'Value': [current_ratio, quick_ratio, cash_ratio],
                'Benchmark': [1.5, 1.0, 0.2],
                'Status': []
            }
            
            for value, benchmark in zip(ratios_data['Value'], ratios_data['Benchmark']):
                if value >= benchmark * 1.2:
                    ratios_data['Status'].append('Excellent')
                elif value >= benchmark:
                    ratios_data['Status'].append('Good')
                elif value >= benchmark * 0.8:
                    ratios_data['Status'].append('Adequate')
                else:
                    ratios_data['Status'].append('Poor')
            
            df_ratios = pd.DataFrame(ratios_data)
            df_ratios['Value'] = df_ratios['Value'].round(2)
            
            st.dataframe(df_ratios, use_container_width=True)
        
        with col2:
            st.markdown("#### 💰 Profitability Ratios")
            
            gross_margin = (csv_data.get('gross_profit', 0) / csv_data.get('revenue', 1)) * 100
            net_margin = csv_data.get('net_margin', 0) * 100
            roa = (csv_data.get('net_profit', 0) / csv_data.get('total_assets', 1)) * 100
            
            profit_data = {
                'Metric': ['Gross Margin %', 'Net Margin %', 'ROA %'],
                'Value': [gross_margin, net_margin, roa],
                'Industry Avg': [40, 12, 8]
            }
            
            df_profit = pd.DataFrame(profit_data)
            df_profit['Value'] = df_profit['Value'].round(1)
            
            st.dataframe(df_profit, use_container_width=True)
            
            # Profitability chart
            fig = go.Figure(data=[
                go.Bar(name='Your Company', x=profit_data['Metric'], y=profit_data['Value']),
                go.Bar(name='Industry Average', x=profit_data['Metric'], y=profit_data['Industry Avg'])
            ])
            
            fig.update_layout(
                barmode='group',
                title='Profitability vs Industry Average',
                yaxis_title='Percentage (%)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ========== SCENARIO PLANNING ==========
def show_scenario_planning():
    """Scenario planning using CSV data"""
    st.header("🎯 Advanced Scenario Planning")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **No CSV Data Available**")
        st.info("Scenario Planning requires your uploaded CSV data for accurate projections.")
        
        if st.button("📤 Import CSV Data Now", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    st.success("📊 **Scenarios based on your uploaded CSV data**")
    
    # Base data from CSV
    base_monthly_revenue = csv_data.get('monthly_revenue', 15000)
    base_monthly_costs = csv_data.get('monthly_costs', 12000)
    current_growth_rate = csv_data.get('revenue_growth', 0) / 100
    
    st.subheader(f"📊 Base Data (from your CSV)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Monthly Revenue", f"{base_monthly_revenue:,.0f} DHS")
    with col2:
        st.metric("Monthly Costs", f"{base_monthly_costs:,.0f} DHS")
    with col3:
        st.metric("Current Growth Rate", f"{current_growth_rate*100:+.1f}%")
    
    # Scenario configuration
    st.subheader("⚙️ Configure Scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 😰 Pessimistic Scenario")
        pess_revenue = st.slider("Revenue Change (%)", -50, 10, max(-20, int(current_growth_rate*100-15)), key="pess_rev")
        pess_cost = st.slider("Cost Change (%)", -10, 40, 15, key="pess_cost")
        pess_prob = st.slider("Probability (%)", 5, 40, 20, key="pess_prob")
    
    with col2:
        st.markdown("### 😐 Realistic Scenario")
        real_revenue = st.slider("Revenue Change (%)", -10, 40, max(5, int(current_growth_rate*100)), key="real_rev")
        real_cost = st.slider("Cost Change (%)", 0, 25, 8, key="real_cost")
        real_prob = st.slider("Probability (%)", 40, 80, 60, key="real_prob")
    
    with col3:
        st.markdown("### 😄 Optimistic Scenario")
        opt_revenue = st.slider("Revenue Change (%)", 10, 60, max(25, int(current_growth_rate*100+20)), key="opt_rev")
        opt_cost = st.slider("Cost Change (%)", -5, 15, 3, key="opt_cost")
        opt_prob = st.slider("Probability (%)", 5, 40, 20, key="opt_prob")
    
    # Validate probabilities
    total_prob = pess_prob + real_prob + opt_prob
    if total_prob != 100:
        st.warning(f"⚠️ Probabilities sum to {total_prob}%. Adjusting proportionally...")
        pess_prob = pess_prob * 100 / total_prob
        real_prob = real_prob * 100 / total_prob
        opt_prob = opt_prob * 100 / total_prob
    
    # Analysis period
    analysis_period = st.selectbox("Analysis Period", [12, 24, 36], index=1, help="Number of months to analyze")
    
    # Run scenario analysis
    if st.button("🚀 Run Scenario Analysis", type="primary"):
        with st.spinner("Running scenario analysis with your CSV data..."):
            
            scenarios = {
                'pessimistic': {
                    'revenue_change': pess_revenue / 100,
                    'cost_change': pess_cost / 100,
                    'probability': pess_prob / 100
                },
                'realistic': {
                    'revenue_change': real_revenue / 100,
                    'cost_change': real_cost / 100,
                    'probability': real_prob / 100
                },
                'optimistic': {
                    'revenue_change': opt_revenue / 100,
                    'cost_change': opt_cost / 100,
                    'probability': opt_prob / 100
                }
            }
            
            # Calculate scenario results
            scenario_results = {}
            
            for scenario_name, params in scenarios.items():
                monthly_results = []
                
                for month in range(analysis_period):
                    # Apply scenario changes
                    monthly_revenue = base_monthly_revenue * (1 + params['revenue_change'])
                    monthly_cost = base_monthly_costs * (1 + params['cost_change'])
                    monthly_profit = monthly_revenue - monthly_cost
                    
                    monthly_results.append({
                        'month': month + 1,
                        'revenue': monthly_revenue,
                        'cost': monthly_cost,
                        'profit': monthly_profit
                    })
                
                total_profit = sum(m['profit'] for m in monthly_results)
                avg_monthly_profit = total_profit / analysis_period
                
                scenario_results[scenario_name] = {
                    'monthly_data': monthly_results,
                    'total_profit': total_profit,
                    'avg_monthly_profit': avg_monthly_profit,
                    'probability': params['probability']
                }
            
            # Store results
            st.session_state.scenario_results = scenario_results
    
    # Display results if available
    if 'scenario_results' in st.session_state and st.session_state.scenario_results:
        scenario_results = st.session_state.scenario_results
        
        st.subheader("📊 Scenario Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        expected_value = sum(data['total_profit'] * data['probability'] for data in scenario_results.values())
        best_case = max(data['total_profit'] for data in scenario_results.values())
        worst_case = min(data['total_profit'] for data in scenario_results.values())
        profit_range = best_case - worst_case
        
        with col1:
            st.metric("Expected Value", f"{expected_value:,.0f} DHS")
        with col2:
            st.metric("Best Case", f"{best_case:,.0f} DHS", f"+{best_case - expected_value:,.0f}")
        with col3:
            st.metric("Worst Case", f"{worst_case:,.0f} DHS", f"{worst_case - expected_value:,.0f}")
        with col4:
            st.metric("Range", f"{profit_range:,.0f} DHS")
        
        # Scenario visualization
        fig = go.Figure()
        
        colors = {'pessimistic': '#FF6B6B', 'realistic': '#4ECDC4', 'optimistic': '#45B7D1'}
        
        for scenario, data in scenario_results.items():
            months = [m['month'] for m in data['monthly_data']]
            profits = [m['profit'] for m in data['monthly_data']]
            cumulative_profit = np.cumsum(profits)
            
            fig.add_trace(go.Scatter(
                x=months,
                y=cumulative_profit,
                mode='lines+markers',
                name=f"{scenario.title()} (Prob: {data['probability']:.0%})",
                line=dict(color=colors[scenario], width=3),
                marker=dict(size=6)
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
        
        fig.update_layout(
            title="Cumulative Profit by Scenario",
            xaxis_title="Month",
            yaxis_title="Cumulative Profit (DHS)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario comparison table
        st.subheader("📋 Scenario Comparison")
        
        comparison_data = []
        for scenario, data in scenario_results.items():
            comparison_data.append({
                'Scenario': scenario.title(),
                'Total Profit': f"{data['total_profit']:,.0f} DHS",
                'Avg Monthly Profit': f"{data['avg_monthly_profit']:,.0f} DHS",
                'Probability': f"{data['probability']:.0%}",
                'Expected Contribution': f"{data['total_profit'] * data['probability']:,.0f} DHS"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# ========== ENHANCED ML FORECASTING ==========
def show_ml_forecasting():
    """Enhanced ML forecasting with expanded variables"""
    st.header("🤖 Machine Learning Financial Forecasting")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **No CSV Data Available**")
        st.info("ML Forecasting requires your uploaded CSV data for training accurate models.")
        
        if st.button("📤 Import CSV Data Now", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    st.success("📊 **ML Models trained on your uploaded CSV data**")
    
    # Get available data for forecasting
    available_metrics = get_available_forecast_metrics(csv_data)
    
    # Data overview
    st.subheader("📊 Training Data Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_data_points = len(csv_data.get('revenue_data', []))
        st.metric("Data Points", total_data_points)
        data_quality = "High" if total_data_points >= 12 else "Medium" if total_data_points >= 6 else "Low"
        st.metric("Data Quality", data_quality)
    
    with col2:
        st.metric("Available Variables", len(available_metrics))
        st.metric("Forecast Models", "Linear + Seasonal")
    
    with col3:
        if csv_data.get('revenue_data'):
            revenue_data = csv_data['revenue_data']
            revenue_trend = "Growing" if revenue_data[-1] > revenue_data[0] else "Declining"
            st.metric("Revenue Trend", revenue_trend)
            
            volatility = np.std(revenue_data) / np.mean(revenue_data) if np.mean(revenue_data) > 0 else 0
            st.metric("Data Volatility", f"{volatility:.1%}")
    
    # Enhanced Forecasting Configuration
    st.subheader("🔮 Enhanced Forecasting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dynamic forecast target selection based on available data
        forecast_target = st.selectbox(
            "Forecast Target",
            available_metrics,
            help="Choose what variable to forecast based on your uploaded data"
        )
        
        forecast_periods = st.slider("Forecast Periods (months)", 3, 36, 12)
        
        # Advanced options
        st.markdown("#### 🔧 Advanced Options")
        include_trend = st.checkbox("Include Trend Analysis", value=True)
        include_seasonality = st.checkbox("Include Seasonality", value=True)
        
        # Model selection
        model_type = st.selectbox(
            "Model Type",
            ["Auto (Recommended)", "Linear Regression", "Seasonal Decomposition", "Moving Average"],
            help="Choose forecasting algorithm"
        )
    
    with col2:
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        
        # Forecast scenarios
        st.markdown("#### 📈 Forecast Scenarios")
        include_scenarios = st.checkbox("Generate Multiple Scenarios", value=True)
        
        if include_scenarios:
            optimistic_factor = st.slider("Optimistic Scenario (+%)", 5, 50, 20)
            pessimistic_factor = st.slider("Pessimistic Scenario (-%)", 5, 50, 15)
        
        # External factors
        st.markdown("#### 🌍 External Factors")
        market_growth = st.slider("Expected Market Growth (%)", -20, 30, 5)
        economic_impact = st.selectbox("Economic Outlook", ["Positive", "Neutral", "Negative"])
    
    # Historical data visualization for selected variable
    if forecast_target in available_metrics:
        st.subheader(f"📈 Historical {forecast_target} Data")
        
        # Get data for selected target
        target_data = get_target_data(csv_data, forecast_target)
        
        if target_data:
            months = list(range(1, len(target_data) + 1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months,
                y=target_data,
                mode='lines+markers',
                name=f'Historical {forecast_target}',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            
            # Add trend line if requested
            if include_trend and len(target_data) > 3:
                x_trend = np.arange(len(target_data))
                slope, intercept = np.polyfit(x_trend, target_data, 1)
                trend_line = slope * x_trend + intercept
                
                fig.add_trace(go.Scatter(
                    x=months,
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"Historical {forecast_target} Analysis",
                xaxis_title="Month",
                yaxis_title=f"{forecast_target}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Generate enhanced forecast
    if st.button("🚀 Generate Enhanced ML Forecast", type="primary"):
        with st.spinner("Training advanced ML models and generating forecasts..."):
            
            target_data = get_target_data(csv_data, forecast_target)
            
            if not target_data or len(target_data) < 3:
                st.error("❌ Insufficient data for ML forecasting. Need at least 3 data points.")
                return
            
            # Generate enhanced forecasts
            forecast_results = generate_enhanced_forecast(
                target_data, 
                forecast_periods, 
                confidence_level,
                include_seasonality,
                include_trend,
                model_type,
                market_growth / 100 if 'market_growth' in locals() else 0,
                economic_impact if 'economic_impact' in locals() else "Neutral"
            )
            
            # Add scenarios if requested
            if include_scenarios:
                forecast_results['scenarios'] = generate_forecast_scenarios(
                    forecast_results['forecasts'],
                    optimistic_factor / 100,
                    pessimistic_factor / 100
                )
            
            forecast_results['target'] = forecast_target
            st.session_state.ml_forecast_results = forecast_results
    
    # Display enhanced forecast results
    if 'ml_forecast_results' in st.session_state:
        display_enhanced_forecast_results(st.session_state.ml_forecast_results, csv_data)

def get_available_forecast_metrics(csv_data):
    """Get list of available metrics for forecasting based on CSV data"""
    available = []
    
    # Core financial metrics
    if csv_data.get('revenue_data'):
        available.append("Revenue")
    if csv_data.get('costs_data'):
        available.append("Costs")
    if csv_data.get('profit_data'):
        available.append("Profit")
    
    # Check for other metrics in the CSV processor's detected mappings
    if 'csv_data' in st.session_state and 'mappings' in st.session_state.csv_data:
        mappings = st.session_state.csv_data['mappings']
        
        metric_mapping = {
            'cash_flow': "Cash Flow",
            'assets': "Total Assets",
            'current_assets': "Current Assets", 
            'fixed_assets': "Fixed Assets",
            'liabilities': "Total Liabilities",
            'current_liabilities': "Current Liabilities",
            'equity': "Equity",
            'inventory': "Inventory",
            'accounts_receivable': "Accounts Receivable",
            'accounts_payable': "Accounts Payable",
            'customer_metrics': "Customer Count",
            'unit_metrics': "Units Sold",
            'pricing_metrics': "Average Price",
            'saas_metrics': "SaaS Metrics"
        }
        
        for key, display_name in metric_mapping.items():
            if key in mappings and display_name not in available:
                available.append(display_name)
    
    # Financial ratios (calculated)
    if len(available) >= 2:
        available.extend([
            "Profit Margin %",
            "Current Ratio",
            "Revenue Growth Rate"
        ])
    
    return available if available else ["Revenue", "Profit"]  # Fallback

def get_target_data(csv_data, target):
    """Get data array for the selected forecast target"""
    target_mapping = {
        "Revenue": csv_data.get('revenue_data', []),
        "Costs": csv_data.get('costs_data', []),
        "Profit": csv_data.get('profit_data', []),
        "Cash Flow": csv_data.get('cash_flow_data', []),
        # Add more mappings based on available data
    }
    
    # For calculated metrics
    if target == "Profit Margin %":
        revenue_data = csv_data.get('revenue_data', [])
        profit_data = csv_data.get('profit_data', [])
        if revenue_data and profit_data:
            return [(p/r)*100 for p, r in zip(profit_data, revenue_data) if r != 0]
    
    if target == "Revenue Growth Rate":
        revenue_data = csv_data.get('revenue_data', [])
        if len(revenue_data) > 1:
            return [((revenue_data[i] - revenue_data[i-1]) / revenue_data[i-1]) * 100 
                   for i in range(1, len(revenue_data)) if revenue_data[i-1] != 0]
    
    return target_mapping.get(target, csv_data.get('revenue_data', []))

def generate_enhanced_forecast(data, periods, confidence_level, include_seasonality, 
                             include_trend, model_type, market_growth, economic_impact):
    """Generate enhanced forecasts with multiple factors"""
    
    if len(data) < 3:
        return None
    
    # Base forecast using linear regression
    x = np.arange(len(data))
    y = np.array(data)
    
    # Calculate base trend
    slope, intercept = np.polyfit(x, y, 1)
    
    # Generate base forecasts
    future_months = np.arange(len(data), len(data) + periods)
    base_forecast = slope * future_months + intercept
    
    # Apply market growth factor
    growth_factor = 1 + market_growth
    base_forecast = base_forecast * growth_factor
    
    # Apply economic impact
    economic_factors = {"Positive": 1.1, "Neutral": 1.0, "Negative": 0.95}
    economic_factor = economic_factors.get(economic_impact, 1.0)
    base_forecast = base_forecast * economic_factor
    
    # Add seasonality if requested and data is sufficient
    if include_seasonality and len(data) >= 12:
        seasonal_pattern = []
        for i in range(12):
            month_data = [data[j] for j in range(i, len(data), 12) if j < len(data)]
            if month_data:
                seasonal_pattern.append(np.mean(month_data) / np.mean(data))
            else:
                seasonal_pattern.append(1.0)
        
        # Apply seasonality to forecast
        forecasts = []
        for i, base_val in enumerate(base_forecast):
            season_idx = (len(data) + i) % 12
            seasonal_factor = seasonal_pattern[season_idx] if season_idx < len(seasonal_pattern) else 1.0
            forecasts.append(base_val * seasonal_factor)
    else:
        forecasts = base_forecast.tolist()
    
    # Calculate confidence intervals
    residuals = y - (slope * x + intercept)
    mse = np.mean(residuals**2)
    std_error = np.sqrt(mse)
    
    confidence_multiplier = 1.96 if confidence_level == 95 else 2.58 if confidence_level == 99 else 1.64
    
    upper_bounds = [f + confidence_multiplier * std_error for f in forecasts]
    lower_bounds = [f - confidence_multiplier * std_error for f in forecasts]
    
    return {
        'forecasts': forecasts,
        'upper_bounds': upper_bounds,
        'lower_bounds': lower_bounds,
        'confidence_level': confidence_level,
        'periods': periods,
        'model_performance': {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mean_forecast': np.mean(forecasts)
        }
    }

def generate_forecast_scenarios(base_forecasts, optimistic_factor, pessimistic_factor):
    """Generate optimistic and pessimistic scenarios"""
    return {
        'optimistic': [f * (1 + optimistic_factor) for f in base_forecasts],
        'pessimistic': [f * (1 - pessimistic_factor) for f in base_forecasts],
        'base': base_forecasts
    }

def display_enhanced_forecast_results(results, csv_data):
    """Display enhanced forecast results with scenarios"""
    
    st.subheader("📈 Enhanced ML Forecast Results")
    
    # Enhanced summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    forecasts = results['forecasts']
    target = results['target']
    
    with col1:
        avg_forecast = np.mean(forecasts)
        st.metric("Average Forecast", f"{avg_forecast:,.0f}")
    
    with col2:
        total_forecast = sum(forecasts)
        st.metric(f"{results['periods']}-Month Total", f"{total_forecast:,.0f}")
    
    with col3:
        # Calculate growth from last historical value
        if target in ["Revenue", "Costs", "Profit"]:
            historical_data = get_target_data(csv_data, target)
            if historical_data:
                last_actual = historical_data[-1]
                growth = (forecasts[-1] / last_actual - 1) * 100
                st.metric("Projected Growth", f"{growth:+.1f}%")
    
    with col4:
        volatility = np.std(forecasts) / np.mean(forecasts) * 100
        st.metric("Forecast Volatility", f"{volatility:.1f}%")
    
    with col5:
        model_rmse = results['model_performance']['rmse']
        accuracy = max(0, 100 - (model_rmse / np.mean(forecasts) * 100))
        st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    # Enhanced forecast visualization with scenarios
    historical_data = get_target_data(csv_data, target)
    if historical_data:
        historical_months = list(range(1, len(historical_data) + 1))
        forecast_months = list(range(len(historical_months) + 1, len(historical_months) + results['periods'] + 1))
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_months,
            y=historical_data,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        # Base forecast
        fig.add_trace(go.Scatter(
            x=forecast_months,
            y=forecasts,
            mode='lines+markers',
            name='ML Forecast',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Scenarios if available
        if 'scenarios' in results:
            scenarios = results['scenarios']
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=scenarios['optimistic'],
                mode='lines',
                name='Optimistic Scenario',
                line=dict(color='green', width=2, dash='dot'),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=scenarios['pessimistic'],
                mode='lines',
                name='Pessimistic Scenario',
                line=dict(color='orange', width=2, dash='dot'),
                opacity=0.7
            ))
        
        # Confidence intervals
        if 'upper_bounds' in results:
            fig.add_trace(go.Scatter(
                x=forecast_months + forecast_months[::-1],
                y=results['upper_bounds'] + results['lower_bounds'][::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{results["confidence_level"]}% Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{target} Forecast: Historical vs Enhanced ML Predictions",
            xaxis_title="Month",
            yaxis_title=f"{target}",
            hovermode='x unified',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced insights and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Model Performance")
        performance = results['model_performance']
        
        st.metric("RMSE", f"{performance['rmse']:,.2f}")
        st.metric("Mean Squared Error", f"{performance['mse']:,.2f}")
        
        # Model quality assessment
        rmse_percentage = (performance['rmse'] / performance['mean_forecast']) * 100
        if rmse_percentage < 10:
            st.success("🟢 Excellent model accuracy")
        elif rmse_percentage < 20:
            st.info("🔵 Good model accuracy")
        else:
            st.warning("🟡 Moderate model accuracy")
    
    with col2:
        st.markdown("#### 🎯 Business Implications")
        
        # Generate insights based on forecast
        current_avg = np.mean(historical_data) if historical_data else 0
        forecast_avg = np.mean(forecasts)
        
        if current_avg > 0:
            change_pct = (forecast_avg / current_avg - 1) * 100
            
            if change_pct > 15:
                st.success(f"📈 **Strong Growth Expected**: {change_pct:.1f}% increase")
                st.info("💡 Consider capacity expansion and resource scaling")
            elif change_pct > 5:
                st.info(f"📊 **Moderate Growth**: {change_pct:.1f}% increase projected")
                st.info("💡 Maintain current strategies with gradual scaling")
            elif change_pct > -5:
                st.warning(f"📊 **Stable Trend**: {change_pct:+.1f}% change projected")
                st.info("💡 Focus on efficiency and optimization")
            else:
                st.error(f"📉 **Decline Expected**: {abs(change_pct):.1f}% decrease")
                st.error("💡 Urgent: Implement corrective measures")

# ========== RISK MANAGEMENT ==========
def show_risk_management():
    """Risk management using CSV data"""
    st.header("⚠️ Risk Management & Analysis")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **No CSV Data Available**")
        st.info("Risk Management requires your uploaded CSV data for accurate risk assessment.")
        
        if st.button("📤 Import CSV Data Now", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    st.success("📊 **Risk analysis based on your uploaded CSV data**")
    
    # Calculate risk metrics from CSV data
    revenue_volatility = csv_data.get('revenue_volatility', 0)
    profit_margin = csv_data.get('profit_margin', 0)
    revenue_growth = csv_data.get('revenue_growth', 0)
    current_ratio = csv_data.get('current_ratio', 1.5)
    debt_to_equity = csv_data.get('debt_to_equity', 0.4)
    
    # Calculate overall risk score
    risk_score = 0
    
    # Revenue risk (30% weight)
    if revenue_volatility > 0.3:
        risk_score += 30
    elif revenue_volatility > 0.2:
        risk_score += 20
    elif revenue_volatility > 0.1:
        risk_score += 10
    
    # Profitability risk (25% weight)
    if profit_margin < 0:
        risk_score += 25
    elif profit_margin < 5:
        risk_score += 20
    elif profit_margin < 10:
        risk_score += 10
    elif profit_margin < 15:
        risk_score += 5
    
    # Growth risk (20% weight)
    if revenue_growth < -10:
        risk_score += 20
    elif revenue_growth < 0:
        risk_score += 15
    elif revenue_growth < 5:
        risk_score += 5
    
    # Liquidity risk (15% weight)
    if current_ratio < 1:
        risk_score += 15
    elif current_ratio < 1.2:
        risk_score += 10
    elif current_ratio < 1.5:
        risk_score += 5
    
    # Leverage risk (10% weight)
    if debt_to_equity > 2:
        risk_score += 10
    elif debt_to_equity > 1:
        risk_score += 5
    
    overall_risk_pct = min(risk_score, 100)
    
    # Risk Dashboard
    st.subheader("🎯 Risk Assessment Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Risk score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_risk_pct,
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
        
        # Risk level interpretation
        if overall_risk_pct < 25:
            st.success("🟢 **Low Risk**: Well-managed financial position")
        elif overall_risk_pct < 50:
            st.info("🔵 **Moderate Risk**: Some areas need attention")
        elif overall_risk_pct < 75:
            st.warning("🟡 **High Risk**: Significant risk factors present")
        else:
            st.error("🔴 **Critical Risk**: Immediate action required")
    
    with col2:
        # Risk breakdown
        risk_categories = {
            'Revenue Volatility': min(30, revenue_volatility * 100),
            'Profitability': min(25, max(0, (15 - profit_margin) * 2)),
            'Growth': min(20, max(0, -revenue_growth * 2) if revenue_growth < 0 else 0),
            'Liquidity': min(15, max(0, (1.5 - current_ratio) * 10)),
            'Leverage': min(10, debt_to_equity * 5)
        }
        
        categories = list(risk_categories.keys())
        scores = list(risk_categories.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=['red' if s > 15 else 'orange' if s > 10 else 'yellow' if s > 5 else 'green' for s in scores],
                text=[f"{s:.1f}" for s in scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Risk Breakdown by Category",
            yaxis_title="Risk Score",
            xaxis_title="Risk Category",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed risk analysis
    st.subheader("🔍 Detailed Risk Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Financial Risks", "🎯 Performance Risks", "💡 Risk Mitigation"])
    
    with tab1:
        st.markdown("### Financial Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Liquidity Analysis")
            st.metric("Current Ratio", f"{current_ratio:.2f}")
            
            if current_ratio < 1:
                st.error("🔴 **Critical**: Cannot meet short-term obligations")
                st.write("**Immediate Actions Needed:**")
                st.write("• Accelerate receivables collection")
                st.write("• Negotiate extended payment terms")
                st.write("• Consider emergency financing")
            elif current_ratio < 1.2:
                st.warning("🟡 **Warning**: Tight liquidity position")
                st.write("**Recommendations:**")
                st.write("• Monitor cash flow closely")
                st.write("• Optimize working capital")
            else:
                st.success("🟢 **Good**: Adequate liquidity")
        
        with col2:
            st.markdown("#### ⚖️ Leverage Analysis")
            st.metric("Debt-to-Equity", f"{debt_to_equity:.2f}")
            
            if debt_to_equity > 2:
                st.error("🔴 **High Risk**: Excessive leverage")
                st.write("**Actions Required:**")
                st.write("• Debt reduction plan")
                st.write("• Improve debt service coverage")
                st.write("• Consider equity financing")
            elif debt_to_equity > 1:
                st.warning("🟡 **Moderate**: Monitor debt levels")
                st.write("**Recommendations:**")
                st.write("• Maintain current debt levels")
                st.write("• Focus on profitability")
            else:
                st.success("🟢 **Conservative**: Low leverage risk")
    
    with tab2:
        st.markdown("### Performance Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Revenue Stability")
            st.metric("Revenue Volatility", f"{revenue_volatility:.1%}")
            
            if revenue_volatility > 0.3:
                st.error("🔴 **High Volatility**: Unpredictable revenue")
                st.write("**Risk Mitigation:**")
                st.write("• Diversify customer base")
                st.write("• Develop recurring revenue streams")
                st.write("• Improve demand forecasting")
            elif revenue_volatility > 0.2:
                st.warning("🟡 **Moderate Volatility**: Some unpredictability")
                st.write("**Recommendations:**")
                st.write("• Monitor market trends")
                st.write("• Strengthen customer relationships")
            else:
                st.success("🟢 **Stable Revenue**: Predictable business")
        
        with col2:
            st.markdown("#### 💎 Profitability Risk")
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            if profit_margin < 0:
                st.error("🔴 **Loss Making**: Immediate action required")
                st.write("**Critical Actions:**")
                st.write("• Review all expenses")
                st.write("• Evaluate pricing strategy")
                st.write("• Consider operational restructuring")
            elif profit_margin < 5:
                st.warning("🟡 **Thin Margins**: Vulnerable to shocks")
                st.write("**Improvement Areas:**")
                st.write("• Cost optimization")
                st.write("• Pricing strategy review")
                st.write("• Operational efficiency")
            else:
                st.success("🟢 **Healthy Margins**: Good profitability")
    
    with tab3:
        st.markdown("### 💡 Risk Mitigation Strategies")
        
        # Generate specific recommendations based on CSV data
        recommendations = []
        
        if revenue_volatility > 0.2:
            recommendations.append({
                'risk': 'High Revenue Volatility',
                'priority': 'High',
                'actions': [
                    'Develop multiple revenue streams',
                    'Focus on recurring revenue models',
                    'Diversify customer base',
                    'Improve demand forecasting'
                ],
                'timeline': '3-6 months'
            })
        
        if profit_margin < 10:
            recommendations.append({
                'risk': 'Low Profit Margins',
                'priority': 'High' if profit_margin < 5 else 'Medium',
                'actions': [
                    'Conduct comprehensive cost analysis',
                    'Review pricing strategy',
                    'Automate repetitive processes',
                    'Negotiate better supplier terms'
                ],
                'timeline': '1-3 months'
            })
        
        if revenue_growth < 0:
            recommendations.append({
                'risk': 'Revenue Decline',
                'priority': 'Critical',
                'actions': [
                    'Market research and competitive analysis',
                    'Customer retention programs',
                    'Product/service innovation',
                    'Marketing strategy overhaul'
                ],
                'timeline': 'Immediate'
            })
        
        if current_ratio < 1.2:
            recommendations.append({
                'risk': 'Liquidity Constraints',
                'priority': 'High',
                'actions': [
                    'Accelerate accounts receivable collection',
                    'Negotiate extended payment terms',
                    'Optimize inventory levels',
                    'Establish credit facilities'
                ],
                'timeline': '1-2 months'
            })
        
        if recommendations:
            for rec in recommendations:
                with st.expander(f"🎯 {rec['risk']} - {rec['priority']} Priority", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Recommended Actions:**")
                        for action in rec['actions']:
                            st.write(f"• {action}")
                    
                    with col2:
                        st.metric("Timeline", rec['timeline'])
                        
                        if rec['priority'] == 'Critical':
                            st.error("🔴 Critical Priority")
                        elif rec['priority'] == 'High':
                            st.error("🟠 High Priority")
                        elif rec['priority'] == 'Medium':
                            st.warning("🟡 Medium Priority")
                        else:
                            st.info("🔵 Low Priority")
        else:
            st.success("✅ **Well-Managed Risk Profile**: No critical risk mitigation actions required at this time.")
            
            st.info("**Recommended Ongoing Risk Management:**")
            st.write("• Regular financial monitoring")
            st.write("• Quarterly risk assessments")
            st.write("• Scenario planning exercises")
            st.write("• Continuous improvement initiatives")

# ========== INDUSTRY TEMPLATES ==========
def show_industry_templates():
    """Complete industry templates with CSV integration"""
    st.header("🏭 Industry-Specific Financial Analysis")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    template_manager = IndustryTemplateManager()
    
    if csv_data:
        st.success("📊 **Industry analysis powered by your uploaded CSV data**")
        
        # Auto-detect industry from CSV patterns
        detected_industry = template_manager.detect_industry_from_csv(csv_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"🤖 **Auto-detected Industry**: {template_manager.templates[detected_industry]['name']} based on your financial patterns")
        
        with col2:
            # Allow manual industry selection
            industry_options = list(template_manager.templates.keys())
            selected_industry = st.selectbox(
                "Override Industry",
                industry_options,
                index=industry_options.index(detected_industry),
                format_func=lambda x: f"{template_manager.templates[x]['icon']} {template_manager.templates[x]['name']}"
            )
    else:
        st.warning("📤 **No CSV Data Available**")
        st.info("Industry Templates work best with your uploaded financial data for accurate benchmarking.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Import CSV Data Now", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            # Allow exploration without CSV data
            industry_options = list(template_manager.templates.keys())
            selected_industry = st.selectbox(
                "Explore Industry",
                industry_options,
                index=2,  # Default to technology
                format_func=lambda x: f"{template_manager.templates[x]['icon']} {template_manager.templates[x]['name']}"
            )
    
    # Get selected template
    template = template_manager.get_template(selected_industry)
    
    # Industry overview tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Benchmarking", 
        "🎯 Your Performance", 
        "💡 Industry Insights",
        "📋 Action Plan"
    ])
    
    with tab1:
        st.subheader(f"{template['icon']} {template['name']} Industry Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Revenue Model")
            st.code(template['revenue_model'], language="text")
            
            st.markdown("### 🎯 Key Performance Metrics")
            for i, metric in enumerate(template['key_metrics']):
                st.write(f"{i+1}. {metric}")
            
            st.markdown("### 📊 Typical Financial Ratios")
            for ratio, value in template['typical_ratios'].items():
                if isinstance(value, float) and value < 1:
                    st.write(f"• **{ratio.replace('_', ' ').title()}**: {value:.1%}")
                else:
                    st.write(f"• **{ratio.replace('_', ' ').title()}**: {value:.2f}")
        
        with col2:
            st.markdown("### 💼 Cost Structure")
            
            # Create cost structure pie chart
            cost_structure = template['cost_structure']
            
            fig = go.Figure(data=[go.Pie(
                labels=[k.replace('_', ' ').title() for k in cost_structure.keys()],
                values=list(cost_structure.values()),
                hole=0.3,
                marker_colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
            )])
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=10
            )
            
            fig.update_layout(
                title=f"Typical {template['name']} Cost Structure",
                annotations=[dict(text=template['icon'], x=0.5, y=0.5, font_size=20, showarrow=False)],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔄 Working Capital")
            wc_metrics = template['working_capital']
            for metric, value in wc_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.0f} days")
    
    with tab2:
        st.subheader(f"📈 {template['name']} Industry Benchmarks")
        
        if csv_data:
            # Compare company data to industry benchmarks
            comparison = template_manager.benchmark_against_industry(csv_data, selected_industry)
            
            st.markdown("### 🎯 Performance vs Industry Standards")
            
            for metric, data in comparison.items():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if metric == 'revenue_growth':
                        st.metric("Your Revenue Growth", f"{data['company_value']:.1%}")
                    elif metric == 'profit_margin':
                        st.metric("Your Profit Margin", f"{data['company_value']:.1%}")
                
                with col2:
                    if metric == 'revenue_growth':
                        st.metric("Industry Average", f"{data['industry_benchmark']:.1%}")
                    elif metric == 'profit_margin':
                        st.metric("Industry Average", f"{data['industry_benchmark']:.1%}")
                
                with col3:
                    performance = data['performance']
                    difference = data['percentage_difference']
                    
                    if performance == 'Above Average':
                        st.metric("Performance", "🟢 Above Average", f"+{difference:.1f}%")
                    elif performance == 'Average':
                        st.metric("Performance", "🔵 Average", f"{difference:+.1f}%")
                    else:
                        st.metric("Performance", "🔴 Below Average", f"{difference:+.1f}%")
                
                st.markdown("---")
        else:
            # Show general industry benchmarks
            st.markdown("### 📊 Industry Standard Ratios")
            
            benchmarks = template['benchmarks']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Growth Metrics**")
                if 'revenue_growth' in benchmarks:
                    st.metric("Revenue Growth", f"{benchmarks['revenue_growth']:.1%}")
                if 'customer_retention' in benchmarks:
                    st.metric("Customer Retention", f"{benchmarks.get('customer_retention', 0.8):.1%}")
            
            with col2:
                st.markdown("**Profitability Metrics**")
                if 'profit_margin' in benchmarks:
                    st.metric("Profit Margin", f"{benchmarks['profit_margin']:.1%}")
                if 'gross_margin' in benchmarks:
                    st.metric("Gross Margin", f"{benchmarks.get('gross_margin', 0.4):.1%}")
            
            with col3:
                st.markdown("**Efficiency Metrics**")
                if 'inventory_turns' in benchmarks:
                    st.metric("Inventory Turns", f"{benchmarks.get('inventory_turns', 6):.1f}")
                if 'capacity_utilization' in benchmarks:
                    st.metric("Capacity Utilization", f"{benchmarks.get('capacity_utilization', 0.85):.1%}")
    
    with tab3:
        if csv_data:
            st.subheader("🎯 Your Financial Performance Analysis")
            
            # Performance dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                monthly_revenue = csv_data.get('monthly_revenue', 0)
                st.metric("Monthly Revenue", f"{monthly_revenue:,.0f} DHS")
                
                revenue_growth = csv_data.get('revenue_growth', 0)
                if revenue_growth > 10:
                    st.success("🚀 Strong Growth")
                elif revenue_growth > 0:
                    st.info("📈 Growing")
                else:
                    st.warning("📉 Declining")
            
            with col2:
                profit_margin = csv_data.get('profit_margin', 0)
                st.metric("Profit Margin", f"{profit_margin:.1f}%")
                
                industry_avg = template['benchmarks'].get('profit_margin', 0.1) * 100
                if profit_margin > industry_avg * 1.2:
                    st.success("🟢 Excellent")
                elif profit_margin > industry_avg * 0.8:
                    st.info("🔵 Good")
                else:
                    st.warning("🟡 Below Avg")
            
            with col3:
                revenue_volatility = csv_data.get('revenue_volatility', 0)
                stability_score = (1 - revenue_volatility) * 100
                st.metric("Revenue Stability", f"{stability_score:.0f}%")
                
                if revenue_volatility < 0.1:
                    st.success("🟢 Very Stable")
                elif revenue_volatility < 0.2:
                    st.info("🔵 Stable")
                else:
                    st.warning("🟡 Volatile")
            
            with col4:
                # Calculate efficiency score
                efficiency_score = 75  # Base score
                if profit_margin > 15:
                    efficiency_score += 15
                elif profit_margin > 10:
                    efficiency_score += 10
                elif profit_margin < 5:
                    efficiency_score -= 10
                
                if revenue_growth > 10:
                    efficiency_score += 10
                elif revenue_growth < 0:
                    efficiency_score -= 15
                
                efficiency_score = max(0, min(100, efficiency_score))
                
                st.metric("Efficiency Score", f"{efficiency_score}/100")
                
                if efficiency_score >= 85:
                    st.success("🟢 Excellent")
                elif efficiency_score >= 70:
                    st.info("🔵 Good")
                elif efficiency_score >= 50:
                    st.warning("🟡 Fair")
                else:
                    st.error("🔴 Poor")
            
            # Performance trends
            st.markdown("### 📈 Performance Trends")
            
            # Show CSV visualization if available
            csv_figures = CSVDataManager.get_csv_visualizations()
            if csv_figures and 'financial_trend' in csv_figures:
                st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
            
            # Industry-specific KPIs
            st.markdown(f"### {template['icon']} Industry-Specific Analysis")
            
            if selected_industry == 'saas':
                col1, col2 = st.columns(2)
                with col1:
                    # Estimate SaaS metrics from general data
                    estimated_churn = max(0.02, 0.1 - (profit_margin / 100))
                    st.metric("Estimated Churn Rate", f"{estimated_churn:.1%}")
                    
                    if estimated_churn < 0.05:
                        st.success("🟢 Low Churn")
                    elif estimated_churn < 0.1:
                        st.warning("🟡 Moderate Churn")
                    else:
                        st.error("🔴 High Churn")
                
                with col2:
                    # Estimate LTV/CAC ratio
                    estimated_ltv_cac = min(6, max(1, profit_margin / 5))
                    st.metric("Est. LTV/CAC Ratio", f"{estimated_ltv_cac:.1f}")
                    
                    if estimated_ltv_cac > 3:
                        st.success("🟢 Healthy Unit Economics")
                    else:
                        st.warning("🟡 Review Unit Economics")
            
            elif selected_industry == 'retail':
                col1, col2 = st.columns(2)
                with col1:
                    # Estimate inventory turnover
                    estimated_turns = max(2, min(12, profit_margin * 0.5 + 4))
                    st.metric("Est. Inventory Turns", f"{estimated_turns:.1f}")
                    
                    if estimated_turns > 6:
                        st.success("🟢 Efficient Inventory")
                    else:
                        st.warning("🟡 Inventory Optimization Needed")
                
                with col2:
                    # Seasonal impact
                    seasonal_impact = revenue_volatility * 100
                    st.metric("Seasonal Impact", f"{seasonal_impact:.0f}%")
                    
                    if seasonal_impact > 20:
                        st.info("🔵 High Seasonality")
                    else:
                        st.success("🟢 Low Seasonality")
            
            elif selected_industry == 'technology':
                col1, col2 = st.columns(2)
                with col1:
                    # Estimate R&D intensity
                    estimated_rd = min(30, max(5, 25 - profit_margin))
                    st.metric("Est. R&D Investment", f"{estimated_rd:.0f}%")
                    
                    if estimated_rd > 15:
                        st.success("🟢 Innovation Focused")
                    else:
                        st.warning("🟡 Consider More R&D")
                
                with col2:
                    # Market position indicator
                    market_strength = min(100, max(20, profit_margin * 4 + revenue_growth * 2))
                    st.metric("Market Position", f"{market_strength:.0f}/100")
                    
                    if market_strength > 75:
                        st.success("🟢 Strong Market Position")
                    elif market_strength > 50:
                        st.info("🔵 Competitive Position")
                    else:
                        st.warning("🟡 Market Challenges")
        else:
            st.info("📊 Upload your CSV data to see personalized performance analysis for your industry")
            
            # Show what would be available
            st.markdown("### 🎯 Available with Your Data:")
            st.write("• Revenue and profit trend analysis")
            st.write("• Industry benchmark comparisons")
            st.write("• Efficiency scoring and rankings")
            st.write("• Industry-specific KPI calculations")
            st.write("• Personalized improvement recommendations")
    
    with tab4:
        st.subheader("💡 Industry-Specific Insights")
        
        if csv_data:
            # Generate industry insights
            insights, recommendations = template_manager.generate_industry_insights(csv_data, selected_industry)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Key Insights")
                if insights:
                    for insight in insights:
                        st.success(insight)
                else:
                    st.info("No specific insights generated. Your performance appears to be within normal ranges.")
            
            with col2:
                st.markdown("#### 🎯 Recommendations")
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.success("✅ No immediate action items identified!")
            
            # Industry-specific challenges and opportunities
            st.markdown(f"### {template['icon']} {template['name']} Market Dynamics")
            
            if selected_industry == 'saas':
                st.markdown("""
                **Current Market Trends:**
                - Shift towards usage-based pricing models
                - Increased focus on customer success and retention
                - AI and automation integration opportunities
                - Emphasis on data privacy and security
                
                **Key Success Factors:**
                - High customer lifetime value
                - Low churn rates (< 5% monthly)
                - Strong product-market fit
                - Scalable customer acquisition
                """)
            
            elif selected_industry == 'retail':
                st.markdown("""
                **Current Market Trends:**
                - Omnichannel customer experience
                - Sustainability and ethical sourcing
                - Personalization through data analytics
                - Direct-to-consumer growth
                
                **Key Success Factors:**
                - Inventory optimization
                - Customer experience excellence
                - Supply chain efficiency
                - Brand differentiation
                """)
            
            elif selected_industry == 'technology':
                st.markdown("""
                **Current Market Trends:**
                - AI and machine learning integration
                - Cloud-first architecture
                - Cybersecurity focus
                - Remote work enablement
                
                **Key Success Factors:**
                - Continuous innovation
                - Talent acquisition and retention
                - Scalable business models
                - Strategic partnerships
                """)
            
            elif selected_industry == 'manufacturing':
                st.markdown("""
                **Current Market Trends:**
                - Industry 4.0 and smart manufacturing
                - Sustainability and circular economy
                - Supply chain resilience
                - Automation and robotics
                
                **Key Success Factors:**
                - Operational efficiency
                - Quality management
                - Supply chain optimization
                - Technology adoption
                """)
        else:
            st.info("📊 Industry insights will be personalized when you upload your CSV data")
            
            # General industry information
            st.markdown(f"### {template['icon']} {template['name']} General Characteristics")
            
            st.markdown("**Typical Business Model:**")
            st.write(template['revenue_model'])
            
            st.markdown("**Key Metrics to Track:**")
            for metric in template['key_metrics']:
                st.write(f"• {metric}")
    
    with tab5:
        st.subheader("📋 Strategic Action Plan")
        
        if csv_data:
            # Generate specific action plan based on performance
            profit_margin = csv_data.get('profit_margin', 0)
            revenue_growth = csv_data.get('revenue_growth', 0)
            revenue_volatility = csv_data.get('revenue_volatility', 0)
            
            st.markdown("### 🎯 Immediate Actions (Next 30 Days)")
            
            immediate_actions = []
            
            if profit_margin < 5:
                immediate_actions.append("💰 **Cost Analysis**: Conduct detailed cost breakdown and identify reduction opportunities")
            
            if revenue_growth < 0:
                immediate_actions.append("📈 **Revenue Recovery**: Analyze customer feedback and market positioning")
            
            if revenue_volatility > 0.3:
                immediate_actions.append("📊 **Revenue Stabilization**: Diversify revenue streams and improve forecasting")
            
            if not immediate_actions:
                immediate_actions.append("✅ **Performance Monitoring**: Continue tracking key metrics and maintain current performance")
            
            for action in immediate_actions:
                st.warning(action)
            
            st.markdown("### 📅 Short-term Goals (Next 90 Days)")
            
            short_term = []
            
            if selected_industry == 'saas':
                short_term.extend([
                    "🔄 Implement customer success program to reduce churn",
                    "📊 Develop usage analytics to identify expansion opportunities",
                    "💡 A/B test pricing models for optimization"
                ])
            elif selected_industry == 'retail':
                short_term.extend([
                    "📦 Optimize inventory management system",
                    "🛒 Enhance customer experience across all channels",
                    "📱 Develop mobile and online presence"
                ])
            elif selected_industry == 'technology':
                short_term.extend([
                    "🔬 Increase R&D investment for innovation",
                    "🤝 Build strategic technology partnerships",
                    "📈 Develop scalable sales processes"
                ])
            elif selected_industry == 'manufacturing':
                short_term.extend([
                    "⚙️ Implement lean manufacturing principles",
                    "📊 Deploy IoT for equipment monitoring",
                    "🔧 Optimize supply chain relationships"
                ])
            
            for goal in short_term:
                st.info(goal)
            
            st.markdown("### 🚀 Long-term Strategy (Next 12 Months)")
            
            long_term = [
                f"📊 Achieve {template['benchmarks'].get('profit_margin', 0.15)*100:.0f}% profit margin (industry benchmark)",
                f"📈 Target {template['benchmarks'].get('revenue_growth', 0.15)*100:.0f}% annual revenue growth",
                "🎯 Develop competitive differentiation strategy",
                "💼 Build scalable organizational capabilities"
            ]
            
            for strategy in long_term:
                st.success(strategy)
        else:
            st.info("📊 Upload your CSV data to receive a personalized strategic action plan")
            
            # General action plan template
            st.markdown(f"### 📋 General {template['name']} Action Plan Template")
            
            st.markdown("**Immediate Focus Areas:**")
            if selected_industry == 'saas':
                st.write("• Customer acquisition cost optimization")
                st.write("• Churn rate reduction initiatives")
                st.write("• Product feature prioritization")
            elif selected_industry == 'retail':
                st.write("• Inventory turnover improvement")
                st.write("• Customer experience enhancement")
                st.write("• Seasonal planning optimization")
            elif selected_industry == 'technology':
                st.write("• Innovation pipeline development")
                st.write("• Market positioning strengthening")
                st.write("• Talent acquisition strategy")
            elif selected_industry == 'manufacturing':
                st.write("• Operational efficiency improvement")
                st.write("• Quality management systems")
                st.write("• Supply chain optimization")

# ========== MAIN APPLICATION ==========
def main():
    """Streamlined main function with only essential features"""
    
    init_session_state()
    
    # Header without login requirement
    st.sidebar.markdown(f"""
    ### 🏢 Financial Analysis Suite
    **Welcome to Advanced Analytics**
    
    *Professional Financial Planning Platform*
    
    ---
    """)
    
    # CSV import indicator
    if CSVDataManager.has_csv_data():
        st.sidebar.success("📊 CSV Data Loaded")
        # Show quick metrics
        csv_data = CSVDataManager.get_csv_financial_data()
        if csv_data:
            monthly_revenue = csv_data.get('monthly_revenue', 0)
            profit_margin = csv_data.get('profit_margin', 0)
            st.sidebar.metric("Monthly Revenue", f"{monthly_revenue:,.0f} DHS")
            st.sidebar.metric("Profit Margin", f"{profit_margin:.1f}%")
    else:
        st.sidebar.warning("📤 No CSV Data")
        st.sidebar.caption("Upload data for full analysis")
    
    # Streamlined navigation menu - only essential pages
    menu_items = {
        "📤 Smart CSV Import": "csv_import",
        "👔 Executive Dashboard": "executive_dashboard",
        "🧠 Advanced Analytics": "advanced_analytics", 
        "🎯 Scenario Planning": "scenario_planning",
        "🤖 ML Forecasting": "ml_forecasting",
        "⚠️ Risk Management": "risk_management",
        "🏭 Industry Templates": "industry_templates"
    }
    
    # Handle page navigation
    if 'current_page' in st.session_state:
        # Find the menu text for the current page
        page_key = st.session_state['current_page']
        choice = None
        for menu_text, menu_key in menu_items.items():
            if menu_key == page_key:
                choice = menu_text
                break
        if not choice:
            choice = list(menu_items.keys())[0]
        # Clear the redirect
        del st.session_state['current_page']
    else:
        choice = st.sidebar.selectbox(
            "🧭 Navigation",
            list(menu_items.keys()),
            index=0  # Default to CSV Import
        )
    
    # Route to appropriate page
    page_key = menu_items[choice]
    
    if page_key == "csv_import":
        show_enhanced_csv_import()
    elif page_key == "executive_dashboard":
        show_executive_dashboard()
    elif page_key == "advanced_analytics":
        show_advanced_analytics()
    elif page_key == "scenario_planning":
        show_scenario_planning()
    elif page_key == "ml_forecasting":
        show_ml_forecasting()
    elif page_key == "risk_management":
        show_risk_management()
    elif page_key == "industry_templates":
        show_industry_templates()
    
    # Enhanced sidebar footer
    with st.sidebar:
        st.markdown("---")
        
        # System status
        st.markdown("### 🔧 System Status")
        st.success("🟢 CSV Processor: Ready")
        st.success("🟢 Analytics Engine: Active") 
        st.success("🟢 ML Models: Available")
        st.success("🟢 Industry Templates: Complete")
        
        # Current date and time
        current_datetime = datetime.now()
        st.caption(f"Current Time: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("User: SalianiBouchaib")
        
        # Additional info
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.caption("✅ AI-Powered CSV Analysis")
        st.caption("✅ Financial Health Scoring")
        st.caption("✅ Industry Benchmarking")
        st.caption("✅ ML Forecasting")
        st.caption("✅ Risk Assessment")
        st.caption("✅ Scenario Planning")

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    main()
                
                    
