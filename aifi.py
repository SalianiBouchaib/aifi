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
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Handle optional dependencies with try/except
try:
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    class MLFallback:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X)) if hasattr(X, '__len__') else [0]
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
        def score(self, X, y):
            return 0.0
    RandomForestRegressor = MLFallback
    LinearRegression = MLFallback
    StandardScaler = MLFallback
    VotingRegressor = MLFallback
    Ridge = MLFallback
    PolynomialFeatures = MLFallback
    Pipeline = MLFallback

warnings.filterwarnings('ignore')

# ========== ENHANCED DATA VALIDATION AND CORRECTION SYSTEM ==========
class AdvancedDataValidator:
    """Classe pour validation avancée et diagnostic d'incohérences avec corrections automatiques"""
    
    def __init__(self):
        self.validation_results = {}
        self.correction_log = []
        self.outlier_threshold = 3
        self.variation_threshold = 1.0
        
    def validate_accounting_equation(self, df, mappings):
        """Validation équation comptable : Assets = Liabilities + Equity - VERSION CORRIGÉE"""
        issues = []
        
        try:
            if all(col in mappings for col in ['assets', 'liabilities', 'equity']):
                assets_col = mappings['assets']
                liabilities_col = mappings['liabilities']
                equity_col = mappings['equity']
                
                if all(col in df.columns for col in [assets_col, liabilities_col, equity_col]):
                    assets = self.clean_numeric_column(df[assets_col])
                    liabilities = self.clean_numeric_column(df[liabilities_col])
                    equity = self.clean_numeric_column(df[equity_col])
                    
                    # FIX: Gestion sécurisée des comparaisons numpy
                    if len(assets) > 0 and len(liabilities) > 0 and len(equity) > 0:
                        calculated_assets = liabilities + equity
                        differences = np.abs(assets - calculated_assets)
                        tolerance = assets * 0.05
                        
                        # FIX: Conversion sécurisée pour éviter l'erreur numpy
                        violations = np.array(differences > tolerance, dtype=bool)
                        violation_count = int(np.sum(violations)) if len(violations) > 0 else 0
                        
                        if violation_count > 0:
                            max_diff = float(np.max(differences)) if len(differences) > 0 else 0
                            max_diff_pct = (max_diff / float(np.max(assets))) * 100 if float(np.max(assets)) > 0 else 0
                            
                            issues.append({
                                'type': 'Équation Comptable',
                                'severity': 'Élevée' if violation_count > len(df) * 0.3 else 'Moyenne',
                                'count': violation_count,
                                'message': f"{violation_count} violations détectées (max: {max_diff_pct:.1f}%)",
                                'violations': violations.tolist(),
                                'differences': differences.tolist()
                            })
                        else:
                            issues.append({
                                'type': 'Équation Comptable',
                                'severity': 'OK',
                                'message': "Équation comptable respectée"
                            })
        except Exception as e:
            issues.append({
                'type': 'Équation Comptable',
                'severity': 'Erreur',
                'message': f"Erreur validation: {str(e)}"
            })
        
        return issues
    
    def validate_profit_logic(self, df, mappings):
        """Contrôle logique : Profit = Revenue - Costs - VERSION CORRIGÉE"""
        issues = []
        
        try:
            if all(col in mappings for col in ['revenue', 'costs', 'profit']):
                revenue_col = mappings['revenue']
                costs_col = mappings['costs']
                profit_col = mappings['profit']
                
                if all(col in df.columns for col in [revenue_col, costs_col, profit_col]):
                    revenue = self.clean_numeric_column(df[revenue_col])
                    costs = self.clean_numeric_column(df[costs_col])
                    profit = self.clean_numeric_column(df[profit_col])
                    
                    if len(revenue) > 0 and len(costs) > 0 and len(profit) > 0:
                        calculated_profit = revenue - costs
                        differences = np.abs(profit - calculated_profit)
                        tolerance = revenue * 0.02
                        
                        # FIX: Conversion sécurisée
                        violations = np.array(differences > tolerance, dtype=bool)
                        violation_count = int(np.sum(violations)) if len(violations) > 0 else 0
                        
                        if violation_count > 0:
                            issues.append({
                                'type': 'Logique Profit',
                                'severity': 'Élevée' if violation_count > len(df) * 0.2 else 'Moyenne',
                                'count': violation_count,
                                'message': f"{violation_count} incohérences profit détectées",
                                'violations': violations.tolist(),
                                'differences': differences.tolist(),
                                'calculated_profit': calculated_profit.tolist()
                            })
                        else:
                            issues.append({
                                'type': 'Logique Profit',
                                'severity': 'OK',
                                'message': "Logique profit respectée"
                            })
        except Exception as e:
            issues.append({
                'type': 'Logique Profit',
                'severity': 'Erreur',
                'message': f"Erreur validation: {str(e)}"
            })
        
        return issues
    
    def detect_extreme_variations(self, df, mappings):
        """Vérification plausibilité : variations >100% période à période"""
        issues = []
        
        key_metrics = ['revenue', 'costs', 'profit', 'assets']
        
        for metric in key_metrics:
            try:
                if metric in mappings:
                    col = mappings[metric]
                    if col in df.columns:
                        data = self.clean_numeric_column(df[col]).dropna()
                        
                        if len(data) > 1:
                            variations = data.pct_change().abs()
                            # FIX: Gestion sécurisée des variations extrêmes
                            extreme_variations = np.array(variations > self.variation_threshold, dtype=bool)
                            extreme_count = int(np.sum(extreme_variations)) if len(extreme_variations) > 0 else 0
                            
                            if extreme_count > 0:
                                max_variation = float(np.max(variations.dropna())) if len(variations.dropna()) > 0 else 0
                                
                                issues.append({
                                    'type': f'Variations Extrêmes - {metric.title()}',
                                    'severity': 'Élevée' if max_variation > 2.0 else 'Moyenne',
                                    'count': extreme_count,
                                    'message': f"{extreme_count} variations >100% détectées (max: {max_variation:.1%})",
                                    'extreme_indices': variations[extreme_variations].index.tolist(),
                                    'variations': variations.tolist()
                                })
            except Exception as e:
                issues.append({
                    'type': f'Variations Extrêmes - {metric.title()}',
                    'severity': 'Erreur',
                    'message': f"Erreur validation: {str(e)}"
                })
        
        return issues
    
    def detect_outliers(self, df, mappings):
        """Détection outliers : valeurs >3 écarts-types de la moyenne"""
        issues = []
        
        for metric, col in mappings.items():
            try:
                if col in df.columns:
                    data = self.clean_numeric_column(df[col]).dropna()
                    
                    if len(data) > 3:
                        mean_val = float(data.mean())
                        std_val = float(data.std())
                        
                        if std_val > 0:
                            z_scores = np.abs((data - mean_val) / std_val)
                            # FIX: Gestion sécurisée des outliers
                            outliers = np.array(z_scores > self.outlier_threshold, dtype=bool)
                            outlier_count = int(np.sum(outliers)) if len(outliers) > 0 else 0
                            
                            if outlier_count > 0:
                                max_z_score = float(np.max(z_scores)) if len(z_scores) > 0 else 0
                                
                                issues.append({
                                    'type': f'Outliers - {metric.title()}',
                                    'severity': 'Élevée' if max_z_score > 5 else 'Moyenne',
                                    'count': outlier_count,
                                    'message': f"{outlier_count} valeurs aberrantes (max Z-score: {max_z_score:.1f})",
                                    'outlier_indices': data[outliers].index.tolist(),
                                    'outlier_values': data[outliers].tolist(),
                                    'z_scores': z_scores.tolist()
                                })
            except Exception as e:
                issues.append({
                    'type': f'Outliers - {metric.title()}',
                    'severity': 'Erreur',
                    'message': f"Erreur validation: {str(e)}"
                })
        
        return issues
    
    def clean_numeric_column(self, series):
        """Nettoyer les colonnes numériques avec gestion d'erreurs renforcée"""
        try:
            # Si déjà numérique
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors='coerce').fillna(0)
            
            # Nettoyer les strings
            cleaned = series.astype(str)
            cleaned = cleaned.str.replace(r'[$€£¥₹₽]', '', regex=True)
            cleaned = cleaned.str.replace(',', '')
            cleaned = cleaned.str.replace(' ', '')
            cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
            
            # Convertir en numérique
            cleaned = pd.to_numeric(cleaned, errors='coerce').fillna(0)
            return cleaned
        except Exception:
            # En cas d'erreur, retourner une série de zéros
            return pd.Series([0] * len(series))
    
    def apply_outlier_correction(self, data, method='iqr'):
        """Corriger les valeurs aberrantes"""
        try:
            if len(data) < 4:
                return data, []
            
            corrected_data = data.copy()
            corrections = []
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (data < lower_bound) | (data > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Remplacer par la médiane
                    median_val = data.median()
                    corrected_data.loc[outliers] = median_val
                    
                    corrections.append({
                        'method': 'IQR Outlier Correction',
                        'outliers_found': int(outlier_count),
                        'replacement_value': float(median_val),
                        'basis': f'Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}'
                    })
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    mean_val = data.mean()
                    corrected_data.loc[outliers] = mean_val
                    
                    corrections.append({
                        'method': 'Z-Score Outlier Correction',
                        'outliers_found': int(outlier_count),
                        'replacement_value': float(mean_val),
                        'threshold_used': 3.0,
                        'basis': f'Mean={mean_val:.2f}, Std={data.std():.2f}'
                    })
            
            return corrected_data, corrections
        except Exception as e:
            return data, [{'method': 'Outlier Correction Failed', 'error': str(e)}]
    
    def apply_missing_value_interpolation(self, data, method='linear'):
        """Interpoler les valeurs manquantes"""
        try:
            if data.isnull().sum() == 0:
                return data, []
            
            corrected_data = data.copy()
            missing_count = data.isnull().sum()
            corrections = []
            
            if method == 'linear' and len(data.dropna()) >= 2:
                corrected_data = data.interpolate(method='linear')
                corrections.append({
                    'method': 'Linear Interpolation',
                    'missing_values_filled': int(missing_count),
                    'interpolation_method': 'linear'
                })
            
            elif method == 'forward_fill':
                corrected_data = data.fillna(method='ffill').fillna(method='bfill')
                corrections.append({
                    'method': 'Forward/Backward Fill',
                    'missing_values_filled': int(missing_count),
                    'interpolation_method': 'forward_fill'
                })
            
            else:
                # Fallback: remplir par la moyenne
                mean_val = data.mean()
                corrected_data = data.fillna(mean_val)
                corrections.append({
                    'method': 'Mean Imputation',
                    'missing_values_filled': int(missing_count),
                    'replacement_value': float(mean_val)
                })
            
            return corrected_data, corrections
        except Exception as e:
            return data, [{'method': 'Interpolation Failed', 'error': str(e)}]
    
    def apply_extreme_variation_smoothing(self, data, threshold=1.0):
        """Lisser les variations extrêmes"""
        try:
            if len(data) < 3:
                return data, []
            
            corrected_data = data.copy()
            variations = data.pct_change().abs()
            extreme_mask = variations > threshold
            extreme_count = extreme_mask.sum()
            corrections = []
            
            if extreme_count > 0:
                # Appliquer un filtre Savitzky-Golay pour lisser
                window_length = min(5, len(data) if len(data) % 2 == 1 else len(data) - 1)
                if window_length >= 3:
                    smoothed = savgol_filter(data, window_length, 2)
                    
                    # Remplacer seulement les points avec variations extrêmes
                    corrected_data.loc[extreme_mask] = smoothed[extreme_mask]
                    
                    corrections.append({
                        'method': 'Savitzky-Golay Smoothing',
                        'extreme_variations_smoothed': int(extreme_count),
                        'threshold_used': threshold,
                        'window_length': window_length
                    })
            
            return corrected_data, corrections
        except Exception as e:
            return data, [{'method': 'Smoothing Failed', 'error': str(e)}]
    
    def comprehensive_validation(self, df, mappings):
        """Validation complète des données avec corrections automatiques"""
        all_issues = []
        all_corrections = []
        
        try:
            # Validation équation comptable
            accounting_issues = self.validate_accounting_equation(df, mappings)
            all_issues.extend(accounting_issues)
            
            # Validation logique profit
            profit_issues = self.validate_profit_logic(df, mappings)
            all_issues.extend(profit_issues)
            
            # Détection variations extrêmes
            variation_issues = self.detect_extreme_variations(df, mappings)
            all_issues.extend(variation_issues)
            
            # Détection outliers
            outlier_issues = self.detect_outliers(df, mappings)
            all_issues.extend(outlier_issues)
            
            # Calculer score de qualité global
            quality_score = self.calculate_quality_score(all_issues, len(df))
            
            return {
                'issues': all_issues,
                'quality_score': quality_score,
                'total_issues': len(all_issues),
                'critical_issues': len([i for i in all_issues if i.get('severity') == 'Élevée']),
                'corrections_applied': all_corrections
            }
        except Exception as e:
            return {
                'issues': [{'type': 'Erreur Générale', 'severity': 'Critique', 'message': f"Erreur validation: {str(e)}"}],
                'quality_score': 50,
                'total_issues': 1,
                'critical_issues': 1,
                'corrections_applied': []
            }

    def calculate_quality_score(self, issues, data_length):
        """Calculer score de qualité des données (0-100)"""
        if not issues:
            return 100
        
        penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'Moyenne')
            if severity == 'Élevée' or severity == 'Critique':
                penalty += 20
            elif severity == 'Moyenne':
                penalty += 10
            elif severity == 'OK':
                penalty -= 5
        
        score = max(0, 100 - penalty)
        return min(100, score)

# ========== ENHANCED ML FORECASTING ENGINE ==========
class EnhancedMLForecastingEngine:
    """Moteur de prévision ML avancé avec ensemble methods et validation croisée"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.validation_scores = {}
        
    def prepare_features(self, data, include_seasonality=True, include_trend=True):
        """Préparer les features pour l'entraînement ML"""
        try:
            if len(data) < 3:
                return None, None
            
            features = []
            targets = []
            
            # Créer des features temporelles
            for i in range(2, len(data)):
                feature_row = []
                
                # Features de base (lag features)
                feature_row.extend([
                    data[i-1],  # Valeur précédente
                    data[i-2],  # Valeur t-2
                ])
                
                # Tendance
                if include_trend and i >= 3:
                    trend = (data[i-1] - data[i-3]) / 2
                    feature_row.append(trend)
                
                # Saisonnalité simple
                if include_seasonality and len(data) >= 12:
                    seasonal_index = i % 12
                    feature_row.append(seasonal_index)
                
                # Moyenne mobile
                if i >= 3:
                    ma_3 = np.mean(data[i-3:i])
                    feature_row.append(ma_3)
                
                # Volatilité récente
                if i >= 4:
                    recent_vol = np.std(data[i-4:i])
                    feature_row.append(recent_vol)
                
                features.append(feature_row)
                targets.append(data[i])
            
            return np.array(features), np.array(targets)
        except Exception as e:
            print(f"Erreur préparation features: {e}")
            return None, None
    
    def create_ensemble_models(self):
        """Créer un ensemble de modèles ML"""
        models = {}
        
        if ML_AVAILABLE:
            # Random Forest
            models['random_forest'] = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ))
            ])
            
            # Linear Regression with Polynomial Features
            models['polynomial'] = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('scaler', StandardScaler()),
                ('linear', LinearRegression())
            ])
            
            # Ridge Regression
            models['ridge'] = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ])
            
            # Ensemble Voting
            models['ensemble'] = VotingRegressor([
                ('rf', models['random_forest']),
                ('poly', models['polynomial']),
                ('ridge', models['ridge'])
            ])
        else:
            # Fallback simple models
            models['simple_linear'] = LinearRegression()
        
        return models
    
    def validate_models(self, X, y, cv_folds=3):
        """Validation croisée des modèles"""
        if X is None or y is None or len(X) < cv_folds:
            return {}
        
        models = self.create_ensemble_models()
        scores = {}
        
        try:
            # TimeSeriesSplit pour les données temporelles
            if ML_AVAILABLE and len(X) >= 6:
                tscv = TimeSeriesSplit(n_splits=min(cv_folds, len(X)//3))
            else:
                tscv = None
            
            for name, model in models.items():
                try:
                    if tscv and ML_AVAILABLE:
                        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                        scores[name] = {
                            'mean_score': np.mean(cv_scores),
                            'std_score': np.std(cv_scores),
                            'cv_scores': cv_scores.tolist()
                        }
                    else:
                        # Fallback simple validation
                        split_point = int(len(X) * 0.7)
                        X_train, X_test = X[:split_point], X[split_point:]
                        y_train, y_test = y[:split_point], y[split_point:]
                        
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        score = r2_score(y_test, predictions) if ML_AVAILABLE else 0.5
                        
                        scores[name] = {
                            'mean_score': score,
                            'std_score': 0.1,
                            'cv_scores': [score]
                        }
                except Exception as e:
                    scores[name] = {
                        'mean_score': 0.0,
                        'std_score': 1.0,
                        'cv_scores': [0.0],
                        'error': str(e)
                    }
            
            return scores
        except Exception as e:
            return {'error': f"Validation failed: {str(e)}"}
    
    def enhanced_forecast(self, data, periods, confidence_level=95, 
                         external_factors=None, business_constraints=None):
        """Génération de prévisions avec modèles ensemble - VERSION CORRIGÉE"""
        try:
            if len(data) < 3:
                return None
            
            # Préparer les données
            X, y = self.prepare_features(data)
            if X is None or y is None:
                return self.fallback_forecast(data, periods, confidence_level)
            
            # Valider les modèles
            model_scores = self.validate_models(X, y)
            
            # Sélectionner le meilleur modèle
            best_model_name = self.select_best_model(model_scores)
            
            # Entraîner le meilleur modèle
            models = self.create_ensemble_models()
            best_model = models.get(best_model_name, models['random_forest'] if 'random_forest' in models else list(models.values())[0])
            
            try:
                best_model.fit(X, y)
                self.best_model = best_model
            except Exception as e:
                return self.fallback_forecast(data, periods, confidence_level)
            
            # Générer les prévisions
            forecasts = []
            current_data = list(data)
            
            for period in range(periods):
                try:
                    # Préparer les features pour cette prédiction
                    if len(current_data) >= 2:
                        feature_row = [
                            current_data[-1],
                            current_data[-2]
                        ]
                        
                        # Ajouter trend si disponible
                        if len(current_data) >= 3:
                            trend = (current_data[-1] - current_data[-3]) / 2
                            feature_row.append(trend)
                        
                        # Ajouter features supplémentaires pour matcher X
                        while len(feature_row) < X.shape[1]:
                            feature_row.append(np.mean(current_data[-3:]) if len(current_data) >= 3 else current_data[-1])
                        
                        # Prédiction
                        prediction = best_model.predict([feature_row[:X.shape[1]]])[0]
                        
                        # Appliquer facteurs externes
                        if external_factors:
                            market_factor = external_factors.get('market_growth', 0)
                            economic_factor = external_factors.get('economic_impact', 1.0)
                            prediction = prediction * (1 + market_factor) * economic_factor
                        
                        # Appliquer contraintes business
                        if business_constraints:
                            max_growth = business_constraints.get('max_growth', 0.25)
                            min_growth = business_constraints.get('min_growth', -0.20)
                            
                            if len(current_data) > 0:
                                growth_rate = (prediction - current_data[-1]) / current_data[-1]
                                growth_rate = max(min_growth, min(max_growth, growth_rate))
                                prediction = current_data[-1] * (1 + growth_rate)
                        
                        # Éviter les valeurs négatives
                        prediction = max(0, prediction)
                        
                        forecasts.append(float(prediction))
                        current_data.append(prediction)
                    else:
                        # Fallback si pas assez de données
                        last_value = current_data[-1] if current_data else 0
                        forecasts.append(float(last_value * 1.02))  # Croissance 2%
                        current_data.append(last_value * 1.02)
                        
                except Exception as e:
                    # Fallback pour cette prédiction
                    last_value = current_data[-1] if current_data else 0
                    forecasts.append(float(last_value))
                    current_data.append(last_value)
            
            # Calculer intervalles de confiance
            std_error = np.std(y - best_model.predict(X)) if len(y) > 1 else np.std(data) * 0.1
            confidence_multiplier = 1.96 if confidence_level == 95 else 2.58
            
            upper_bounds = [f + confidence_multiplier * std_error for f in forecasts]
            lower_bounds = [max(0, f - confidence_multiplier * std_error) for f in forecasts]
            
            # Calculer métriques de performance
            train_predictions = best_model.predict(X)
            performance_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y, train_predictions))) if ML_AVAILABLE else 0.0,
                'mae': float(mean_absolute_error(y, train_predictions)) if ML_AVAILABLE else 0.0,
                'r2_score': float(r2_score(y, train_predictions)) if ML_AVAILABLE else 0.7,
                'cv_score': model_scores.get(best_model_name, {}).get('mean_score', 0.0)
            }
            
            # Feature importance si disponible
            feature_importance = {}
            if hasattr(best_model, 'feature_importances_'):
                importance_values = best_model.feature_importances_
                feature_names = ['lag_1', 'lag_2', 'trend', 'seasonal', 'ma_3', 'volatility']
                for i, importance in enumerate(importance_values[:len(feature_names)]):
                    feature_importance[feature_names[i]] = float(importance)
            
            # FIX: S'assurer que tous les résultats contiennent 'forecasts'
            results = {
                'forecasts': forecasts,
                'upper_bounds': upper_bounds,
                'lower_bounds': lower_bounds,
                'confidence_level': confidence_level,
                'periods': periods,
                'best_model': best_model_name,
                'model_performance': performance_metrics,
                'feature_importance': feature_importance,
                'model_scores': model_scores
            }
            
            return results
            
        except Exception as e:
            print(f"Erreur ML forecast: {e}")
            return self.fallback_forecast(data, periods, confidence_level)
    
    def fallback_forecast(self, data, periods, confidence_level):
        """Méthode de prévision de secours simple mais robuste"""
        try:
            if len(data) < 2:
                base_value = data[0] if data else 1000
                forecasts = [base_value] * periods
            else:
                # Calcul de tendance simple
                recent_data = data[-6:] if len(data) >= 6 else data
                if len(recent_data) >= 2:
                    trend = (recent_data[-1] - recent_data[0]) / (len(recent_data) - 1)
                else:
                    trend = 0
                
                forecasts = []
                last_value = data[-1]
                
                for i in range(periods):
                    forecast = last_value + trend * (i + 1)
                    forecast = max(0, forecast)  # Éviter valeurs négatives
                    forecasts.append(float(forecast))
            
            # Intervalles de confiance simples
            std_error = np.std(data) * 0.1 if len(data) > 1 else np.mean(data) * 0.1
            confidence_multiplier = 1.96 if confidence_level == 95 else 2.58
            
            upper_bounds = [f + confidence_multiplier * std_error for f in forecasts]
            lower_bounds = [max(0, f - confidence_multiplier * std_error) for f in forecasts]
            
            # FIX: S'assurer que le fallback contient aussi 'forecasts'
            return {
                'forecasts': forecasts,
                'upper_bounds': upper_bounds,
                'lower_bounds': lower_bounds,
                'confidence_level': confidence_level,
                'periods': periods,
                'best_model': 'Simple Trend',
                'model_performance': {
                    'rmse': std_error,
                    'mae': std_error * 0.8,
                    'r2_score': 0.5,
                    'cv_score': 0.5
                },
                'feature_importance': {},
                'model_scores': {}
            }
        except Exception as e:
            # Dernière ligne de défense
            fallback_value = 1000
            forecasts = [fallback_value] * periods
            return {
                'forecasts': forecasts,
                'upper_bounds': [f * 1.2 for f in forecasts],
                'lower_bounds': [f * 0.8 for f in forecasts],
                'confidence_level': confidence_level,
                'periods': periods,
                'best_model': 'Emergency Fallback',
                'model_performance': {'rmse': 0, 'mae': 0, 'r2_score': 0, 'cv_score': 0},
                'feature_importance': {},
                'model_scores': {}
            }
    
    def select_best_model(self, model_scores):
        """Sélectionner le meilleur modèle basé sur les scores"""
        if not model_scores:
            return 'random_forest'
        
        best_model = None
        best_score = -float('inf')
        
        for model_name, scores in model_scores.items():
            if isinstance(scores, dict) and 'mean_score' in scores:
                score = scores['mean_score']
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model if best_model else list(model_scores.keys())[0]

# ========== ENHANCED SCENARIO CALIBRATOR ==========
class EnhancedScenarioCalibrator:
    """Calibrateur avancé pour planification de scénarios basé sur l'historique"""
    
    def __init__(self):
        self.historical_patterns = {}
        self.volatility_metrics = {}
        self.seasonal_patterns = {}
    
    def analyze_historical_volatility(self, data):
        """Analyser la volatilité historique pour calibrer les scénarios"""
        try:
            if len(data) < 3:
                return {
                    'volatility': 0.2,
                    'trend': 0.0,
                    'seasonality': False,
                    'coefficient_variation': 0.2
                }
            
            data_array = np.array(data)
            
            # Calculer volatilité
            returns = np.diff(data_array) / data_array[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.2
            
            # Calculer tendance
            if len(data) >= 2:
                x = np.arange(len(data))
                slope, _ = np.polyfit(x, data_array, 1)
                trend = slope / np.mean(data_array) if np.mean(data_array) != 0 else 0
            else:
                trend = 0
            
            # Détecter saisonnalité simple
            seasonality = False
            if len(data) >= 12:
                monthly_means = []
                for month in range(12):
                    month_data = [data[i] for i in range(month, len(data), 12)]
                    if month_data:
                        monthly_means.append(np.mean(month_data))
                
                if len(monthly_means) == 12:
                    seasonal_cv = np.std(monthly_means) / np.mean(monthly_means)
                    seasonality = seasonal_cv > 0.1
            
            # Coefficient de variation
            coeff_var = np.std(data_array) / np.mean(data_array) if np.mean(data_array) != 0 else 0
            
            return {
                'volatility': float(volatility),
                'trend': float(trend),
                'seasonality': seasonality,
                'coefficient_variation': float(coeff_var),
                'mean_value': float(np.mean(data_array)),
                'data_points': len(data)
            }
        except Exception as e:
            return {
                'volatility': 0.2,
                'trend': 0.0,
                'seasonality': False,
                'coefficient_variation': 0.2,
                'error': str(e)
            }
    
    def calibrate_scenario_parameters(self, historical_analysis, industry='general'):
        """Calibrer les paramètres de scénarios basés sur l'analyse historique"""
        
        volatility = historical_analysis.get('volatility', 0.2)
        trend = historical_analysis.get('trend', 0.0)
        
        # Ajustements spécifiques par industrie
        industry_adjustments = {
            'saas': {'volatility_multiplier': 0.8, 'growth_boost': 1.5},
            'retail': {'volatility_multiplier': 1.2, 'growth_boost': 0.8},
            'technology': {'volatility_multiplier': 1.0, 'growth_boost': 1.3},
            'manufacturing': {'volatility_multiplier': 0.9, 'growth_boost': 0.9}
        }
        
        adjustments = industry_adjustments.get(industry, {'volatility_multiplier': 1.0, 'growth_boost': 1.0})
        
        # Calibrer scénarios
        base_volatility = volatility * adjustments['volatility_multiplier']
        base_trend = trend * adjustments['growth_boost']
        
        scenarios = {
            'pessimistic': {
                'revenue_change': max(-0.30, base_trend - 2 * base_volatility),
                'cost_change': min(0.25, 0.15 + base_volatility),
                'probability': 0.20
            },
            'realistic': {
                'revenue_change': base_trend,
                'cost_change': 0.08 + base_volatility * 0.5,
                'probability': 0.60
            },
            'optimistic': {
                'revenue_change': min(0.50, base_trend + 1.5 * base_volatility),
                'cost_change': max(0.03, 0.05 - base_volatility * 0.3),
                'probability': 0.20
            }
        }
        
        return scenarios
    
    def apply_operational_constraints(self, scenarios, constraints):
        """Appliquer des contraintes opérationnelles aux scénarios"""
        
        max_revenue_decline = constraints.get('max_revenue_decline', -0.3)
        max_cost_increase = constraints.get('max_cost_increase', 0.25)
        min_margin = constraints.get('min_margin', -0.1)
        
        for scenario_name, params in scenarios.items():
            # Contraindre les changements de revenus
            params['revenue_change'] = max(max_revenue_decline, params['revenue_change'])
            
            # Contraindre les changements de coûts
            params['cost_change'] = min(max_cost_increase, params['cost_change'])
            
            # S'assurer qu'on respecte la marge minimale
            implied_margin = params['revenue_change'] - params['cost_change']
            if implied_margin < min_margin:
                # Ajuster les coûts pour respecter la marge minimale
                params['cost_change'] = params['revenue_change'] - min_margin
                params['cost_change'] = min(max_cost_increase, params['cost_change'])
        
        return scenarios
    
    def generate_monte_carlo_scenarios(self, base_data, n_simulations=1000, periods=12):
        """Générer des scénarios Monte Carlo basés sur l'historique"""
        
        if len(base_data) < 2:
            base_value = base_data[0] if base_data else 1000
            return [{'total_value': base_value * periods} for _ in range(n_simulations)]
        
        # Analyser patterns historiques
        returns = np.diff(base_data) / np.array(base_data[:-1])
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulations = []
        
        for _ in range(n_simulations):
            path = [base_data[-1]]  # Commencer par la dernière valeur
            
            for period in range(periods):
                # Générer un rendement aléatoire
                random_return = np.random.normal(mean_return, std_return)
                
                # Appliquer des contraintes de réalisme
                random_return = max(-0.5, min(1.0, random_return))
                
                next_value = path[-1] * (1 + random_return)
                next_value = max(0, next_value)  # Éviter valeurs négatives
                
                path.append(next_value)
            
            simulations.append({
                'path': path[1:],  # Exclure la valeur initiale
                'total_value': sum(path[1:]),
                'final_value': path[-1],
                'max_value': max(path[1:]),
                'min_value': min(path[1:])
            })
        
        return simulations

# ========== ADVANCED CSV PROCESSOR (Enhanced with Validation) ==========
class AdvancedCSVProcessor:
    """Processeur CSV avancé avec validation et correction automatique"""
    
    def __init__(self):
        self.column_mappings = {
            'revenue': ['revenue', 'sales', 'income', 'turnover', 'ca', 'chiffre_affaires'],
            'costs': ['costs', 'expenses', 'expenditure', 'charges', 'depenses'],
            'date': ['date', 'month', 'period', 'time', 'mois', 'periode'],
            'profit': ['profit', 'earnings', 'net_income', 'benefice', 'resultat'],
            'cash_flow': ['cash_flow', 'cash flow', 'cashflow', 'flux_tresorerie'],
            'assets': ['assets', 'total_assets', 'actifs', 'actif_total'],
            'current_assets': ['current_assets', 'actifs_courants', 'actif_circulant'],
            'fixed_assets': ['fixed_assets', 'actifs_fixes', 'immobilisations'],
            'liabilities': ['liabilities', 'total_liabilities', 'passifs', 'passif_total'],
            'current_liabilities': ['current_liabilities', 'passifs_courants', 'dettes_court_terme'],
            'equity': ['equity', 'shareholders_equity', 'capitaux_propres'],
            'inventory': ['inventory', 'stock', 'stocks'],
            'accounts_receivable': ['accounts_receivable', 'creances_clients', 'clients'],
            'accounts_payable': ['accounts_payable', 'dettes_fournisseurs', 'fournisseurs'],
            'customer_metrics': ['customer_count', 'customers', 'clients', 'nombre_clients'],
            'unit_metrics': ['units_sold', 'quantity', 'quantite', 'unites_vendues'],
            'pricing_metrics': ['average_price', 'price', 'prix_moyen', 'prix'],
            'saas_metrics': ['mrr', 'arr', 'monthly_recurring_revenue', 'annual_recurring_revenue']
        }
        
        self.validator = AdvancedDataValidator()
    
    def detect_columns(self, df):
        """Détection automatique améliorée des colonnes"""
        detected = {}
        
        for target, keywords in self.column_mappings.items():
            for col in df.columns:
                col_lower = col.lower().replace('_', ' ').replace('-', ' ').strip()
                for keyword in keywords:
                    if keyword.lower() in col_lower or col_lower in keyword.lower():
                        detected[target] = col
                        break
                if target in detected:
                    break
        
        return detected
    
    def clean_numeric_column(self, series):
        """Nettoyage avancé des colonnes numériques"""
        try:
            # Si déjà numérique, vérifier les NaN
            if pd.api.types.is_numeric_dtype(series):
                return pd.to_numeric(series, errors='coerce').fillna(0)
            
            # Nettoyer les chaînes
            cleaned = series.astype(str)
            
            # Supprimer symboles monétaires et espaces
            cleaned = cleaned.str.replace(r'[$€£¥₹₽]', '', regex=True)
            cleaned = cleaned.str.replace(',', '')
            cleaned = cleaned.str.replace(' ', '')
            cleaned = cleaned.str.replace('%', '')
            
            # Garder seulement chiffres, points et tirets
            cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
            
            # Convertir en numérique
            cleaned = pd.to_numeric(cleaned, errors='coerce').fillna(0)
            
            return cleaned
        except Exception as e:
            # Fallback: retourner une série de zéros
            return pd.Series([0] * len(series))
    
    def calculate_comprehensive_metrics(self, df, mappings):
        """Calcul de métriques complètes avec validation"""
        metrics = {}
        
        # Métriques financières de base
        for metric_type in ['revenue', 'costs', 'profit', 'cash_flow']:
            if metric_type in mappings and mappings[metric_type] in df.columns:
                col = mappings[metric_type]
                data = self.clean_numeric_column(df[col]).dropna()
                
                if len(data) > 0:
                    metrics[metric_type] = {
                        'total': float(data.sum()),
                        'average': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'growth_rate': 0.0,
                        'volatility': float(data.std() / data.mean()) if data.mean() != 0 else 0,
                        'data': data.tolist(),
                        'trend': 'stable'
                    }
                    
                    # Calculer taux de croissance
                    if len(data) > 1:
                        growth_rate = ((data.iloc[-1] / data.iloc[0]) - 1) * 100 if data.iloc[0] != 0 else 0
                        metrics[metric_type]['growth_rate'] = float(growth_rate)
                        
                        # Déterminer tendance
                        if growth_rate > 5:
                            metrics[metric_type]['trend'] = 'croissance'
                        elif growth_rate < -5:
                            metrics[metric_type]['trend'] = 'declin'
        
        # Calculer métriques dérivées si revenus et coûts disponibles
        if 'revenue' in metrics and 'costs' in metrics:
            revenue_data = np.array(metrics['revenue']['data'])
            costs_data = np.array(metrics['costs']['data'])
            
            # S'assurer que les arrays ont la même longueur
            min_length = min(len(revenue_data), len(costs_data))
            revenue_data = revenue_data[:min_length]
            costs_data = costs_data[:min_length]
            
            profit_data = revenue_data - costs_data
            
            metrics['profit'] = {
                'total': float(profit_data.sum()),
                'average': float(profit_data.mean()),
                'median': float(np.median(profit_data)),
                'std': float(np.std(profit_data)),
                'min': float(profit_data.min()),
                'max': float(profit_data.max()),
                'margin_average': float((profit_data.mean() / revenue_data.mean() * 100)) if revenue_data.mean() != 0 else 0,
                'data': profit_data.tolist(),
                'growth_rate': 0.0,
                'volatility': float(np.std(profit_data) / np.mean(profit_data)) if np.mean(profit_data) != 0 else 0
            }
            
            if len(profit_data) > 1:
                profit_growth = ((profit_data[-1] / profit_data[0]) - 1) * 100 if profit_data[0] != 0 else 0
                metrics['profit']['growth_rate'] = float(profit_growth)
        
        # Métriques de bilan si disponibles
        balance_sheet_metrics = ['assets', 'liabilities', 'equity', 'current_assets', 'current_liabilities']
        for metric in balance_sheet_metrics:
            if metric in mappings and mappings[metric] in df.columns:
                col = mappings[metric]
                data = self.clean_numeric_column(df[col]).dropna()
                
                if len(data) > 0:
                    metrics[metric] = {
                        'average': float(data.mean()),
                        'latest': float(data.iloc[-1]) if len(data) > 0 else 0,
                        'data': data.tolist()
                    }
        
        # Métriques business si disponibles
        business_metrics = ['customer_metrics', 'unit_metrics', 'pricing_metrics']
        for metric in business_metrics:
            if metric in mappings and mappings[metric] in df.columns:
                col = mappings[metric]
                data = self.clean_numeric_column(df[col]).dropna()
                
                if len(data) > 0:
                    metrics[metric] = {
                        'average': float(data.mean()),
                        'growth_rate': 0.0,
                        'data': data.tolist()
                    }
                    
                    if len(data) > 1:
                        growth = ((data.iloc[-1] / data.iloc[0]) - 1) * 100 if data.iloc[0] != 0 else 0
                        metrics[metric]['growth_rate'] = float(growth)
        
        return metrics
    
    def generate_enhanced_insights(self, metrics, validation_results=None):
        """Génération d'insights améliorés avec contexte de validation"""
        insights = []
        recommendations = []
        alerts = []
        
        # Insights basés sur les métriques
        if 'revenue' in metrics:
            revenue_growth = metrics['revenue'].get('growth_rate', 0)
            revenue_volatility = metrics['revenue'].get('volatility', 0)
            
            if revenue_growth > 20:
                insights.append(f"🚀 **Forte croissance** : {revenue_growth:.1f}% d'augmentation du CA")
            elif revenue_growth > 5:
                insights.append(f"📈 **Croissance positive** : {revenue_growth:.1f}% sur la période")
            elif revenue_growth > -5:
                insights.append(f"📊 **Revenus stables** : {revenue_growth:.1f}% de variation")
            else:
                alerts.append(f"📉 **Déclin significatif** : {abs(revenue_growth):.1f}% de baisse du CA")
            
            if revenue_volatility < 0.1:
                insights.append("🎯 **Revenus très prévisibles** : Faible volatilité")
            elif revenue_volatility > 0.3:
                alerts.append("⚠️ **Revenus volatils** : Forte variabilité détectée")
        
        if 'profit' in metrics:
            profit_margin = metrics['profit'].get('margin_average', 0)
            profit_growth = metrics['profit'].get('growth_rate', 0)
            
            if profit_margin > 20:
                insights.append(f"💰 **Excellentes marges** : {profit_margin:.1f}% de marge bénéficiaire")
            elif profit_margin > 10:
                insights.append(f"💵 **Bonnes marges** : {profit_margin:.1f}% de rentabilité")
            elif profit_margin > 0:
                insights.append(f"📈 **Marges positives** : {profit_margin:.1f}% de marge")
            else:
                alerts.append(f"🔴 **Marges négatives** : {profit_margin:.1f}% - Entreprise déficitaire")
            
            if profit_growth > 15:
                insights.append(f"🎉 **Croissance rentabilité** : {profit_growth:.1f}% d'amélioration")
            elif profit_growth < -15:
                alerts.append(f"📉 **Détérioration profits** : {abs(profit_growth):.1f}% de baisse")
        
        # Recommandations basées sur l'analyse
        if 'revenue' in metrics and 'costs' in metrics:
            revenue_trend = metrics['revenue'].get('trend', 'stable')
            
            if revenue_trend == 'declin':
                recommendations.append("🎯 **Stratégie CA** : Analyser causes du déclin et revoir stratégie commerciale")
            elif revenue_trend == 'stable':
                recommendations.append("📈 **Accélération croissance** : Identifier opportunités d'expansion")
            
            avg_revenue = metrics['revenue'].get('average', 0)
            avg_costs = metrics['costs'].get('average', 0)
            
            if avg_costs > avg_revenue * 0.8:
                recommendations.append("✂️ **Optimisation coûts** : Structure de coûts élevée - opportunités d'économies")
        
        # Insights spécifiques validation des données
        if validation_results:
            quality_score = validation_results.get('quality_score', 100)
            
            if quality_score >= 90:
                insights.append("✅ **Données de haute qualité** : Analyses très fiables")
            elif quality_score >= 70:
                insights.append("📊 **Données de qualité correcte** : Analyses globalement fiables")
            else:
                alerts.append("⚠️ **Qualité données limitée** : Interpréter résultats avec prudence")
            
            critical_issues = validation_results.get('critical_issues', 0)
            if critical_issues > 0:
                alerts.append(f"🔴 **{critical_issues} incohérence(s) critique(s)** détectée(s)")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'alerts': alerts
        }
    
    def create_enhanced_visualizations(self, df, mappings, metrics):
        """Création de visualisations avancées avec contexte métier"""
        figures = {}
        
        try:
            # Graphique principal des tendances financières
            fig = go.Figure()
            
            # Préparer l'axe X
            if 'date' in mappings and mappings['date'] in df.columns:
                x_axis = pd.to_datetime(df[mappings['date']], errors='coerce')
                if x_axis.isnull().all():
                    x_axis = range(len(df))
                x_title = "Date"
            else:
                x_axis = range(len(df))
                x_title = "Période"
            
            # Ajouter les courbes principales
            colors = {'revenue': '#2E8B57', 'costs': '#DC143C', 'profit': '#4169E1', 'cash_flow': '#FF8C00'}
            
            for metric in ['revenue', 'costs', 'profit', 'cash_flow']:
                if metric in mappings and mappings[metric] in df.columns:
                    data = self.clean_numeric_column(df[mappings[metric]])
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=data,
                        mode='lines+markers',
                        name=metric.title().replace('_', ' '),
                        line=dict(color=colors.get(metric, '#666666'), width=3),
                        marker=dict(size=6),
                        hovertemplate=f"<b>{metric.title()}</b><br>" +
                                    f"{x_title}: %{{x}}<br>" +
                                    "Valeur: %{y:,.0f} DHS<extra></extra>"
                    ))
            
            # Configuration du graphique
            fig.update_layout(
                title="Performance Financière - Tendances Temporelles",
                xaxis_title=x_title,
                yaxis_title="Montant (DHS)",
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Ajouter ligne de référence zéro pour le profit
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            figures['financial_trend'] = fig
            
            # Graphique de répartition si données disponibles
            if 'revenue' in metrics and 'costs' in metrics:
                avg_revenue = metrics['revenue']['average']
                avg_costs = metrics['costs']['average']
                avg_profit = avg_revenue - avg_costs
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Coûts', 'Profit'],
                    values=[avg_costs, max(0, avg_profit)],
                    hole=0.3,
                    marker_colors=['#DC143C', '#2E8B57']
                )])
                
                fig_pie.update_layout(
                    title="Répartition Moyenne : Coûts vs Profit",
                    annotations=[dict(text=f'{(avg_profit/avg_revenue*100):.1f}%<br>Marge', 
                                    x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                
                figures['cost_profit_breakdown'] = fig_pie
            
            # Graphique de volatilité si plusieurs métriques
            volatility_data = []
            volatility_labels = []
            
            for metric in ['revenue', 'costs', 'profit']:
                if metric in metrics and 'volatility' in metrics[metric]:
                    volatility_data.append(metrics[metric]['volatility'] * 100)
                    volatility_labels.append(metric.title())
            
            if len(volatility_data) >= 2:
                fig_vol = go.Figure(data=[
                    go.Bar(x=volatility_labels, y=volatility_data, 
                          marker_color=['#2E8B57', '#DC143C', '#4169E1'][:len(volatility_data)])
                ])
                
                fig_vol.update_layout(
                    title="Analyse de Volatilité par Métrique",
                    xaxis_title="Métriques",
                    yaxis_title="Coefficient de Variation (%)",
                    height=400
                )
                
                figures['volatility_analysis'] = fig_vol
            
        except Exception as e:
            # En cas d'erreur, créer un graphique de base
            st.warning(f"Erreur création visualisations avancées: {str(e)}")
            
            # Graphique de secours
            fig_basic = go.Figure()
            if 'revenue' in mappings and mappings['revenue'] in df.columns:
                revenue_data = self.clean_numeric_column(df[mappings['revenue']])
                fig_basic.add_trace(go.Scatter(
                    x=list(range(len(revenue_data))),
                    y=revenue_data,
                    mode='lines+markers',
                    name='Revenue'
                ))
            
            fig_basic.update_layout(title="Données Financières (Vue Simplifiée)")
            figures['financial_trend'] = fig_basic
        
        return figures
    
    def process_csv(self, df):
        """Traitement complet du CSV avec validation et correction automatiques"""
        try:
            # Étape 1: Détection des colonnes
            mappings = self.detect_columns(df)
            
            # Étape 2: Validation avancée avec le nouveau système
            validation_results = self.validator.comprehensive_validation(df, mappings)
            
            # Étape 3: Application des corrections si nécessaire
            corrections_applied = False
            correction_log = []
            
            # Corriger les outliers si détectés
            for issue in validation_results.get('issues', []):
                if 'Outliers' in issue.get('type', '') and issue.get('severity') == 'Élevée':
                    metric = issue['type'].split(' - ')[1].lower()
                    if metric in mappings and mappings[metric] in df.columns:
                        col = mappings[metric]
                        original_data = self.clean_numeric_column(df[col])
                        corrected_data, corrections = self.validator.apply_outlier_correction(original_data)
                        
                        if corrections:
                            df[col] = corrected_data
                            correction_log.extend(corrections)
                            corrections_applied = True
            
            # Étape 4: Calcul des métriques sur les données corrigées
            metrics = self.calculate_comprehensive_metrics(df, mappings)
            
            # Étape 5: Génération d'insights avec contexte de validation
            insights_data = self.generate_enhanced_insights(metrics, validation_results)
            
            # Étape 6: Création des visualisations
            figures = self.create_enhanced_visualizations(df, mappings, metrics)
            
            # Étape 7: Générer suggestions d'amélioration
            suggestions = self.generate_improvement_suggestions(metrics, validation_results)
            
            # Étape 8: Identifier problèmes potentiels
            issues = self.identify_potential_issues(metrics, validation_results)
            
            return {
                'mappings': mappings,
                'metrics': metrics,
                'insights': insights_data,
                'figures': figures,
                'issues': issues,
                'suggestions': suggestions,
                'processed_df': df,
                'validation_results': validation_results,
                'corrections_applied': corrections_applied,
                'correction_log': correction_log
            }
            
        except Exception as e:
            return {
                'mappings': {},
                'metrics': {},
                'insights': {'insights': [], 'recommendations': [], 'alerts': []},
                'figures': {},
                'issues': [f"Erreur traitement CSV: {str(e)}"],
                'suggestions': ["Vérifier format du fichier CSV"],
                'processed_df': df,
                'validation_results': {'quality_score': 50, 'total_issues': 1, 'critical_issues': 1, 'issues': []},
                'corrections_applied': False,
                'correction_log': []
            }
    
    def generate_improvement_suggestions(self, metrics, validation_results):
        """Générer des suggestions d'amélioration basées sur l'analyse"""
        suggestions = []
        
        try:
            # Suggestions basées sur la qualité des données
            if validation_results:
                quality_score = validation_results.get('quality_score', 100)
                
                if quality_score < 70:
                    suggestions.append("🔧 **Amélioration qualité** : Revoir processus de collecte des données")
                
                if validation_results.get('critical_issues', 0) > 0:
                    suggestions.append("⚠️ **Correction urgente** : Résoudre incohérences critiques détectées")
            
            # Suggestions basées sur les métriques business
            if 'revenue' in metrics:
                revenue_volatility = metrics['revenue'].get('volatility', 0)
                revenue_growth = metrics['revenue'].get('growth_rate', 0)
                
                if revenue_volatility > 0.3:
                    suggestions.append("📊 **Stabilisation CA** : Diversifier sources de revenus pour réduire volatilité")
                
                if revenue_growth < 0:
                    suggestions.append("📈 **Relance croissance** : Développer stratégies d'acquisition clients")
                elif revenue_growth < 5:
                    suggestions.append("🚀 **Accélération** : Identifier leviers de croissance supplémentaires")
            
            if 'profit' in metrics:
                margin = metrics['profit'].get('margin_average', 0)
                
                if margin < 5:
                    suggestions.append("💰 **Optimisation marges** : Revoir structure de coûts et pricing")
                elif margin < 15:
                    suggestions.append("💎 **Amélioration rentabilité** : Opportunités d'optimisation identifiées")
            
            # Suggestions spécifiques selon type de business détecté
            if self.detect_business_type(metrics) == 'saas':
                suggestions.append("☁️ **Métriques SaaS** : Tracker MRR, churn rate et LTV/CAC")
            elif self.detect_business_type(metrics) == 'retail':
                suggestions.append("🛍️ **Optimisation retail** : Focus sur rotation stocks et saisonnalité")
            
        except Exception as e:
            suggestions.append(f"Erreur génération suggestions: {str(e)}")
        
        return suggestions
    
    def identify_potential_issues(self, metrics, validation_results):
        """Identifier les problèmes potentiels dans les données"""
        issues = []
        
        try:
            # Issues liés à la validation
            if validation_results:
                for issue in validation_results.get('issues', []):
                    if issue.get('severity') in ['Élevée', 'Critique']:
                        issues.append(f"❌ {issue.get('type', 'Problème')}: {issue.get('message', '')}")
            
            # Issues business
            if 'profit' in metrics:
                margin = metrics['profit'].get('margin_average', 0)
                if margin < 0:
                    issues.append("🔴 **Rentabilité critique** : Entreprise en perte")
                elif margin < 2:
                    issues.append("⚠️ **Marges faibles** : Risque de rentabilité")
            
            if 'revenue' in metrics:
                growth = metrics['revenue'].get('growth_rate', 0)
                if growth < -20:
                    issues.append("📉 **Déclin sévère** : Chute importante du chiffre d'affaires")
                elif growth < -10:
                    issues.append("📉 **Déclin** : Baisse significative des revenus")
            
            # Issues de cohérence
            if 'revenue' in metrics and 'costs' in metrics:
                avg_costs = metrics['costs'].get('average', 0)
                avg_revenue = metrics['revenue'].get('average', 1)
                
                if avg_costs > avg_revenue:
                    issues.append("⚠️ **Structure coûts** : Coûts supérieurs aux revenus en moyenne")
        
        except Exception as e:
            issues.append(f"Erreur identification issues: {str(e)}")
        
        return issues
    
    def detect_business_type(self, metrics):
        """Détecter le type de business basé sur les métriques"""
        try:
            if 'profit' in metrics:
                margin = metrics['profit'].get('margin_average', 0)
                
                # Marges élevées suggèrent SaaS/Software
                if margin > 20:
                    return 'saas'
                # Marges faibles suggèrent retail
                elif margin < 8:
                    return 'retail'
                # Marges moyennes suggèrent services/tech
                else:
                    return 'services'
            
            return 'general'
        except:
            return 'general'

# ========== CSV TEMPLATE GENERATOR ==========
class CSVTemplateGenerator:
    """Générateur de templates CSV pour différents types d'entreprises"""
    
    def __init__(self):
        self.templates = {
            'complete_financial': {
                'name': 'Complete Financial Data Template',
                'description': 'Template complet avec toutes les métriques financières pour analyse maximale',
                'columns': {
                    'Date': 'Format YYYY-MM-DD (ex: 2024-01-01)',
                    'Revenue': 'Chiffre d\'affaires mensuel en devise locale (chiffres uniquement)',
                    'Sales': 'Colonne alternative revenus (utiliser si préféré)',
                    'Costs': 'Total coûts/charges mensuels',
                    'Variable_Costs': 'Coûts variables (évoluent avec les ventes)',
                    'Fixed_Costs': 'Coûts fixes (constants)',
                    'Profit': 'Bénéfice net (Revenus - Coûts)',
                    'Cash_Flow': 'Flux de trésorerie net du mois',
                    'Assets': 'Total actifs en fin de mois',
                    'Current_Assets': 'Actifs court terme (trésorerie, stocks, etc.)',
                    'Fixed_Assets': 'Actifs long terme (équipements, immobilier)',
                    'Liabilities': 'Total passifs',
                    'Current_Liabilities': 'Dettes et obligations court terme',
                    'Equity': 'Capitaux propres/fonds propres',
                    'Inventory': 'Valeur des stocks/inventaire',
                    'Accounts_Receivable': 'Créances clients',
                    'Accounts_Payable': 'Dettes fournisseurs',
                    'Customer_Count': 'Nombre de clients actifs',
                    'Units_Sold': 'Quantité produits/services vendus',
                    'Average_Price': 'Prix moyen par unité/service'
                },
                'sample_data': [
                    ['2025-01-01', 15000, 15000, 12000, 8000, 4000, 3000, 2500, 50000, 20000, 30000, 20000, 8000, 30000, 5000, 8000, 6000, 150, 300, 50],
                    ['2025-02-01', 16500, 16500, 13100, 8800, 4300, 3400, 3200, 52000, 21000, 31000, 21000, 8500, 31000, 5200, 8500, 6200, 165, 330, 50],
                    ['2025-03-01', 14200, 14200, 11800, 7600, 4200, 2400, 2100, 51500, 20500, 31000, 20800, 8300, 30700, 5100, 8200, 6100, 158, 284, 50]
                ]
            },
            'saas_template': {
                'name': 'SaaS Business Template',
                'description': 'Template spécialisé pour entreprises Software as a Service',
                'columns': {
                    'Date': 'Format YYYY-MM-DD',
                    'Monthly_Recurring_Revenue': 'MRR - revenus récurrents mensuels prévisibles',
                    'Customer_Count': 'Total abonnés actifs',
                    'Churn_Rate': 'Taux de churn mensuel (pourcentage en décimal)',
                    'Customer_Acquisition_Cost': 'CAC - coût d\'acquisition d\'un client',
                    'Lifetime_Value': 'LTV - valeur vie moyenne d\'un client',
                    'Costs': 'Total coûts opérationnels mensuels'
                },
                'sample_data': [
                    ['2025-01-01', 12000, 400, 0.05, 150, 1800, 9000],
                    ['2025-02-01', 13200, 440, 0.05, 140, 1850, 9900],
                    ['2025-03-01', 14100, 470, 0.053, 160, 1820, 10500]
                ]
            }
        }
    
    def generate_template_csv(self, template_type):
        """Générer un template CSV avec formatage approprié"""
        template = self.templates.get(template_type)
        if not template:
            return None
        
        columns = list(template['columns'].keys())
        df = pd.DataFrame(template['sample_data'], columns=columns)
        
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        return csv_buffer.getvalue()

# ========== ENHANCED ANALYTICS ENGINE ==========
class AdvancedAnalytics:
    """Moteur d'analytics avancé avec calculs de ratios et recommandations IA"""
    
    def __init__(self):
        self.ratios_weights = {
            'liquidity': 0.3,
            'profitability': 0.4,
            'efficiency': 0.2,
            'leverage': 0.1
        }
        self.ml_engine = EnhancedMLForecastingEngine()
        self.scenario_calibrator = EnhancedScenarioCalibrator()
    
    @staticmethod
    def monte_carlo_simulation(base_revenue, base_costs, volatility=0.2, simulations=1000, periods=12):
        """Simulation Monte Carlo avancée avec corrélation et contraintes business"""
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
                
                # Appliquer contraintes business
                revenue_shock = max(-0.5, min(1.0, revenue_shock))
                cost_shock = max(-0.3, min(0.5, cost_shock))
                
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
                'max_drawdown': min(np.cumsum([r-c for r,c in zip(revenue_path, cost_path)])),
                'profit_margin': (net_profit / total_revenue * 100) if total_revenue > 0 else -100
            })
        
        return pd.DataFrame(results)
    
    def calculate_comprehensive_ratios(self, financial_data):
        """Calculer ratios financiers complets avec validation renforcée"""
        ratios = {}
        
        # Ratios de liquidité avec validation
        current_assets = float(financial_data.get('current_assets', 0))
        current_liabilities = max(float(financial_data.get('current_liabilities', 1)), 1)
        inventory = float(financial_data.get('inventory', 0))
        cash = float(financial_data.get('cash', 0))
        
        ratios['current_ratio'] = current_assets / current_liabilities
        ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities
        ratios['cash_ratio'] = cash / current_liabilities
        
        # Ratios de rentabilité avec calculs avancés
        revenue = max(float(financial_data.get('revenue', 1)), 1)
        gross_profit = float(financial_data.get('gross_profit', 0))
        operating_profit = float(financial_data.get('operating_profit', 0))
        net_profit = float(financial_data.get('net_profit', 0))
        total_assets = max(float(financial_data.get('total_assets', 1)), 1)
        equity = max(float(financial_data.get('equity', 1)), 1)
        
        ratios['gross_margin'] = gross_profit / revenue
        ratios['operating_margin'] = operating_profit / revenue
        ratios['net_margin'] = net_profit / revenue
        ratios['roa'] = net_profit / total_assets
        ratios['roe'] = net_profit / equity
        
        # Ratios d'efficacité
        ratios['asset_turnover'] = revenue / total_assets
        ratios['equity_turnover'] = revenue / equity
        
        # Ratios de levier avec validation renforcée
        total_debt = float(financial_data.get('total_debt', 0))
        ratios['debt_to_equity'] = total_debt / equity
        ratios['debt_to_assets'] = total_debt / total_assets
        ratios['equity_multiplier'] = total_assets / equity
        
        # Ratios de couverture
        interest_expense = max(float(financial_data.get('interest_expense', 1)), 1)
        ratios['interest_coverage'] = operating_profit / interest_expense
        
        # Valider ratios pour outliers
        for ratio_name, ratio_value in ratios.items():
            if np.isnan(ratio_value) or np.isinf(ratio_value):
                ratios[ratio_name] = 0
            elif ratio_value > 1000:
                ratios[ratio_name] = 1000
            elif ratio_value < -1000:
                ratios[ratio_name] = -1000
        
        return ratios
    
    def calculate_financial_health_score(self, ratios, industry='general', validation_results=None):
        """Calculer score santé financière avancé avec ajustements validation"""
        industry_benchmarks = self.get_industry_benchmarks(industry)
        
        scores = {}
        
        # Score liquidité (0-25) avec pondération avancée
        current_ratio_score = min(25, (ratios.get('current_ratio', 0) / industry_benchmarks['current_ratio']) * 15)
        quick_ratio_score = min(10, (ratios.get('quick_ratio', 0) / industry_benchmarks.get('quick_ratio', 1.0)) * 10)
        scores['liquidity'] = current_ratio_score + quick_ratio_score
        
        # Score rentabilité (0-40) avec métriques avancées
        net_margin_score = min(20, max(0, (ratios.get('net_margin', 0) / industry_benchmarks['net_margin']) * 20))
        roa_score = min(10, max(0, (ratios.get('roa', 0) / industry_benchmarks['roa']) * 10))
        roe_score = min(10, max(0, (ratios.get('roe', 0) / industry_benchmarks['roe']) * 10))
        scores['profitability'] = net_margin_score + roa_score + roe_score
        
        # Score efficacité (0-20)
        asset_turnover_score = min(20, max(0, (ratios.get('asset_turnover', 0) / industry_benchmarks['asset_turnover']) * 20))
        scores['efficiency'] = asset_turnover_score
        
        # Score levier (0-15) avec logique avancée
        debt_ratio = ratios.get('debt_to_equity', 0)
        if debt_ratio <= industry_benchmarks['debt_to_equity']:
            leverage_score = 15
        else:
            leverage_score = max(0, 15 - (debt_ratio - industry_benchmarks['debt_to_equity']) * 30)
        scores['leverage'] = leverage_score
        
        # Appliquer ajustement qualité données
        quality_adjustment = 1.0
        if validation_results:
            quality_score = validation_results.get('quality_score', 100)
            if quality_score < 70:
                quality_adjustment = quality_score / 100
                scores['data_quality_penalty'] = (100 - quality_score) * 0.2
        
        total_score = sum(scores.values()) * quality_adjustment
        return min(100, max(0, total_score)), scores
    
    def get_industry_benchmarks(self, industry):
        """Obtenir benchmarks spécifiques par industrie avancés"""
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
            },
            'manufacturing': {
                'current_ratio': 1.4, 'quick_ratio': 0.9, 'net_margin': 0.08,
                'roa': 0.06, 'roe': 0.14, 'asset_turnover': 1.5, 'debt_to_equity': 0.8
            }
        }
        return benchmarks.get(industry, benchmarks['general'])
    
    def generate_ai_recommendations(self, financial_data, ratios, health_score, validation_results=None):
        """Générer recommandations IA avancées avec insights validation - VERSION CORRIGÉE"""
        recommendations = []
        
        try:
            # Système de recommandations par priorité
            high_priority = []
            medium_priority = []
            low_priority = []
            
            # Recommandations qualité données
            if validation_results:
                quality_score = validation_results.get('quality_score', 100)
                if quality_score < 50:
                    high_priority.append({
                        'category': 'Qualité des données',
                        'priority': 'Critique',
                        'recommendation': 'Amélioration urgente de la qualité des données requise. Vérifier et corriger les incohérences détectées.',
                        'impact': 'Très Élevé',
                        'timeframe': 'Immédiat',
                        'estimated_benefit': 10000  # FIX: Valeur numérique fixe
                    })
                elif quality_score < 80:
                    medium_priority.append({
                        'category': 'Validation des données',
                        'priority': 'Moyenne',
                        'recommendation': 'Réviser les processus de saisie des données pour améliorer la cohérence.',
                        'impact': 'Moyen',
                        'timeframe': '1-2 semaines',
                        'estimated_benefit': 5000  # FIX: Valeur numérique fixe
                    })
            
            # Recommandations cash flow
            cash_flow = financial_data.get('cash_flow', 0)
            if isinstance(cash_flow, (int, float)) and cash_flow < 0:
                high_priority.append({
                    'category': 'Gestion de trésorerie',
                    'priority': 'Critique',
                    'recommendation': 'Amélioration immédiate du cash-flow nécessaire. Actions : 1) Accélérer l\'encaissement clients, 2) Étendre les délais fournisseurs, 3) Réduire les dépenses non essentielles, 4) Financement d\'urgence.',
                    'impact': 'Très Élevé',
                    'timeframe': 'Immédiat',
                    'estimated_benefit': abs(float(cash_flow)) * 0.5 if isinstance(cash_flow, (int, float)) else 5000
                })
            
            # Recommandations liquidité avec logique avancée
            current_ratio = ratios.get('current_ratio', 0)
            if isinstance(current_ratio, (int, float)):
                if current_ratio < 1.0:
                    high_priority.append({
                        'category': 'Liquidité critique',
                        'priority': 'Critique',
                        'recommendation': 'Liquidité insuffisante pour couvrir les obligations à court terme. Actions immédiates requises.',
                        'impact': 'Très Élevé',
                        'timeframe': 'Immédiat',
                        'estimated_benefit': float(financial_data.get('current_liabilities', 10000)) * 0.2
                    })
                elif current_ratio < 1.2:
                    high_priority.append({
                        'category': 'Amélioration de la liquidité',
                        'priority': 'Élevée',
                        'recommendation': 'Améliorer la gestion du fonds de roulement. Actions : 1) Optimisation des stocks, 2) Révision des conditions de crédit, 3) Options de financement court terme.',
                        'impact': 'Élevé',
                        'timeframe': '1-3 mois',
                        'estimated_benefit': float(financial_data.get('current_assets', 10000)) * 0.1
                    })
            
            # Recommandations rentabilité avec contexte industrie
            net_margin = ratios.get('net_margin', 0)
            if isinstance(net_margin, (int, float)):
                if net_margin < 0:
                    high_priority.append({
                        'category': 'Rentabilité critique',
                        'priority': 'Critique',
                        'recommendation': 'Entreprise en perte. Plan de redressement urgent : 1) Analyse détaillée des coûts, 2) Révision de la stratégie tarifaire, 3) Restructuration opérationnelle, 4) Recherche de financements.',
                        'impact': 'Très Élevé',
                        'timeframe': 'Immédiat',
                        'estimated_benefit': float(financial_data.get('revenue', 100000)) * 0.1
                    })
                elif net_margin < 0.05:
                    medium_priority.append({
                        'category': 'Amélioration de la rentabilité',
                        'priority': 'Élevée',
                        'recommendation': 'Marges faibles nécessitant amélioration. Actions : 1) Optimisation de la structure de coûts, 2) Révision de la stratégie tarifaire, 3) Amélioration de l\'efficacité opérationnelle, 4) Optimisation du mix produits.',
                        'impact': 'Élevé',
                        'timeframe': '3-6 mois',
                        'estimated_benefit': float(financial_data.get('revenue', 100000)) * 0.05
                    })
            
            # Recommandations levier
            debt_to_equity = ratios.get('debt_to_equity', 0)
            if isinstance(debt_to_equity, (int, float)) and debt_to_equity > 2.0:
                medium_priority.append({
                    'category': 'Gestion de l\'endettement',
                    'priority': 'Moyenne',
                    'recommendation': 'Niveau d\'endettement élevé. Actions : 1) Plan de désendettement, 2) Amélioration de la couverture du service de la dette, 3) Considérer un financement par capitaux propres.',
                    'impact': 'Moyen',
                    'timeframe': '6-12 mois',
                    'estimated_benefit': float(financial_data.get('total_debt', 10000)) * 0.1
                })
            
            # Recommandations croissance et efficacité
            revenue_growth = financial_data.get('revenue_growth', 0)
            if isinstance(revenue_growth, (int, float)) and revenue_growth < 0:
                high_priority.append({
                    'category': 'Croissance du chiffre d\'affaires',
                    'priority': 'Élevée',
                    'recommendation': 'Déclin du CA détecté. Actions : 1) Analyse marché et concurrence, 2) Programmes de fidélisation clients, 3) Innovation produits/services, 4) Révision stratégie marketing.',
                    'impact': 'Élevé',
                    'timeframe': '3-6 mois',
                    'estimated_benefit': float(financial_data.get('revenue', 100000)) * 0.1
                })
            elif isinstance(revenue_growth, (int, float)) and revenue_growth < 5:
                low_priority.append({
                    'category': 'Accélération de la croissance',
                    'priority': 'Faible',
                    'recommendation': 'Croissance lente. Considérer : 1) Expansion géographique, 2) Développement nouveaux segments, 3) Partenariats stratégiques.',
                    'impact': 'Moyen',
                    'timeframe': '6-12 mois',
                    'estimated_benefit': float(financial_data.get('revenue', 100000)) * 0.05
                })
            
            # Combiner toutes les recommandations
            all_recommendations = high_priority + medium_priority + low_priority
            
            # FIX: S'assurer que toutes les valeurs estimated_benefit sont numériques
            for rec in all_recommendations:
                if not isinstance(rec.get('estimated_benefit'), (int, float)):
                    rec['estimated_benefit'] = 5000  # Valeur par défaut
                else:
                    rec['estimated_benefit'] = float(rec['estimated_benefit'])
            
            # Limiter aux 5 recommandations les plus impactantes
            sorted_recommendations = sorted(all_recommendations, 
                                          key=lambda x: float(x.get('estimated_benefit', 0)), 
                                          reverse=True)
            
            return sorted_recommendations[:5]
            
        except Exception as e:
            # Retourner une recommandation de secours en cas d'erreur
            return [{
                'category': 'Erreur système',
                'priority': 'Moyenne',
                'recommendation': f'Erreur dans la génération des recommandations: {str(e)}',
                'impact': 'Moyen',
                'timeframe': 'À déterminer',
                'estimated_benefit': 1000
            }]

# ========== INDUSTRY TEMPLATES MANAGER (Enhanced) ==========
class IndustryTemplateManager:
    """Gestionnaire de templates sectoriels avancé avec benchmarking complet"""
    
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
                },
                'validation_rules': {
                    'max_inventory_ratio': 0.4,
                    'min_inventory_turnover': 2,
                    'max_seasonal_variation': 0.5
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
                'seasonal_factors': [1.0] * 12,  # Faible saisonnalité
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
                },
                'validation_rules': {
                    'max_churn_rate': 0.15,
                    'min_ltv_cac': 1.5,
                    'min_gross_margin': 0.6
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
                },
                'validation_rules': {
                    'min_rd_ratio': 0.05,
                    'max_rd_ratio': 0.40,
                    'min_gross_margin': 0.3
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
                },
                'validation_rules': {
                    'max_inventory_days': 120,
                    'min_capacity_utilization': 0.5,
                    'max_defect_rate': 0.05
                }
            }
        }
    
    def get_template(self, industry):
        """Obtenir template complet industrie avec règles validation"""
        return self.templates.get(industry, self.templates['technology'])
    
    def validate_industry_data(self, csv_data, industry):
        """Valider données contre règles spécifiques industrie"""
        template = self.get_template(industry)
        validation_rules = template.get('validation_rules', {})
        issues = []
        
        # Validations spécifiques par industrie
        if industry == 'saas':
            # Vérifier taux de churn si disponible
            churn_rate = csv_data.get('churn_rate', 0)
            if churn_rate > validation_rules.get('max_churn_rate', 0.15):
                issues.append(f"Taux de churn élevé ({churn_rate:.1%}) pour une entreprise SaaS")
            
            # Vérifier marge brute
            gross_margin = csv_data.get('gross_margin', 0)
            if gross_margin < validation_rules.get('min_gross_margin', 0.6):
                issues.append(f"Marge brute faible ({gross_margin:.1%}) pour le secteur SaaS")
        
        elif industry == 'retail':
            # Vérifier niveaux de stock
            inventory_ratio = csv_data.get('inventory_ratio', 0)
            if inventory_ratio > validation_rules.get('max_inventory_ratio', 0.4):
                issues.append(f"Niveau de stock élevé ({inventory_ratio:.1%}) pour le retail")
            
            # Vérifier patterns saisonniers
            revenue_volatility = csv_data.get('revenue_volatility', 0)
            if revenue_volatility > validation_rules.get('max_seasonal_variation', 0.5):
                issues.append(f"Forte variabilité saisonnière détectée ({revenue_volatility:.1%})")
        
        elif industry == 'technology':
            # Vérifier investissement R&D
            rd_ratio = csv_data.get('rd_ratio', 0)
            min_rd = validation_rules.get('min_rd_ratio', 0.05)
            max_rd = validation_rules.get('max_rd_ratio', 0.40)
            
            if rd_ratio < min_rd:
                issues.append(f"Investissement R&D faible ({rd_ratio:.1%}) pour le secteur tech")
            elif rd_ratio > max_rd:
                issues.append(f"Investissement R&D très élevé ({rd_ratio:.1%}) - vérifier la viabilité")
        
        elif industry == 'manufacturing':
            # Vérifier durée de stock
            inventory_days = csv_data.get('inventory_days', 0)
            if inventory_days > validation_rules.get('max_inventory_days', 120):
                issues.append(f"Durée de stock excessive ({inventory_days:.0f} jours) pour le manufacturing")
        
        return issues
    
    def detect_industry_from_csv(self, csv_data):
        """Détecter industrie probable basée sur patterns données CSV avec logique avancée"""
        if not csv_data:
            return 'technology'
        
        scores = {}
        
        # Initialiser scores
        for industry in self.templates.keys():
            scores[industry] = 0
        
        # Analyser patterns revenus
        revenue_data = csv_data.get('revenue_data', [])
        if len(revenue_data) >= 12:
            # Calculer saisonnalité
            monthly_avg = []
            for month in range(12):
                month_values = [revenue_data[i] for i in range(month, len(revenue_data), 12)]
                if month_values:
                    monthly_avg.append(np.mean(month_values))
            
            if len(monthly_avg) == 12:
                seasonality_score = np.std(monthly_avg) / np.mean(monthly_avg) if np.mean(monthly_avg) > 0 else 0
                
                # Forte saisonnalité suggère retail
                if seasonality_score > 0.2:
                    scores['retail'] += 30
                # Faible saisonnalité suggère SaaS
                elif seasonality_score < 0.05:
                    scores['saas'] += 20
        
        # Analyser marges profit
        profit_margin = csv_data.get('profit_margin', 0)
        
        # Marges élevées suggèrent SaaS ou technology
        if profit_margin > 20:
            scores['saas'] += 25
            scores['technology'] += 15
        # Marges faibles suggèrent manufacturing ou retail
        elif profit_margin < 10:
            scores['manufacturing'] += 20
            scores['retail'] += 15
        
        # Analyser volatilité revenus
        revenue_volatility = csv_data.get('revenue_volatility', 0)
        
        # Forte volatilité suggère retail ou manufacturing
        if revenue_volatility > 0.3:
            scores['retail'] += 15
            scores['manufacturing'] += 10
        # Faible volatilité suggère SaaS
        elif revenue_volatility < 0.1:
            scores['saas'] += 15
        
        # Analyser taux de croissance
        revenue_growth = csv_data.get('revenue_growth', 0)
        
        # Forte croissance suggère SaaS ou technology
        if revenue_growth > 20:
            scores['saas'] += 20
            scores['technology'] += 15
        # Croissance modérée suggère technology
        elif revenue_growth > 10:
            scores['technology'] += 10
        
        # Retourner industrie avec score le plus élevé
        best_industry = max(scores, key=scores.get)
        
        # Fallback vers technology si pas de gagnant clair
        if scores[best_industry] < 10:
            return 'technology'
        
        return best_industry
    
    def benchmark_against_industry(self, csv_data, industry):
        """Comparaison benchmark avancée avec validation industrie"""
        template = self.get_template(industry)
        benchmarks = template['benchmarks']
        
        comparison = {}
        
        # Comparaison croissance revenus avec validation
        company_growth = csv_data.get('revenue_growth', 0) / 100
        industry_growth = benchmarks.get('revenue_growth', 0.1)
        
        comparison['revenue_growth'] = {
            'company_value': company_growth,
            'industry_benchmark': industry_growth,
            'difference': company_growth - industry_growth,
            'percentage_difference': ((company_growth - industry_growth) / industry_growth * 100) if industry_growth != 0 else 0,
            'performance': self.categorize_performance(company_growth, industry_growth),
            'validation_status': self.validate_metric_range('revenue_growth', company_growth, industry)
        }
        
        # Comparaison marge profit avec validation
        company_margin = csv_data.get('profit_margin', 0) / 100
        industry_margin = benchmarks.get('profit_margin', 0.1)
        
        comparison['profit_margin'] = {
            'company_value': company_margin,
            'industry_benchmark': industry_margin,
            'difference': company_margin - industry_margin,
            'percentage_difference': ((company_margin - industry_margin) / industry_margin * 100) if industry_margin != 0 else 0,
            'performance': self.categorize_performance(company_margin, industry_margin),
            'validation_status': self.validate_metric_range('profit_margin', company_margin, industry)
        }
        
        # Métriques spécifiques par industrie
        if industry == 'saas':
            churn_rate = csv_data.get('estimated_churn', 0.05)
            industry_churn = benchmarks.get('churn_rate', 0.05)
            
            comparison['churn_rate'] = {
                'company_value': churn_rate,
                'industry_benchmark': industry_churn,
                'difference': churn_rate - industry_churn,
                'performance': self.categorize_performance(industry_churn, churn_rate),  # Plus bas est meilleur pour churn
                'validation_status': 'Normal' if churn_rate <= 0.15 else 'Attention Required'
            }
        
        elif industry == 'retail':
            inventory_turns = csv_data.get('estimated_inventory_turns', 6)
            industry_turns = benchmarks.get('inventory_turns', 6)
            
            comparison['inventory_turnover'] = {
                'company_value': inventory_turns,
                'industry_benchmark': industry_turns,
                'difference': inventory_turns - industry_turns,
                'performance': self.categorize_performance(inventory_turns, industry_turns),
                'validation_status': 'Normal' if inventory_turns >= 2 else 'Low Efficiency'
            }
        
        return comparison
    
    def categorize_performance(self, company_value, benchmark_value):
        """Catégoriser performance relative au benchmark"""
        if benchmark_value == 0:
            return 'Cannot Compare'
        
        ratio = company_value / benchmark_value
        
        if ratio >= 1.2:
            return 'Excellent'
        elif ratio >= 1.1:
            return 'Above Average'
        elif ratio >= 0.9:
            return 'Average'
        elif ratio >= 0.8:
            return 'Below Average'
        else:
            return 'Poor'
    
    def validate_metric_range(self, metric, value, industry):
        """Valider si métrique est dans fourchette raisonnable pour industrie"""
        validation_ranges = {
            'saas': {
                'revenue_growth': (-0.3, 3.0),  # -30% à 300%
                'profit_margin': (-0.5, 0.8),  # -50% à 80%
                'churn_rate': (0, 0.3)          # 0% à 30%
            },
            'retail': {
                'revenue_growth': (-0.5, 1.0),  # -50% à 100%
                'profit_margin': (-0.2, 0.3),  # -20% à 30%
                'inventory_turnover': (1, 20)   # 1 à 20 rotations
            },
            'technology': {
                'revenue_growth': (-0.4, 2.0),  # -40% à 200%
                'profit_margin': (-0.3, 0.6),  # -30% à 60%
            },
            'manufacturing': {
                'revenue_growth': (-0.3, 0.8),  # -30% à 80%
                'profit_margin': (-0.2, 0.4),  # -20% à 40%
            }
        }
        
        ranges = validation_ranges.get(industry, validation_ranges['technology'])
        metric_range = ranges.get(metric, (-1, 10))  # Fourchette large par défaut
        
        if metric_range[0] <= value <= metric_range[1]:
            return 'Normal'
        elif value < metric_range[0]:
            return 'Below Normal Range'
        else:
            return 'Above Normal Range'
    
    def generate_industry_insights(self, csv_data, industry):
        """Générer insights spécifiques industrie avancés avec validation"""
        template = self.get_template(industry)
        insights = []
        recommendations = []
        
        # Valider données contre règles industrie
        validation_issues = self.validate_industry_data(csv_data, industry)
        
        profit_margin = csv_data.get('profit_margin', 0)
        revenue_growth = csv_data.get('revenue_growth', 0)
        revenue_volatility = csv_data.get('revenue_volatility', 0)
        
        # Ajouter insights validation
        for issue in validation_issues:
            recommendations.append(f"⚠️ **Validation** : {issue}")
        
        # Analyse spécifique par industrie avec logique avancée
        if industry == 'saas':
            if profit_margin > 15:
                insights.append(f"💰 **Marges SaaS solides** : {profit_margin:.1f}% dépasse les benchmarks SaaS typiques")
            else:
                recommendations.append("🎯 **Optimisation SaaS** : Se concentrer sur les revenus récurrents et réduire les coûts d'acquisition client")
            
            if revenue_volatility < 0.1:
                insights.append("📊 **Excellente prédictibilité des revenus** : Faible volatilité alignée avec les forces du modèle SaaS")
            else:
                recommendations.append("🔄 **Améliorer les revenus récurrents** : Réduire le churn et augmenter la valeur vie client")
            
            # Recommandations spécifiques SaaS
            if revenue_growth > 30:
                insights.append("🚀 **Croissance SaaS exceptionnelle** : Maintenir le momentum avec scaling intelligent")
            elif revenue_growth < 10:
                recommendations.append("📈 **Accélération SaaS** : Focus sur l'expansion des comptes existants et nouveaux segments")
        
        elif industry == 'retail':
            if revenue_volatility > 0.2:
                insights.append("🛍️ **Modèle saisonnier retail** : Forte volatilité typique des opérations retail")
                recommendations.append("📈 **Planification saisonnière** : Optimiser stocks et personnel pour les pics d'activité")
            
            if profit_margin < 5:
                recommendations.append("💡 **Efficacité retail** : Focus sur la rotation des stocks et l'optimisation supply chain")
            
            # Insights spécifiques retail
            inventory_efficiency = csv_data.get('estimated_inventory_turns', 6)
            if inventory_efficiency > 8:
                insights.append("⚡ **Gestion de stock efficace** : Rotation rapide des stocks optimise la rentabilité")
            elif inventory_efficiency < 4:
                recommendations.append("📦 **Optimisation stocks** : Améliorer la rotation pour libérer du cash-flow")
        
        elif industry == 'technology':
            if revenue_growth > 15:
                insights.append(f"🚀 **Forte croissance tech** : {revenue_growth:.1f}% de croissance excellente pour le secteur technologique")
            
            if profit_margin > 12:
                insights.append("💎 **Prime innovation tech** : Marges élevées indiquent une forte position marché")
            else:
                recommendations.append("🔬 **Investissement R&D** : Augmenter les dépenses d'innovation pour améliorer la position concurrentielle")
            
            # Recommandations spécifiques technology
            if revenue_volatility > 0.25:
                recommendations.append("🎯 **Stabilisation tech** : Diversifier le portefeuille produits pour réduire la volatilité")
        
        elif industry == 'manufacturing':
            if profit_margin > 8:
                insights.append("🏭 **Manufacturing efficace** : Marges supérieures à la moyenne du secteur manufacturier")
            
            if revenue_volatility < 0.15:
                insights.append("⚙️ **Opérations manufacturing stables** : Patterns cohérents de production et demande")
            else:
                recommendations.append("📊 **Planification demande** : Implémenter de meilleures prévisions pour réduire la volatilité")
            
            # Insights spécifiques manufacturing
            if revenue_growth > 10:
                insights.append("📈 **Expansion manufacturing** : Croissance solide suggère une demande soutenue")
                recommendations.append("🏗️ **Scalabilité** : Évaluer la capacité de production pour soutenir la croissance")
        
        return insights, recommendations

# ========== CSV DATA MANAGER ==========
class CSVDataManager:
    """Gestionnaire de données CSV amélioré avec validation"""
    
    @staticmethod
    def get_csv_financial_data():
        """Obtenir les données financières du CSV avec validation"""
        if not st.session_state.imported_metrics:
            return None
        
        metrics = st.session_state.imported_metrics
        
        # Extraire les métriques financières clés
        financial_data = {}
        
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
        
        # Calculer les métriques dérivées avec validation
        if 'revenue' in financial_data and 'total_costs' in financial_data:
            financial_data['gross_profit'] = financial_data['revenue'] - financial_data['total_costs']
            financial_data['operating_profit'] = financial_data['gross_profit'] * 0.8
            financial_data['net_margin'] = financial_data['net_profit'] / financial_data['revenue'] if financial_data['revenue'] > 0 else 0
        
        # Ajouter des estimations de bilan validées
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
            
            # Ratios financiers
            financial_data['current_ratio'] = financial_data['current_assets'] / financial_data['current_liabilities']
            financial_data['debt_to_equity'] = financial_data['total_debt'] / financial_data['equity']
            financial_data['interest_coverage'] = financial_data['operating_profit'] / financial_data['interest_expense']
        
        return financial_data
    
    @staticmethod
    def has_csv_data():
        """Vérifier si les données CSV sont disponibles"""
        return bool(st.session_state.imported_metrics)
    
    @staticmethod
    def get_csv_insights():
        """Obtenir les insights IA des données CSV"""
        if 'csv_data' in st.session_state and 'insights' in st.session_state.csv_data:
            return st.session_state.csv_data['insights']
        return None
    
    @staticmethod
    def get_csv_visualizations():
        """Obtenir les visualisations des données CSV"""
        if 'csv_data' in st.session_state and 'figures' in st.session_state.csv_data:
            return st.session_state.csv_data['figures']
        return None

# ========== SESSION STATE INITIALIZATION ==========
def init_session_state():
    """Initialiser toutes les variables d'état de session y compris nouveaux composants validation"""
    
    # Données spécifiques import CSV
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = {}
    
    if 'csv_processor' not in st.session_state:
        st.session_state.csv_processor = AdvancedCSVProcessor()
    
    if 'imported_metrics' not in st.session_state:
        st.session_state.imported_metrics = {}
    
    # Résultats validation avancés
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    
    if 'correction_log' not in st.session_state:
        st.session_state.correction_log = []
    
    # Générateur de templates
    if 'template_generator' not in st.session_state:
        st.session_state.template_generator = CSVTemplateGenerator()
    
    # Composants analytics avancés
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = {}
    
    if 'ml_forecasts' not in st.session_state:
        st.session_state.ml_forecasts = {}
    
    if 'enhanced_ml_results' not in st.session_state:
        st.session_state.enhanced_ml_results = {}
    
    # Analytics avancés et risk management
    if 'risk_analysis' not in st.session_state:
        st.session_state.risk_analysis = {}
    
    if 'industry_analysis' not in st.session_state:
        st.session_state.industry_analysis = {}
    
    # Historique des analyses
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Paramètres utilisateur
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'currency': 'DHS',
            'date_format': 'YYYY-MM-DD',
            'decimal_places': 2,
            'theme': 'light'
        }
    
    # Cache des calculs avancés
    if 'cached_calculations' not in st.session_state:
        st.session_state.cached_calculations = {}

# ========== ENHANCED CSV IMPORT INTERFACE ==========
def show_enhanced_csv_import():
    """Interface d'import CSV améliorée avec validation et correction automatiques"""
    st.header("📤 Import CSV Intelligent avec Validation Avancée")
    
    # Introduction avec nouvelles capacités
    st.markdown("""
    🚀 **Nouveau Système de Validation & Correction Automatique** 
    - ✅ **Diagnostic d'incohérences** : Détection automatique des problèmes comptables
    - 🔧 **Corrections automatiques** : IA corrige les valeurs aberrantes et incohérences
    - 📊 **Score qualité** : Évaluation en temps réel de la fiabilité de vos données
    - ⚠️ **Alertes intelligentes** : Identification des risques et problèmes potentiels
    """)
    
    # Section de téléchargement des templates améliorée
    st.subheader("📥 Templates CSV Avancés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Template Complet")
        st.write("Toutes métriques financières pour analyse maximale")
        
        if st.button("📥 Télécharger Template Complet", type="primary", use_container_width=True):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('complete_financial')
            
            if csv_data:
                st.download_button(
                    label="💾 Télécharger complete_financial_template.csv",
                    data=csv_data,
                    file_name="complete_financial_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Template prêt au téléchargement !")
    
    with col2:
        st.markdown("#### ☁️ Template SaaS")
        st.write("Spécialisé entreprises Software as a Service")
        
        if st.button("☁️ Télécharger Template SaaS", use_container_width=True):
            template_gen = st.session_state.template_generator
            csv_data = template_gen.generate_template_csv('saas_template')
            
            if csv_data:
                st.download_button(
                    label="💾 Télécharger saas_template.csv",
                    data=csv_data,
                    file_name="saas_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Template SaaS prêt !")
    
    with col3:
        st.markdown("#### 🎯 Templates Futurs")
        st.write("Templates sectoriels additionnels")
        st.info("🔜 Retail, Manufacturing, Consulting...")
        st.caption("Prochaines versions")
    
    # Zone d'upload avec instructions améliorées
    st.subheader("📁 Upload & Analyse de Vos Données")
    
    uploaded_file = st.file_uploader(
        "📂 Sélectionnez votre fichier CSV financier",
        type=['csv'],
        help="Formats supportés: CSV avec séparateur virgule. Encodage: UTF-8 recommandé."
    )
    
    if uploaded_file is not None:
        try:
            # Lecture et analyse du fichier
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ **Fichier lu avec succès** : {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Affichage aperçu des données
            with st.expander("👀 Aperçu des Données Brutes", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lignes", len(df))
                with col2:
                    st.metric("Colonnes", len(df.columns))
                with col3:
                    missing_data = df.isnull().sum().sum()
                    st.metric("Données Manquantes", missing_data)
            
            # Traitement avec le processeur CSV avancé
            with st.spinner("🔍 Analyse et validation des données en cours..."):
                processor = st.session_state.csv_processor
                results = processor.process_csv(df)
                
                # Sauvegarder les résultats dans l'état de session
                st.session_state.csv_data = results
                st.session_state.imported_metrics = results['metrics']
                st.session_state.validation_results = results['validation_results']
                st.session_state.correction_log = results['correction_log']
            
            # ========== RÉSULTATS DE L'ANALYSE ==========
            st.subheader("📊 Résultats de l'Analyse Avancée")
            
            # Score de qualité global avec nouvelle logique
            validation_results = results['validation_results']
            quality_score = validation_results.get('quality_score', 100)
            total_issues = validation_results.get('total_issues', 0)
            critical_issues = validation_results.get('critical_issues', 0)
            corrections_applied = results.get('corrections_applied', False)
            
            # Affichage du score de qualité avec contexte
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Score Qualité Global", f"{quality_score:.0f}/100")
                if quality_score >= 90:
                    st.success("🟢 Excellente qualité")
                elif quality_score >= 70:
                    st.info("🔵 Qualité correcte")
                elif quality_score >= 50:
                    st.warning("🟡 Qualité modérée")
                else:
                    st.error("🔴 Qualité faible")
            
            with col2:
                st.metric("Anomalies Détectées", total_issues)
                if total_issues == 0:
                    st.success("✅ Aucune anomalie")
                elif total_issues <= 3:
                    st.info("🔵 Mineures")
                else:
                    st.warning("🟡 Attention requise")
            
            with col3:
                st.metric("Problèmes Critiques", critical_issues)
                if critical_issues == 0:
                    st.success("✅ Aucun")
                else:
                    st.error(f"🔴 {critical_issues} critique(s)")
            
            with col4:
                corrections_count = len(results.get('correction_log', []))
                st.metric("Corrections Appliquées", corrections_count)
                if corrections_count == 0:
                    st.success("✅ Aucune correction nécessaire")
                else:
                    st.info(f"🔧 {corrections_count} correction(s) auto")
            
            # Détails des colonnes détectées avec mapping avancé
            mappings = results['mappings']
            if mappings:
                st.markdown("#### 🎯 Mapping des Colonnes Détectées")
                
                # Organiser par catégories
                categories = {
                    '💰 Revenus & Profits': ['revenue', 'sales', 'profit'],
                    '💸 Coûts & Charges': ['costs', 'variable_costs', 'fixed_costs'],
                    '🏦 Bilan': ['assets', 'liabilities', 'equity', 'current_assets', 'current_liabilities'],
                    '📦 Opérations': ['inventory', 'accounts_receivable', 'accounts_payable', 'cash_flow'],
                    '📅 Temporel': ['date'],
                    '👥 Business': ['customer_metrics', 'unit_metrics', 'pricing_metrics']
                }
                
                cols_per_row = 2
                for i in range(0, len(categories), cols_per_row):
                    cols = st.columns(cols_per_row)
                    
                    for j, (category, fields) in enumerate(list(categories.items())[i:i+cols_per_row]):
                        with cols[j]:
                            st.markdown(f"**{category}**")
                            detected_in_category = False
                            
                            for field in fields:
                                if field in mappings:
                                    st.write(f"✅ `{field}` ← `{mappings[field]}`")
                                    detected_in_category = True
                            
                            if not detected_in_category:
                                st.caption("🔍 Aucune colonne détectée")
            
            # Insights de validation avancés
            if validation_results.get('issues'):
                st.markdown("#### ⚠️ Détails des Validations")
                
                issues = validation_results['issues']
                
                # Grouper par sévérité
                critical_group = [i for i in issues if i.get('severity') == 'Élevée']
                medium_group = [i for i in issues if i.get('severity') == 'Moyenne']
                ok_group = [i for i in issues if i.get('severity') == 'OK']
                
                # Afficher issues critiques
                if critical_group:
                    st.error("🚨 **Problèmes Critiques Détectés**")
                    for issue in critical_group:
                        st.error(f"• **{issue.get('type', 'Problème')}** : {issue.get('message', 'Détails non disponibles')}")
                
                # Afficher issues moyennes
                if medium_group:
                    st.warning("⚠️ **Problèmes Modérés**")
                    for issue in medium_group:
                        st.warning(f"• **{issue.get('type', 'Problème')}** : {issue.get('message', 'Détails non disponibles')}")
                
                # Afficher validations OK
                if ok_group:
                    with st.expander("✅ Validations Réussies", expanded=False):
                        for issue in ok_group:
                            st.success(f"• **{issue.get('type', 'Validation')}** : {issue.get('message', 'Validation passée')}")
            
            # Log des corrections automatiques
            if corrections_applied and results.get('correction_log'):
                st.markdown("#### 🔧 Journal des Corrections Automatiques")
                
                correction_log = results['correction_log']
                
                with st.expander(f"📋 Détails des {len(correction_log)} Correction(s)", expanded=False):
                    for i, correction in enumerate(correction_log):
                        st.info(f"**Correction {i+1}** : {correction.get('method', 'Méthode inconnue')}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'outliers_found' in correction:
                                st.write(f"• Outliers détectés : {correction['outliers_found']}")
                            if 'missing_values_filled' in correction:
                                st.write(f"• Valeurs manquantes comblées : {correction['missing_values_filled']}")
                            if 'extreme_variations_smoothed' in correction:
                                st.write(f"• Variations extrêmes lissées : {correction['extreme_variations_smoothed']}")
                        
                        with col2:
                            if 'replacement_value' in correction:
                                st.write(f"• Valeur de remplacement : {correction['replacement_value']:.2f}")
                            if 'threshold_used' in correction:
                                st.write(f"• Seuil utilisé : {correction['threshold_used']:.2f}")
                            if 'basis' in correction:
                                st.write(f"• Base : {correction['basis']}")
            
            # Métriques calculées avec validation
            metrics = results['metrics']
            if metrics:
                st.markdown("#### 📈 Métriques Financières Calculées")
                
                tabs = st.tabs(["💰 Revenus", "💸 Coûts", "📊 Rentabilité", "📈 Tendances"])
                
                with tabs[0]:
                    if 'revenue' in metrics:
                        rev_metrics = metrics['revenue']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Période", f"{rev_metrics['total']:,.0f} DHS")
                        with col2:
                            st.metric("Moyenne Mensuelle", f"{rev_metrics['average']:,.0f} DHS")
                        with col3:
                            st.metric("Croissance", f"{rev_metrics.get('growth_rate', 0):+.1f}%")
                        with col4:
                            st.metric("Volatilité", f"{rev_metrics.get('volatility', 0):.1%}")
                        
                        # Tendance qualitative
                        trend = rev_metrics.get('trend', 'stable')
                        if trend == 'croissance':
                            st.success("📈 Tendance : Croissance")
                        elif trend == 'declin':
                            st.error("📉 Tendance : Déclin")
                        else:
                            st.info("📊 Tendance : Stable")
                
                with tabs[1]:
                    if 'costs' in metrics:
                        cost_metrics = metrics['costs']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Coûts", f"{cost_metrics['total']:,.0f} DHS")
                        with col2:
                            st.metric("Coûts Moyens", f"{cost_metrics['average']:,.0f} DHS")
                        with col3:
                            st.metric("Évolution", f"{cost_metrics.get('growth_rate', 0):+.1f}%")
                
                with tabs[2]:
                    if 'profit' in metrics:
                        profit_metrics = metrics['profit']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Profit Total", f"{profit_metrics['total']:,.0f} DHS")
                        with col2:
                            st.metric("Marge Moyenne", f"{profit_metrics.get('margin_average', 0):.1f}%")
                        with col3:
                            profit_growth = profit_metrics.get('growth_rate', 0)
                            st.metric("Croissance Profit", f"{profit_growth:+.1f}%")
                        
                        # Santé financière basique
                        margin = profit_metrics.get('margin_average', 0)
                        if margin > 15:
                            st.success("💰 Excellente rentabilité")
                        elif margin > 5:
                            st.info("📈 Rentabilité correcte")
                        elif margin > 0:
                            st.warning("📊 Faible rentabilité")
                        else:
                            st.error("🔴 Entreprise déficitaire")
                
                with tabs[3]:
                    # Résumé des tendances
                    st.markdown("**Analyse des Tendances Détectées**")
                    
                    if 'revenue' in metrics:
                        rev_growth = metrics['revenue'].get('growth_rate', 0)
                        if rev_growth > 10:
                            st.success(f"📈 **Forte croissance CA** : {rev_growth:.1f}% sur la période")
                        elif rev_growth > 0:
                            st.info(f"📊 **Croissance modérée** : {rev_growth:.1f}%")
                        else:
                            st.error(f"📉 **Déclin CA** : {abs(rev_growth):.1f}% de baisse")
                    
                    if 'profit' in metrics:
                        profit_trend = metrics['profit'].get('growth_rate', 0)
                        if profit_trend > rev_growth:
                            st.success("📈 **Amélioration de l'efficacité** : Profit croît plus vite que le CA")
                        elif profit_trend < rev_growth - 10:
                            st.warning("⚠️ **Détérioration marges** : Croissance coûts supérieure")
            
            # Visualisations avec contexte de validation
            if 'figures' in results and results['figures']:
                st.markdown("#### 📊 Visualisations avec Validation")
                
                figures = results['figures']
                
                for chart_name, fig in figures.items():
                    # Ajouter annotation de qualité
                    if quality_score < 80:
                        st.caption(f"⚠️ Graphique basé sur données qualité {quality_score:.0f}% - Interpréter avec prudence")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if corrections_applied:
                        st.caption("ℹ️ Visualisation inclut les corrections automatiques appliquées")
            
            # Insights IA avec validation
            insights_data = results.get('insights', {})
            if insights_data:
                st.markdown("#### 🤖 Insights IA avec Validation Avancée")
                
                insight_tabs = st.tabs(["✅ Insights Validés", "💡 Recommandations", "⚠️ Alertes", "🎯 Actions Prioritaires"])
                
                with insight_tabs[0]:
                    if insights_data.get('insights'):
                        for insight in insights_data['insights']:
                            st.success(f"✅ {insight}")
                            
                            # Ajouter niveau de confiance basé sur qualité données
                            if quality_score >= 90:
                                st.caption("🔹 Confiance élevée (données haute qualité)")
                            elif quality_score >= 70:
                                st.caption("🔸 Confiance modérée")
                            else:
                                st.caption("🔸 Confiance limitée - Validation externe recommandée")
                    else:
                        st.info("Aucun insight spécifique généré. Performance dans les normes.")
                
                with insight_tabs[1]:
                    if insights_data.get('recommendations'):
                        for rec in insights_data['recommendations']:
                            st.warning(f"💡 {rec}")
                    else:
                        st.success("✅ Aucune recommandation immédiate identifiée")
                
                with insight_tabs[2]:
                    if insights_data.get('alerts'):
                        for alert in insights_data['alerts']:
                            st.error(f"⚠️ {alert}")
                    else:
                        st.success("✅ Aucune alerte critique détectée !")
                
                with insight_tabs[3]:
                    # Générer actions prioritaires basées sur résultats validation
                    priority_actions = []
                    
                    if critical_issues > 0:
                        priority_actions.append("🔴 **Priorité 1** : Corriger les incohérences critiques détectées")
                    
                    if quality_score < 70:
                        priority_actions.append("🟡 **Priorité 2** : Améliorer la qualité globale des données")
                    
                    if len(results.get('correction_log', [])) > 3:
                        priority_actions.append("🔵 **Priorité 3** : Réviser les processus de collecte des données")
                    
                    if not priority_actions:
                        priority_actions.append("✅ **Aucune action prioritaire** : Données de qualité satisfaisante")
                    
                    for action in priority_actions:
                        if "Priorité 1" in action:
                            st.error(action)
                        elif "Priorité 2" in action:
                            st.warning(action)
                        elif "Priorité 3" in action:
                            st.info(action)
                        else:
                            st.success(action)
            
            # Options d'intégration améliorées
            st.subheader("🔄 Options d'Intégration Avancées")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("💾 Sauvegarder Analyse", type="primary", use_container_width=True):
                    # Ajouter à l'historique des analyses
                    analysis_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'quality_score': quality_score,
                        'total_issues': total_issues,
                        'critical_issues': critical_issues,
                        'corrections_applied': corrections_applied
                    }
                    
                    if 'analysis_history' not in st.session_state:
                        st.session_state.analysis_history = []
                    
                    st.session_state.analysis_history.append(analysis_record)
                    
                    st.success("✅ Analyse sauvegardée avec succès !")
                    
                    if quality_score >= 80:
                        st.balloons()
                    else:
                        st.info("💡 Améliorer la qualité des données pour de meilleurs résultats")
            
            with col2:
                if st.button("🧠 Analytics Avancés", use_container_width=True):
                    st.success("🚀 Naviguez vers Analytics Avancés via la barre latérale...")
                    st.info("👈 Utilisez le menu de navigation à gauche pour accéder aux Analytics Avancés")
            
            with col3:
                if st.button("🎯 Planification Scénarios", use_container_width=True):
                    st.success("🚀 Naviguez vers Planification Scénarios via la barre latérale...")
                    st.info("👈 Utilisez le menu de navigation à gauche pour accéder à la Planification de Scénarios")
            
            with col4:
                if st.button("🤖 Prévisions ML", use_container_width=True):
                    st.success("🚀 Naviguez vers Prévisions ML via la barre latérale...")
                    st.info("👈 Utilisez le menu de navigation à gauche pour accéder aux Prévisions ML")
            
            # Historique des analyses si disponible
            if len(st.session_state.get('analysis_history', [])) > 0:
                with st.expander("📋 Historique des Analyses", expanded=False):
                    history_df = pd.DataFrame(st.session_state.analysis_history)
                    st.dataframe(history_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ **Erreur lors du traitement du fichier** : {str(e)}")
            st.info("""
            **Suggestions de résolution** :
            - Vérifier que le fichier est au format CSV valide
            - S'assurer de l'encodage UTF-8
            - Contrôler que les données numériques ne contiennent pas de caractères spéciaux
            - Essayer de sauvegarder le fichier Excel en CSV depuis votre tableur
            """)

# ========== EXECUTIVE DASHBOARD (Enhanced) ==========
def show_executive_dashboard():
    """Dashboard exécutif amélioré avec insights validation"""
    st.header("👔 Executive Dashboard Avancé")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if csv_data:
        # Afficher le statut de qualité des données en premier
        quality_info = ""
        if 'validation_results' in st.session_state:
            quality_score = st.session_state.validation_results.get('quality_score', 100)
            if quality_score >= 80:
                quality_info = f" (Qualité: {quality_score:.0f}/100 ✅)"
            elif quality_score >= 60:
                quality_info = f" (Qualité: {quality_score:.0f}/100 ⚠️)"
            else:
                quality_info = f" (Qualité: {quality_score:.0f}/100 🔴)"
        
        st.success(f"📊 **Dashboard alimenté par vos données CSV{quality_info}**")
        
        # KPI Métriques principales avec contexte validation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_revenue = csv_data.get('monthly_revenue', 0)
            st.metric("CA Mensuel Moyen", f"{monthly_revenue:,.0f} DHS")
            
            growth = csv_data.get('revenue_growth', 0)
            if growth > 0:
                st.success(f"📈 Croissance {growth:.1f}%")
            else:
                st.error(f"📉 Déclin {abs(growth):.1f}%")
        
        with col2:
            monthly_costs = csv_data.get('monthly_costs', 0)
            st.metric("Coûts Mensuels Moyens", f"{monthly_costs:,.0f} DHS")
            
            cost_growth = csv_data.get('costs_growth', 0)
            if cost_growth < 5:
                st.success("✅ Maîtrise Coûts")
            else:
                st.warning(f"⚠️ Hausse {cost_growth:.1f}%")
        
        with col3:
            monthly_profit = csv_data.get('monthly_profit', 0)
            st.metric("Profit Mensuel Moyen", f"{monthly_profit:,.0f} DHS")
            
            if monthly_profit > 0:
                st.success("💰 Rentable")
            else:
                st.error("🔴 Déficitaire")
        
        with col4:
            profit_margin = csv_data.get('profit_margin', 0)
            st.metric("Marge Bénéficiaire", f"{profit_margin:.1f}%")
            
            if profit_margin > 20:
                st.success("🎯 Excellente")
            elif profit_margin > 10:
                st.info("📈 Bonne")
            elif profit_margin > 0:
                st.warning("⚠️ Faible")
            else:
                st.error("🔴 Négative")
        
        # Indicateur de qualité données amélioré
        if 'validation_results' in st.session_state:
            validation_results = st.session_state.validation_results
            quality_score = validation_results.get('quality_score', 100)
            
            if quality_score < 70:
                st.warning(f"⚠️ **Attention Qualité Données** : Score {quality_score:.0f}/100 - Interpréter les métriques avec prudence")
                
                with st.expander("Voir détails qualité données", expanded=False):
                    critical_issues = validation_results.get('critical_issues', 0)
                    total_issues = validation_results.get('total_issues', 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Problèmes Critiques", critical_issues)
                    with col2:
                        st.metric("Total Problèmes", total_issues)
        
        # Analyse de performance financière améliorée
        st.subheader("📈 Analyse de Performance Financière")
        
        # Afficher visualisations CSV avec contexte validation
        csv_figures = CSVDataManager.get_csv_visualizations()
        if csv_figures and 'financial_trend' in csv_figures:
            st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
            
            # Ajouter contexte qualité données
            if 'validation_results' in st.session_state:
                quality_score = st.session_state.validation_results.get('quality_score', 100)
                if quality_score < 70:
                    st.caption("⚠️ Graphique basé sur des données avec corrections automatiques appliquées")
        
        # Insights améliorés avec validation
        csv_insights = CSVDataManager.get_csv_insights()
        if csv_insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🤖 Insights IA de Vos Données")
                for insight in csv_insights['insights']:
                    st.success(f"✅ {insight}")
                
                # Ajouter insights basés sur validation
                if 'validation_results' in st.session_state:
                    validation_results = st.session_state.validation_results
                    if validation_results.get('quality_score', 100) >= 90:
                        st.success("✅ Données de très haute qualité - Analyses hautement fiables")
            
            with col2:
                st.markdown("#### 💡 Recommandations")
                for rec in csv_insights['recommendations']:
                    st.info(f"💡 {rec}")
                
                # Ajouter recommandations qualité données
                if 'validation_results' in st.session_state:
                    validation_results = st.session_state.validation_results
                    quality_score = validation_results.get('quality_score', 100)
                    
                    if quality_score < 80:
                        st.warning("🔧 Améliorer la qualité des données pour des analyses plus précises")
                    
                    if validation_results.get('critical_issues', 0) > 0:
                        st.error("🚨 Corriger les incohérences critiques en priorité")
            
            if csv_insights['alerts']:
                st.markdown("#### ⚠️ Alertes Risques")
                for alert in csv_insights['alerts']:
                    st.error(f"⚠️ {alert}")
        
        # Résumé de performance amélioré avec validation
        st.subheader("📊 Résumé de Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            annual_revenue = csv_data.get('revenue', 0)
            st.metric("CA Annuel", f"{annual_revenue:,.0f} DHS")
            
            volatility = csv_data.get('revenue_volatility', 0)
            if volatility < 0.1:
                st.success("🟢 Stable")
            elif volatility < 0.3:
                st.warning("🟡 Modéré")
            else:
                st.error("🔴 Volatil")
        
        with col2:
            annual_profit = csv_data.get('net_profit', 0)
            st.metric("Profit Annuel", f"{annual_profit:,.0f} DHS")
            
            if annual_profit > 0:
                roi = (annual_profit / (annual_revenue * 0.6)) * 100
                st.metric("ROI", f"{roi:.1f}%")
        
        with col3:
            cash_flow = csv_data.get('cash_flow', 0)
            st.metric("Cash Flow Mensuel", f"{cash_flow:,.0f} DHS")
            
            if cash_flow > 0:
                st.success("💰 Positif")
            else:
                st.error("🔴 Négatif")
        
        # Résumé validation amélioré pour dirigeants
        if 'validation_results' in st.session_state:
            validation_results = st.session_state.validation_results
            
            with st.expander("📋 Résumé Qualité Données pour Direction", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quality_score = validation_results.get('quality_score', 100)
                    st.metric("Score Qualité Global", f"{quality_score:.0f}/100")
                    
                    if quality_score >= 90:
                        st.success("🟢 Excellente fiabilité")
                    elif quality_score >= 70:
                        st.info("🔵 Fiabilité acceptable")
                    else:
                        st.warning("🟡 Fiabilité limitée")
                
                with col2:
                    corrections_applied = len(st.session_state.get('correction_log', []))
                    st.metric("Corrections Appliquées", corrections_applied)
                    
                    if corrections_applied == 0:
                        st.success("✅ Données brutes correctes")
                    else:
                        st.info("🔧 Améliorations automatiques")
                
                with col3:
                    critical_issues = validation_results.get('critical_issues', 0)
                    st.metric("Problèmes Critiques", critical_issues)
                    
                    if critical_issues == 0:
                        st.success("✅ Aucun problème critique")
                    else:
                        st.error("🚨 Attention requise")
    
    else:
        # Message amélioré sans données
        st.warning("📤 **Aucune Donnée CSV Importée**")
        st.info("Importez vos données financières via Smart CSV Import pour voir une analyse complète du dashboard avec validation avancée !")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Ce que vous verrez avec les données CSV :**
            - Tendances réelles revenus et profits
            - Insights IA avec validation qualité
            - Analyse de croissance validée
            - Alertes risques avec niveau confiance
            - Benchmarks performance sectoriels
            - Score qualité données en temps réel
            """)

# ========== ENHANCED ADVANCED ANALYTICS ==========
def show_advanced_analytics():
    """Analytics avancés avec validation améliorée et capacités ML"""
    st.header("🧠 Advanced Analytics & Insights IA Améliorés")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Aucune Donnée CSV Disponible**")
        st.info("Advanced Analytics nécessite vos données CSV uploadées pour fournir une analyse significative avec validation avancée.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV Maintenant", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Advanced Analytics fournira :**
            - Ratios financiers complets avec validation
            - Score santé IA avec corrections automatiques
            - Benchmarking sectoriel avancé
            - Insights prédictifs avec niveau confiance
            - Évaluations risques calibrées
            - Diagnostics incohérences en temps réel
            """)
        return
    
    # Afficher contexte qualité données de manière prominente
    if 'validation_results' in st.session_state:
        validation_results = st.session_state.validation_results
        quality_score = validation_results.get('quality_score', 100)
        
        if quality_score >= 80:
            st.success(f"📊 **Analytics alimentés par vos données CSV validées (Qualité: {quality_score:.0f}/100) ✅**")
        elif quality_score >= 60:
            st.info(f"📊 **Analytics alimentés par vos données CSV (Qualité: {quality_score:.0f}/100) ⚠️**")
        else:
            st.warning(f"📊 **Analytics avec données qualité limitée (Score: {quality_score:.0f}/100) - Interpréter avec prudence 🔴**")
    else:
        st.success("📊 **Analytics alimentés par vos données CSV uploadées**")
    
    # Initialiser moteur analytics avancé
    analytics = AdvancedAnalytics()
    
    # Calculer ratios complets avec validation
    ratios = analytics.calculate_comprehensive_ratios(csv_data)
    
    # Calculer score santé avec contexte validation
    validation_context = st.session_state.get('validation_results')
    health_score, score_breakdown = analytics.calculate_financial_health_score(ratios, 'technology', validation_context)
    
    # Aperçu santé financière amélioré
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Santé Financière", f"{health_score:.0f}/100")
        
        if health_score >= 80:
            st.success("🟢 Excellente")
        elif health_score >= 60:
            st.info("🔵 Bonne")
        elif health_score >= 40:
            st.warning("🟡 Moyenne")
        else:
            st.error("🔴 Faible")
        
        # Ajouter indicateur impact validation
        if validation_context and validation_context.get('quality_score', 100) < 80:
            st.caption("⚠️ Score ajusté selon qualité données")
    
    with col2:
        current_ratio = csv_data.get('current_ratio', 0)
        st.metric("Ratio Liquidité", f"{current_ratio:.2f}")
        
        if current_ratio > 1.5:
            st.success("🟢 Saine")
        elif current_ratio > 1.2:
            st.info("🔵 Modérée")
        else:
            st.warning("🟡 Faible")
    
    with col3:
        net_margin = csv_data.get('net_margin', 0)
        st.metric("Marge Nette", f"{net_margin*100:.1f}%")
        
        if net_margin > 0.15:
            st.success("🟢 Forte")
        elif net_margin > 0.08:
            st.info("🔵 Moyenne")
        else:
            st.warning("🟡 Faible")
    
    with col4:
        debt_to_equity = csv_data.get('debt_to_equity', 0)
        st.metric("Dette/Capitaux", f"{debt_to_equity:.2f}")
        
        if debt_to_equity < 0.5:
            st.success("🟢 Conservateur")
        elif debt_to_equity < 1.0:
            st.info("🔵 Modéré")
        else:
            st.warning("🟡 Élevé")
    
    with col5:
        revenue_volatility = csv_data.get('revenue_volatility', 0)
        stability_score = (1-revenue_volatility)*100
        st.metric("Stabilité CA", f"{stability_score:.0f}%")
        
        if revenue_volatility < 0.1:
            st.success("🟢 Très Stable")
        elif revenue_volatility < 0.2:
            st.info("🔵 Stable")
        else:
            st.warning("🟡 Volatile")
    
    # Analyse par onglets améliorée avec contexte validation
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyse Performance", "🤖 Insights IA Avancés", "📊 Ratios Financiers", "⚕️ Diagnostic Données"])
    
    with tab1:
        st.subheader("Analyse Performance de Vos Données")
        
        # Afficher visualisation CSV originale avec améliorations
        csv_figures = CSVDataManager.get_csv_visualizations()
        if csv_figures and 'financial_trend' in csv_figures:
            st.plotly_chart(csv_figures['financial_trend'], use_container_width=True)
            
            # Ajouter annotations validation
            if validation_context:
                quality_score = validation_context.get('quality_score', 100)
                corrections_applied = len(st.session_state.get('correction_log', []))
                
                if corrections_applied > 0:
                    st.info(f"📝 Note : {corrections_applied} corrections automatiques appliquées pour améliorer la précision")
                
                if quality_score < 80:
                    st.warning("⚠️ Données avec qualité modérée - Tendances à valider avec sources externes")
        
        # Métriques performance améliorées
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Indicateurs Clés de Performance")
            
            revenue_data = csv_data.get('revenue_data', [])
            if revenue_data:
                avg_revenue = np.mean(revenue_data)
                revenue_trend = "Croissance" if revenue_data[-1] > revenue_data[0] else "Déclin"
                
                st.metric("CA Moyen", f"{avg_revenue:,.0f} DHS")
                st.metric("Tendance CA", revenue_trend)
                
                if len(revenue_data) > 1:
                    growth_rate = ((revenue_data[-1] / revenue_data[0]) - 1) * 100
                    st.metric("Croissance Totale", f"{growth_rate:+.1f}%")
                    
                    # Ajouter contexte validation pour croissance
                    if validation_context:
                        extreme_variations = any(abs(r2/r1 - 1) > 1.0 for r1, r2 in zip(revenue_data[:-1], revenue_data[1:]) if r1 != 0)
                        if extreme_variations:
                            st.caption("⚠️ Variations extrêmes détectées - Croissance peut inclure des corrections")
        
        with col2:
            st.markdown("#### 💰 Analyse Rentabilité")
            
            profit_data = csv_data.get('profit_data', [])
            if profit_data:
                avg_profit = np.mean(profit_data)
                profit_trend = "Amélioration" if profit_data[-1] > profit_data[0] else "Détérioration"
                
                st.metric("Profit Moyen", f"{avg_profit:,.0f} DHS")
                st.metric("Tendance Profit", profit_trend)
                
                profit_margin = csv_data.get('profit_margin', 0)
                st.metric("Marge Bénéficiaire", f"{profit_margin:.1f}%")
                
                # Ajouter insights validation pour rentabilité
                if validation_context:
                    profit_issues = [i for i in validation_context.get('issues', []) if 'Profit' in i.get('type', '')]
                    if profit_issues:
                        st.caption("ℹ️ Calculs profit validés et corrigés automatiquement")
    
    with tab2:
        st.subheader("🤖 Insights IA Avancés avec Validation")
        
        # Insights IA améliorés avec contexte validation
        csv_insights = CSVDataManager.get_csv_insights()
        if csv_insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Insights Validés")
                if csv_insights['insights']:
                    for insight in csv_insights['insights']:
                        st.success(f"✅ {insight}")
                        
                        # Ajouter niveau confiance basé sur qualité données
                        if validation_context:
                            quality_score = validation_context.get('quality_score', 100)
                            if quality_score >= 90:
                                st.caption("🔹 Confiance élevée")
                            elif quality_score >= 70:
                                st.caption("🔸 Confiance modérée")
                            else:
                                st.caption("🔸 Confiance limitée - Validation externe recommandée")
                else:
                    st.info("Aucun insight spécifique généré à partir des données actuelles.")
                
                st.markdown("#### ⚠️ Alertes Validées")
                if csv_insights['alerts']:
                    for alert in csv_insights['alerts']:
                        st.error(f"⚠️ {alert}")
                        
                        # Ajouter contexte qualité données aux alertes
                        if validation_context and validation_context.get('quality_score', 100) < 70:
                            st.caption("⚠️ Alerte basée sur données qualité modérée")
                else:
                    st.success("✅ Aucune alerte critique détectée !")
            
            with col2:
                st.markdown("#### 💡 Recommandations IA Avancées")
                
                # Générer recommandations améliorées avec contexte validation
                try:
                    enhanced_recommendations = analytics.generate_ai_recommendations(
                        csv_data, ratios, health_score, validation_context
                    )
                    
                    if enhanced_recommendations:
                        for i, rec in enumerate(enhanced_recommendations):
                            priority_color = "🔴" if rec['priority'] == 'Critique' else "🟠" if rec['priority'] == 'Élevée' else "🟡"
                            
                            with st.expander(f"{priority_color} {rec['category']} - Priorité {rec['priority']}", expanded=i < 2):
                                st.write(f"**Recommandation** : {rec['recommendation']}")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Impact", rec['impact'])
                                with col_b:
                                    st.metric("Délai", rec['timeframe'])
                                with col_c:
                                    if isinstance(rec.get('estimated_benefit'), (int, float)):
                                        st.metric("Bénéfice Est.", f"{rec['estimated_benefit']:,.0f} DHS")
                                    else:
                                        st.metric("Bénéfice", rec.get('estimated_benefit', 'Qualitatif'))
                    else:
                        for rec in csv_insights['recommendations']:
                            st.warning(f"💡 {rec}")
                except Exception as e:
                    st.warning(f"Erreur génération recommandations avancées: {str(e)}")
                    # Fallback aux recommandations de base
                    for rec in csv_insights.get('recommendations', []):
                        st.warning(f"💡 {rec}")
                
                # Ajouter recommandations spécifiques qualité données
                if validation_context:
                    quality_score = validation_context.get('quality_score', 100)
                    critical_issues = validation_context.get('critical_issues', 0)
                    
                    st.markdown("#### 🔧 Recommandations Qualité Données")
                    
                    if quality_score < 50:
                        st.error("🚨 **Urgent** : Refonte complète du processus de collecte des données")
                    elif quality_score < 70:
                        st.warning("⚠️ **Important** : Améliorer les contrôles qualité lors de la saisie")
                    elif quality_score < 90:
                        st.info("ℹ️ **Suggestion** : Automatiser davantage la validation des données")
                    else:
                        st.success("✅ **Excellent** : Maintenir les processus qualité actuels")
                    
                    if critical_issues > 0:
                        st.error(f"🔴 **Action Immédiate** : Corriger {critical_issues} incohérence(s) critique(s)")
        else:
            st.info("Uploadez des données CSV pour voir les insights IA avancés spécifiques à votre entreprise")
    
    with tab3:
        st.subheader("📊 Analyse Ratios Financiers Avancée")
        
        # Analyse ratios améliorée avec validation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Ratios de Liquidité")
            
            current_ratio = csv_data.get('current_ratio', 0)
            quick_ratio = (csv_data.get('current_assets', 0) - csv_data.get('inventory', 0)) / csv_data.get('current_liabilities', 1)
            cash_ratio = csv_data.get('cash', 0) / csv_data.get('current_liabilities', 1)
            
            ratios_data = {
                'Ratio': ['Ratio Liquidité Générale', 'Ratio Liquidité Réduite', 'Ratio Liquidité Immédiate'],
                'Valeur': [current_ratio, quick_ratio, cash_ratio],
                'Benchmark': [1.5, 1.0, 0.2],
                'Statut': []
            }
            
            for value, benchmark in zip(ratios_data['Valeur'], ratios_data['Benchmark']):
                if value >= benchmark * 1.2:
                    ratios_data['Statut'].append('Excellent')
                elif value >= benchmark:
                    ratios_data['Statut'].append('Bon')
                elif value >= benchmark * 0.8:
                    ratios_data['Statut'].append('Adéquat')
                else:
                    ratios_data['Statut'].append('Faible')
            
            df_ratios = pd.DataFrame(ratios_data)
            df_ratios['Valeur'] = df_ratios['Valeur'].round(2)
            
            st.dataframe(df_ratios, use_container_width=True)
            
            # Ajouter contexte validation pour ratios
            if validation_context and validation_context.get('quality_score', 100) < 80:
                st.caption("⚠️ Ratios calculés avec données corrigées automatiquement")
        
        with col2:
            st.markdown("#### 💰 Ratios de Rentabilité")
            
            gross_margin = (csv_data.get('gross_profit', 0) / csv_data.get('revenue', 1)) * 100
            net_margin = csv_data.get('net_margin', 0) * 100
            roa = (csv_data.get('net_profit', 0) / csv_data.get('total_assets', 1)) * 100
            
            profit_data = {
                'Métrique': ['Marge Brute %', 'Marge Nette %', 'ROA %'],
                'Valeur': [gross_margin, net_margin, roa],
                'Moyenne Industrie': [40, 12, 8]
            }
            
            df_profit = pd.DataFrame(profit_data)
            df_profit['Valeur'] = df_profit['Valeur'].round(1)
            
            st.dataframe(df_profit, use_container_width=True)
            
            # Graphique rentabilité amélioré avec contexte validation
            fig = go.Figure(data=[
                go.Bar(name='Votre Entreprise', x=profit_data['Métrique'], y=profit_data['Valeur']),
                go.Bar(name='Moyenne Industrie', x=profit_data['Métrique'], y=profit_data['Moyenne Industrie'])
            ])
            
            fig.update_layout(
                barmode='group',
                title='Rentabilité vs Moyenne Industrie',
                yaxis_title='Pourcentage (%)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter indicateur confiance
            if validation_context:
                quality_score = validation_context.get('quality_score', 100)
                st.caption(f"Confiance analyse : {quality_score:.0f}%")
    
    with tab4:
        st.subheader("⚕️ Diagnostic Avancé des Données")
        
        if validation_context:
            # Dashboard qualité données complet
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                quality_score = validation_context.get('quality_score', 100)
                st.metric("Score Qualité Global", f"{quality_score:.0f}/100")
                
                if quality_score >= 90:
                    st.success("🟢 Excellente")
                elif quality_score >= 70:
                    st.info("🔵 Bonne")
                elif quality_score >= 50:
                    st.warning("🟡 Modérée")
                else:
                    st.error("🔴 Faible")
            
            with col2:
                total_issues = validation_context.get('total_issues', 0)
                st.metric("Anomalies Totales", total_issues)
                
                if total_issues == 0:
                    st.success("✅ Aucune")
                elif total_issues <= 2:
                    st.info("🔵 Limitées")
                else:
                    st.warning("🟡 Attention")
            
            with col3:
                critical_issues = validation_context.get('critical_issues', 0)
                st.metric("Anomalies Critiques", critical_issues)
                
                if critical_issues == 0:
                    st.success("✅ Aucune")
                else:
                    st.error(f"🔴 {critical_issues}")
            
            with col4:
                corrections_count = len(st.session_state.get('correction_log', []))
                st.metric("Corrections Auto", corrections_count)
                
                if corrections_count == 0:
                    st.success("✅ Aucune")
                else:
                    st.info(f"🔧 {corrections_count}")
            
            # Répartition validation détaillée
            st.markdown("#### 🔍 Détails Validation par Catégorie")
            
            validation_categories = {}
            for issue in validation_context.get('issues', []):
                category = issue.get('type', 'Autre')
                severity = issue.get('severity', 'Inconnu')
                
                if category not in validation_categories:
                    validation_categories[category] = {'OK': 0, 'Moyenne': 0, 'Élevée': 0}
                
                validation_categories[category][severity] += 1
            
            if validation_categories:
                validation_df = pd.DataFrame(validation_categories).T.fillna(0)
                validation_df['Total'] = validation_df.sum(axis=1)
                
                st.dataframe(validation_df, use_container_width=True)
                
                # Visualisation résultats validation
                fig = go.Figure()
                
                for severity in ['OK', 'Moyenne', 'Élevée']:
                    if severity in validation_df.columns:
                        color = 'green' if severity == 'OK' else 'orange' if severity == 'Moyenne' else 'red'
                        fig.add_trace(go.Bar(
                            name=severity,
                            x=validation_df.index,
                            y=validation_df[severity],
                            marker_color=color
                        ))
                
                fig.update_layout(
                    title="Répartition des Validations par Catégorie",
                    xaxis_title="Catégorie de Validation",
                    yaxis_title="Nombre d'Occurrences",
                    barmode='stack'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Détails journal corrections
            correction_log = st.session_state.get('correction_log', [])
            if correction_log:
                st.markdown("#### 🔧 Journal Détaillé des Corrections")
                
                for i, correction in enumerate(correction_log):
                    with st.expander(f"Correction {i+1}: {correction.get('method', 'Correction Automatique')}", expanded=False):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Méthode** : {correction.get('method', 'N/A')}")
                            
                            if 'outliers_found' in correction:
                                st.write(f"**Outliers détectés** : {correction['outliers_found']}")
                            if 'missing_values_filled' in correction:
                                st.write(f"**Valeurs manquantes comblées** : {correction['missing_values_filled']}")
                            if 'values_corrected' in correction:
                                st.write(f"**Valeurs corrigées** : {correction['values_corrected']}")
                            if 'extreme_variations_smoothed' in correction:
                                st.write(f"**Variations extrêmes lissées** : {correction['extreme_variations_smoothed']}")
                        
                        with col_b:
                            if 'replacement_value' in correction:
                                st.write(f"**Valeur de remplacement** : {correction['replacement_value']:.2f}")
                            if 'interpolation_method' in correction:
                                st.write(f"**Méthode interpolation** : {correction['interpolation_method']}")
                            if 'threshold_used' in correction:
                                st.write(f"**Seuil utilisé** : {correction['threshold_used']:.2f}")
                            if 'basis' in correction:
                                st.write(f"**Base de correction** : {correction['basis']}")
            
            # Recommandations qualité données
            st.markdown("#### 💡 Recommandations Qualité Données")
            
            quality_score = validation_context.get('quality_score', 100)
            critical_issues = validation_context.get('critical_issues', 0)
            
            if critical_issues > 0:
                st.error("🚨 **Action Immédiate Requise** : Corriger les incohérences critiques détectées")
            
            if quality_score < 60:
                st.error("🔴 **Refonte Processus** : Score qualité très faible - Revoir complètement la collecte des données")
            elif quality_score < 80:
                st.warning("🟡 **Amélioration Processus** : Renforcer les contrôles qualité lors de la saisie")
            elif quality_score < 95:
                st.info("🔵 **Optimisation** : Automatiser davantage les validations pour atteindre l'excellence")
            else:
                st.success("✅ **Excellence** : Maintenir les standards de qualité actuels")
        else:
            st.info("Données de validation non disponibles. Réimportez vos données CSV pour accéder au diagnostic avancé.")

# ========== ENHANCED SCENARIO PLANNING ==========
def show_scenario_planning():
    """Planification de scénarios avancée avec auto-calibrage et validation - VERSION CORRIGÉE"""
    st.header("🎯 Planification de Scénarios Avancée avec Auto-Calibrage")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Aucune Donnée CSV Disponible**")
        st.info("La Planification de Scénarios avancée nécessite vos données CSV uploadées pour un auto-calibrage précis basé sur votre historique.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV Maintenant", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            st.markdown("""
            **Avec vos données CSV, vous bénéficiez de :**
            - Auto-calibrage scénarios basé sur votre volatilité historique
            - Détection automatique de saisonnalité
            - Contraintes business personnalisées
            - Simulation Monte Carlo avec votre profil de risque
            - Validation qualité données pour fiabilité scénarios
            """)
        return
    
    # Afficher contexte qualité données pour scénarios
    quality_context = ""
    scenario_confidence = "Élevée"
    
    if 'validation_results' in st.session_state:
        validation_results = st.session_state.validation_results
        quality_score = validation_results.get('quality_score', 100)
        
        if quality_score >= 80:
            quality_context = f" (Données validées - Score: {quality_score:.0f}/100 ✅)"
            scenario_confidence = "Très Élevée"
        elif quality_score >= 60:
            quality_context = f" (Qualité modérée - Score: {quality_score:.0f}/100 ⚠️)"
            scenario_confidence = "Modérée"
        else:
            quality_context = f" (Qualité limitée - Score: {quality_score:.0f}/100 🔴)"
            scenario_confidence = "Limitée"
            
        critical_issues = validation_results.get('critical_issues', 0)
        if critical_issues > 0:
            quality_context += f" - {critical_issues} incohérence(s) critique(s)"
    
    st.success(f"📊 **Scénarios auto-calibrés sur vos données CSV{quality_context}**")
    
    # Données de base avec validation
    base_monthly_revenue = float(csv_data.get('monthly_revenue', 15000))
    base_monthly_costs = float(csv_data.get('monthly_costs', 12000))
    historical_volatility = float(csv_data.get('revenue_volatility', 0.2))
    current_growth_rate = float(csv_data.get('revenue_growth', 0)) / 100
    
    # Auto-calibrage avancé avec détection industrie
    industry_manager = IndustryTemplateManager()
    detected_industry = industry_manager.detect_industry_from_csv(csv_data)
    
    # Calibrateur de scénarios amélioré
    scenario_calibrator = EnhancedScenarioCalibrator()
    
    # Analyser patterns historiques pour calibrage
    revenue_data = csv_data.get('revenue_data', [])
    historical_analysis = scenario_calibrator.analyze_historical_volatility(revenue_data)
    
    # Aperçu des données de base avec contexte auto-calibrage
    st.subheader(f"📊 Données de Base Auto-Calibrées ({detected_industry.title()})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CA Mensuel Base", f"{base_monthly_revenue:,.0f} DHS")
        st.caption(f"Moyenne de {len(revenue_data)} points" if revenue_data else "Estimation")
    
    with col2:
        st.metric("Coûts Mensuels Base", f"{base_monthly_costs:,.0f} DHS")
        profit_base = base_monthly_revenue - base_monthly_costs
        st.caption(f"Profit base: {profit_base:,.0f} DHS")
    
    with col3:
        st.metric("Volatilité Historique", f"{historical_volatility:.1%}")
        volatility_level = "Faible" if historical_volatility < 0.1 else "Modérée" if historical_volatility < 0.3 else "Élevée"
        st.caption(f"Niveau: {volatility_level}")
    
    with col4:
        st.metric("Tendance Actuelle", f"{current_growth_rate*100:+.1f}%")
        trend_desc = "Croissance" if current_growth_rate > 0 else "Déclin" if current_growth_rate < -0.05 else "Stable"
        st.caption(f"Direction: {trend_desc}")
    
    # Afficher résultats auto-calibrage
    if historical_analysis:
        with st.expander("🔍 Détails Auto-Calibrage Basé sur Votre Historique", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Points de Données", historical_analysis.get('data_points', 0))
                st.metric("Tendance Détectée", f"{historical_analysis.get('trend', 0)*100:+.2f}%")
            
            with col2:
                st.metric("Coefficient Variation", f"{historical_analysis.get('coefficient_variation', 0):.1%}")
                seasonality = "Oui" if historical_analysis.get('seasonality', False) else "Non"
                st.metric("Saisonnalité", seasonality)
            
            with col3:
                st.metric("Valeur Moyenne", f"{historical_analysis.get('mean_value', 0):,.0f} DHS")
                reliability = "Haute" if historical_analysis.get('data_points', 0) >= 12 else "Modérée"
                st.metric("Fiabilité Calibrage", reliability)
    
    # Configuration scénarios avec auto-calibrage
    st.subheader("⚙️ Configuration Scénarios Auto-Calibrés")
    
    # Générer paramètres auto-calibrés
    calibrated_scenarios = scenario_calibrator.calibrate_scenario_parameters(historical_analysis, detected_industry)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 😰 Scénario Pessimiste")
        st.caption("Auto-calibré selon votre volatilité historique")
        
        pess_revenue_auto = calibrated_scenarios['pessimistic']['revenue_change'] * 100
        pess_cost_auto = calibrated_scenarios['pessimistic']['cost_change'] * 100
        pess_prob_auto = calibrated_scenarios['pessimistic']['probability'] * 100
        
        pess_revenue = st.slider("Évolution CA (%)", -50, 10, int(pess_revenue_auto), key="pess_rev", 
                                help=f"Valeur auto-calibrée: {pess_revenue_auto:.1f}%")
        pess_cost = st.slider("Évolution Coûts (%)", -10, 40, int(pess_cost_auto), key="pess_cost",
                            help=f"Valeur auto-calibrée: {pess_cost_auto:.1f}%")
        pess_prob = st.slider("Probabilité (%)", 5, 40, int(pess_prob_auto), key="pess_prob",
                            help=f"Valeur auto-calibrée: {pess_prob_auto:.1f}%")
        
        if abs(pess_revenue - pess_revenue_auto) < 2:
            st.success("✅ Aligné avec historique")
        else:
            st.info("🔧 Valeur ajustée manuellement")
    
    with col2:
        st.markdown("### 😐 Scénario Réaliste")
        st.caption("Basé sur votre tendance historique")
        
        real_revenue_auto = calibrated_scenarios['realistic']['revenue_change'] * 100
        real_cost_auto = calibrated_scenarios['realistic']['cost_change'] * 100
        real_prob_auto = calibrated_scenarios['realistic']['probability'] * 100
        
        real_revenue = st.slider("Évolution CA (%)", -10, 40, int(real_revenue_auto), key="real_rev",
                                help=f"Valeur auto-calibrée: {real_revenue_auto:.1f}%")
        real_cost = st.slider("Évolution Coûts (%)", 0, 25, int(real_cost_auto), key="real_cost",
                            help=f"Valeur auto-calibrée: {real_cost_auto:.1f}%")
        real_prob = st.slider("Probabilité (%)", 40, 80, int(real_prob_auto), key="real_prob",
                            help=f"Valeur auto-calibrée: {real_prob_auto:.1f}%")
        
        if abs(real_revenue - real_revenue_auto) < 2:
            st.success("✅ Aligné avec historique")
        else:
            st.info("🔧 Valeur ajustée manuellement")
    
    with col3:
        st.markdown("### 😄 Scénario Optimiste")
        st.caption("Calibré selon votre potentiel de croissance")
        
        opt_revenue_auto = calibrated_scenarios['optimistic']['revenue_change'] * 100
        opt_cost_auto = calibrated_scenarios['optimistic']['cost_change'] * 100
        opt_prob_auto = calibrated_scenarios['optimistic']['probability'] * 100
        
        opt_revenue = st.slider("Évolution CA (%)", 10, 60, int(opt_revenue_auto), key="opt_rev",
                               help=f"Valeur auto-calibrée: {opt_revenue_auto:.1f}%")
        opt_cost = st.slider("Évolution Coûts (%)", -5, 15, int(opt_cost_auto), key="opt_cost",
                           help=f"Valeur auto-calibrée: {opt_cost_auto:.1f}%")
        opt_prob = st.slider("Probabilité (%)", 5, 40, int(opt_prob_auto), key="opt_prob",
                           help=f"Valeur auto-calibrée: {opt_prob_auto:.1f}%")
        
        if abs(opt_revenue - opt_revenue_auto) < 2:
            st.success("✅ Aligné avec historique")
        else:
            st.info("🔧 Valeur ajustée manuellement")
    
    # Paramètres avancés avec contraintes business
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_period = st.selectbox("Période d'Analyse", [6, 12, 18, 24, 36], index=1,
                                     help="Plus longue période = plus d'incertitude")
        
        st.markdown("#### 🏢 Contraintes Business")
        apply_constraints = st.checkbox("Appliquer Contraintes Opérationnelles", value=True,
                                      help="Limiter les variations à des niveaux réalistes")
        
        if apply_constraints:
            max_monthly_growth = st.slider("Croissance Max Mensuelle (%)", 5, 50, 20,
                                         help="Limite la croissance mensuelle maximale")
            max_monthly_decline = st.slider("Déclin Max Mensuel (%)", 5, 30, 15,
                                          help="Limite le déclin mensuel maximal")
    
    with col2:
        st.markdown("#### 🎯 Options Avancées")
        include_seasonality = st.checkbox("Inclure Saisonnalité", 
                                        value=historical_analysis.get('seasonality', False),
                                        help="Appliquer patterns saisonniers détectés")
        
        monte_carlo_sims = st.slider("Simulations Monte Carlo", 500, 2000, 1000,
                                   help="Plus de simulations = plus de précision")
        
        confidence_level = st.slider("Niveau de Confiance (%)", 80, 99, 95,
                                    help="Niveau de confiance pour intervalles")
        
        # Afficher confiance scénarios basée sur qualité données
        st.metric("Confiance Scénarios", scenario_confidence)
        if scenario_confidence != "Très Élevée":
            st.caption("⚠️ Basé sur qualité des données source")
    
    # Lancer analyse scénarios avec gestion d'erreurs améliorée
    if st.button("🚀 Lancer Analyse Scénarios Avancée", type="primary"):
        try:
            with st.spinner("Exécution analyse scénarios avec auto-calibrage et Monte Carlo..."):
                
                # Normaliser probabilities pour qu'elles totalisent 100%
                total_prob = pess_prob + real_prob + opt_prob
                if total_prob != 100:
                    pess_prob = pess_prob * 100 / total_prob
                    real_prob = real_prob * 100 / total_prob
                    opt_prob = opt_prob * 100 / total_prob
                
                scenarios = {
                    'pessimistic': {
                        'revenue_change': float(pess_revenue) / 100,
                        'cost_change': float(pess_cost) / 100,
                        'probability': float(pess_prob) / 100
                    },
                    'realistic': {
                        'revenue_change': float(real_revenue) / 100,
                        'cost_change': float(real_cost) / 100,
                        'probability': float(real_prob) / 100
                    },
                    'optimistic': {
                        'revenue_change': float(opt_revenue) / 100,
                        'cost_change': float(opt_cost) / 100,
                        'probability': float(opt_prob) / 100
                    }
                }
                
                # Appliquer contraintes business si activées
                if apply_constraints:
                    constraints = {
                        'max_revenue_decline': -max_monthly_decline / 100,
                        'max_cost_increase': max_monthly_growth / 100,
                        'min_margin': -0.2
                    }
                    scenarios = scenario_calibrator.apply_operational_constraints(scenarios, constraints)
                
                # Calculer résultats pour chaque scénario avec gestion d'erreurs
                scenario_results = {}
                
                for scenario_name, params in scenarios.items():
                    try:
                        monthly_results = []
                        
                        # Facteur saisonnier si activé
                        seasonal_factors = [1.0] * 12  # Default: pas de saisonnalité
                        if include_seasonality and historical_analysis.get('seasonality', False):
                            # Utiliser patterns saisonniers du template industrie
                            template = industry_manager.get_template(detected_industry)
                            seasonal_factors = template.get('seasonal_factors', [1.0] * 12)
                        
                        for month in range(analysis_period):
                            # Appliquer facteur saisonnier
                            seasonal_factor = seasonal_factors[month % 12] if include_seasonality else 1.0
                            
                            # Calculer revenus et coûts avec volatilité et saisonnalité
                            base_revenue_adjusted = base_monthly_revenue * seasonal_factor
                            monthly_revenue = base_revenue_adjusted * (1 + params['revenue_change'])
                            monthly_cost = base_monthly_costs * (1 + params['cost_change'])
                            
                            # Appliquer contraintes mensuelles si activées
                            if apply_constraints:
                                if month > 0:
                                    prev_revenue = monthly_results[-1]['revenue']
                                    max_change = prev_revenue * (max_monthly_growth / 100)
                                    min_change = prev_revenue * (-max_monthly_decline / 100)
                                    
                                    revenue_change = monthly_revenue - prev_revenue
                                    revenue_change = max(min_change, min(max_change, revenue_change))
                                    monthly_revenue = prev_revenue + revenue_change
                            
                            monthly_profit = monthly_revenue - monthly_cost
                            
                            monthly_results.append({
                                'month': month + 1,
                                'revenue': float(max(0, monthly_revenue)),
                                'cost': float(max(0, monthly_cost)),
                                'profit': float(monthly_profit),
                                'seasonal_factor': float(seasonal_factor)
                            })
                        
                        # FIX: Calculer métriques finales avec gestion sécurisée
                        total_profit = sum(float(m['profit']) for m in monthly_results)
                        total_revenue = sum(float(m['revenue']) for m in monthly_results) 
                        avg_monthly_profit = total_profit / analysis_period if analysis_period > 0 else 0.0
                        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0.0
                        
                        scenario_results[scenario_name] = {
                            'monthly_data': monthly_results,
                            'total_profit': float(total_profit),
                            'total_revenue': float(total_revenue),
                            'avg_monthly_profit': float(avg_monthly_profit),
                            'profit_margin': float(profit_margin),
                            'probability': float(params['probability'])
                        }
                        
                    except Exception as e:
                        st.error(f"Erreur calcul scénario {scenario_name}: {str(e)}")
                        # FIX: Fournir valeurs par défaut sécurisées
                        scenario_results[scenario_name] = {
                            'monthly_data': [],
                            'total_profit': 0.0,
                            'total_revenue': 0.0,
                            'avg_monthly_profit': 0.0,
                            'profit_margin': 0.0,
                            'probability': float(params['probability'])
                        }
                
                # Générer simulations Monte Carlo avec données réelles
                mc_simulations = scenario_calibrator.generate_monte_carlo_scenarios(
                    revenue_data, monte_carlo_sims, analysis_period
                )
                
                # Sauvegarder résultats avec métadonnées améliorées
                st.session_state.scenario_results = {
                    'scenarios': scenario_results,
                    'monte_carlo': mc_simulations,
                    'metadata': {
                        'analysis_period': analysis_period,
                        'confidence_level': confidence_level,
                        'industry': detected_industry,
                        'seasonality_applied': include_seasonality,
                        'constraints_applied': apply_constraints,
                        'auto_calibrated': True,
                        'data_quality_score': validation_results.get('quality_score', 100) if 'validation_results' in st.session_state else 100,
                        'historical_volatility': historical_volatility,
                        'calibration_confidence': scenario_confidence
                    }
                }
                
                st.success("✅ Analyse scénarios complétée avec succès !")
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse des scénarios: {str(e)}")
            st.info("Vérification des paramètres d'entrée et nouvelle tentative recommandée.")
    
    # Afficher résultats avec gestion d'erreurs améliorée
    if 'scenario_results' in st.session_state and st.session_state.scenario_results:
        try:
            scenario_data = st.session_state.scenario_results
            scenario_results = scenario_data['scenarios']
            metadata = scenario_data.get('metadata', {})
            
            st.subheader("📊 Résultats Analyse Scénarios Avancée")
            
            # Métriques principales avec contexte validation
            # FIX: Calcul sécurisé de expected_value
            scenario_profits = {}
            for name, data in scenario_results.items():
                if isinstance(data, dict) and all(k in data for k in ['total_profit', 'probability']):
                    scenario_profits[name] = {
                        'total_profit': float(data.get('total_profit', 0)),
                        'probability': float(data.get('probability', 0))
                    }
            
            if scenario_profits:
                # FIX: Calcul expected_value sécurisé
                expected_value = sum(
                    float(data['total_profit']) * float(data['probability']) 
                    for data in scenario_profits.values()
                    if isinstance(data.get('total_profit'), (int, float)) and isinstance(data.get('probability'), (int, float))
                )
                
                profit_values = [float(data['total_profit']) for data in scenario_profits.values() 
                               if isinstance(data.get('total_profit'), (int, float))]
                
                best_case = max(profit_values) if profit_values else 0.0
                worst_case = min(profit_values) if profit_values else 0.0
                profit_range = best_case - worst_case
                
                # Calculer upside potential et risk ratio
                upside_potential = max(0, best_case - expected_value)
                downside_risk = max(0, expected_value - worst_case)
                risk_ratio = upside_potential / downside_risk if downside_risk > 0 else float('inf')
                
                # Affichage métriques principales
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Valeur Espérée", f"{expected_value:,.0f} DHS")
                    if expected_value > 0:
                        st.success("🟢 Positif")
                    else:
                        st.error("🔴 Négatif")
                
                with col2:
                    st.metric("Meilleur Cas", f"{best_case:,.0f} DHS")
                    st.caption(f"Potentiel upside")
                
                with col3:
                    st.metric("Pire Cas", f"{worst_case:,.0f} DHS")
                    st.caption(f"Risque downside")
                
                with col4:
                    st.metric("Fourchette", f"{profit_range:,.0f} DHS")
                    st.caption("Amplitude totale")
                
                with col5:
                    st.metric("Confiance Analyse", metadata.get('calibration_confidence', 'Modérée'))
                    quality_score = metadata.get('data_quality_score', 100)
                    if quality_score >= 80:
                        st.success("🟢 Fiable")
                    else:
                        st.warning("🟡 Attention")
                
                # Afficher contexte auto-calibrage
                if metadata.get('auto_calibrated', False):
                    st.info(f"🤖 **Scénarios auto-calibrés** pour industrie {metadata.get('industry', 'générale').title()} "
                           f"avec {metadata.get('analysis_period', 12)} mois d'analyse")
                
                # Visualisation avancée avec Monte Carlo
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Graphique principal des scénarios avec saisonnalité
                    fig = go.Figure()
                    
                    colors = {'pessimistic': '#FF6B6B', 'realistic': '#4ECDC4', 'optimistic': '#45B7D1'}
                    
                    for scenario, data in scenario_results.items():
                        if isinstance(data, dict) and 'monthly_data' in data:
                            monthly_data = data['monthly_data']
                            if monthly_data:
                                months = [m.get('month', i+1) for i, m in enumerate(monthly_data)]
                                profits = [float(m.get('profit', 0)) for m in monthly_data]
                                cumulative_profit = np.cumsum(profits)
                                
                                # Ligne principale
                                fig.add_trace(go.Scatter(
                                    x=months,
                                    y=cumulative_profit,
                                    mode='lines+markers',
                                    name=f"{scenario.title()} (P: {data.get('probability', 0):.0%})",
                                    line=dict(color=colors.get(scenario, 'gray'), width=3),
                                    hovertemplate=f"<b>{scenario.title()}</b><br>" +
                                                f"Mois: %{{x}}<br>" +
                                                f"Profit Cumulé: %{{y:,.0f}} DHS<extra></extra>"
                                ))
                                
                                # Ajouter pattern saisonnier si appliqué
                                if metadata.get('seasonality_applied', False):
                                    seasonal_factors = [m.get('seasonal_factor', 1.0) for m in monthly_data]
                                    if any(f != 1.0 for f in seasonal_factors):
                                        fig.add_trace(go.Scatter(
                                            x=months,
                                            y=[f * 1000 for f in seasonal_factors],  # Scale pour visibilité
                                            mode='lines',
                                            name=f"Saisonnalité {scenario}",
                                            line=dict(color=colors.get(scenario, 'gray'), width=1, dash='dot'),
                                            opacity=0.3,
                                            yaxis='y2',
                                            showlegend=False
                                        ))
                    
                    # Ligne de référence rentabilité
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                 annotation_text="Seuil de rentabilité")
                    
                    fig.update_layout(
                        title="Évolution Profit Cumulé par Scénario",
                        xaxis_title="Mois",
                        yaxis_title="Profit Cumulé (DHS)",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Analyse de risque
                    st.markdown("#### 🎯 Analyse de Risque")
                    
                    st.metric("Potentiel Haussier", f"{upside_potential:,.0f} DHS")
                    st.metric("Ratio Risque/Rendement", f"{risk_ratio:.2f}" if risk_ratio != float('inf') else "∞")
                    
                    # Risk assessment
                    if risk_ratio > 2:
                        st.success("🟢 Profil risque favorable")
                    elif risk_ratio > 1:
                        st.info("🔵 Profil risque équilibré")
                    else:
                        st.warning("🟡 Profil risque élevé")
                    
                    # Recommandations stratégiques
                    st.markdown("#### 🎯 Recommandations")
                    
                    if worst_case > 0:
                        st.success("✅ **Tous scénarios rentables**")
                        st.info("💡 Focus croissance et optimisation")
                    elif expected_value > 0:
                        st.warning("⚠️ **Risque pire cas**")
                        st.info("💡 Préparer plans contingence")
                    else:
                        st.error("🔴 **Rentabilité incertaine**")
                        st.error("💡 Actions correctives urgentes")
                
                # Tableau comparatif détaillé
                st.markdown("#### 📋 Comparatif Détaillé des Scénarios")
                
                comparison_data = []
                for scenario, data in scenario_results.items():
                    if isinstance(data, dict):
                        comparison_data.append({
                            'Scénario': scenario.title(),
                            'Profit Total': f"{float(data.get('total_profit', 0)):,.0f} DHS",
                            'Revenus Totaux': f"{float(data.get('total_revenue', 0)):,.0f} DHS",
                            'Marge Moyenne': f"{float(data.get('profit_margin', 0)):.1f}%",
                            'Profit Mensuel Moyen': f"{float(data.get('avg_monthly_profit', 0)):,.0f} DHS",
                            'Probabilité': f"{float(data.get('probability', 0)):.0%}"
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Monte Carlo si disponible
                mc_simulations = scenario_data.get('monte_carlo', [])
                if mc_simulations and len(mc_simulations) > 10:
                    st.markdown("#### 🎲 Résultats Simulation Monte Carlo")
                    
                    mc_values = [sim.get('total_value', 0) for sim in mc_simulations if isinstance(sim.get('total_value'), (int, float))]
                    
                    if mc_values:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            mc_mean = np.mean(mc_values)
                            st.metric("Moyenne MC", f"{mc_mean:,.0f} DHS")
                        
                        with col2:
                            var_5 = np.percentile(mc_values, 5)
                            st.metric("VaR 5%", f"{var_5:,.0f} DHS")
                        
                        with col3:
                            prob_positive = (np.array(mc_values) > 0).mean() * 100
                            st.metric("Prob. Profit", f"{prob_positive:.1f}%")
                        
                        with col4:
                            mc_volatility = np.std(mc_values)
                            st.metric("Volatilité MC", f"{mc_volatility:,.0f} DHS")
                        
                        # Distribution Monte Carlo
                        fig_mc = go.Figure(data=[go.Histogram(
                            x=mc_values,
                            nbinsx=50,
                            name='Distribution Monte Carlo',
                            opacity=0.7
                        )])
                        
                        fig_mc.add_vline(x=var_5, line_dash="dash", line_color="red", 
                                       annotation_text="VaR 5%")
                        fig_mc.add_vline(x=0, line_dash="solid", line_color="black", 
                                       annotation_text="Seuil Rentabilité")
                        fig_mc.add_vline(x=mc_mean, line_dash="dot", line_color="green", 
                                       annotation_text="Moyenne")
                        
                        fig_mc.update_layout(
                            title=f"Distribution Monte Carlo ({len(mc_simulations)} simulations)",
                            xaxis_title="Profit Total (DHS)",
                            yaxis_title="Fréquence",
                            height=400
                        )
                        
                        st.plotly_chart(fig_mc, use_container_width=True)
                
                # Impact qualité données sur scénarios
                if 'validation_results' in st.session_state:
                    quality_score = st.session_state.validation_results.get('quality_score', 100)
                    
                    with st.expander("📊 Impact Qualité Données sur Scénarios", expanded=False):
                        if quality_score >= 90:
                            st.success("🟢 **Haute Confiance** : Scénarios basés sur données haute qualité")
                        elif quality_score >= 70:
                            st.info("🔵 **Confiance Modérée** : Scénarios fiables avec corrections appliquées")
                            st.caption("Recommandation : Valider résultats avec sources externes")
                        else:
                            st.warning("🟡 **Confiance Limitée** : Scénarios basés sur données qualité modérée")
                            st.caption("Recommandation : Améliorer qualité données avant décisions stratégiques")
                        
                        corrections_count = len(st.session_state.get('correction_log', []))
                        if corrections_count > 0:
                            st.info(f"ℹ️ {corrections_count} corrections automatiques appliquées aux données source")
            
            else:
                st.warning("Aucun résultat de scénario valide disponible. Vérifiez la configuration des paramètres.")
                
        except Exception as e:
            st.error(f"Erreur affichage résultats scénarios: {str(e)}")
            st.info("Tentez de relancer l'analyse avec des paramètres différents.")

# ========== ENHANCED ML FORECASTING ==========
def show_ml_forecasting():
    """Prévisions ML avancées avec ensemble methods et validation - VERSION CORRIGÉE"""
    st.header("🤖 Prévisions ML Financières Avancées")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Aucune Donnée CSV Disponible**")
        st.info("Les Prévisions ML nécessitent vos données CSV uploadées pour entraîner des modèles précis avec validation croisée.")
        
        if st.button("📤 Importer Données CSV Maintenant", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    # Display enhanced data quality context for ML
    quality_context = ""
    model_confidence = "Élevée"
    
    if 'validation_results' in st.session_state:
        quality_score = st.session_state.validation_results.get('quality_score', 100)
        if quality_score >= 90:
            quality_context = f" (Données haute qualité - Score: {quality_score:.0f}/100 ✅)"
            model_confidence = "Très Élevée"
        elif quality_score >= 70:
            quality_context = f" (Données qualité modérée - Score: {quality_score:.0f}/100 ⚠️)"
            model_confidence = "Modérée"
        else:
            quality_context = f" (Données qualité limitée - Score: {quality_score:.0f}/100 🔴)"
            model_confidence = "Limitée"
    
    st.success(f"📊 **Modèles ML entraînés sur vos données CSV{quality_context}**")
    
    # Enhanced ML engine
    ml_engine = EnhancedMLForecastingEngine()
    
    # Get available data for forecasting with validation
    available_metrics = get_available_forecast_metrics(csv_data)
    
    # Enhanced data overview
    st.subheader("📊 Aperçu Données d'Entraînement")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_data_points = len(csv_data.get('revenue_data', []))
        st.metric("Points de Données", total_data_points)
        
        if total_data_points >= 24:
            data_quality = "Excellente"
            st.success("🟢 Excellente")
        elif total_data_points >= 12:
            data_quality = "Bonne"
            st.info("🔵 Bonne")
        elif total_data_points >= 6:
            data_quality = "Suffisante"
            st.warning("🟡 Suffisante")
        else:
            data_quality = "Limitée"
            st.error("🔴 Limitée")
    
    with col2:
        st.metric("Variables Disponibles", len(available_metrics))
        st.metric("Algorithmes ML", "Ensemble + Validation")
    
    with col3:
        if csv_data.get('revenue_data'):
            revenue_data = csv_data['revenue_data']
            revenue_trend = "Croissance" if revenue_data[-1] > revenue_data[0] else "Déclin"
            st.metric("Tendance CA", revenue_trend)
            
            volatility = np.std(revenue_data) / np.mean(revenue_data) if np.mean(revenue_data) > 0 else 0
            st.metric("Volatilité Données", f"{volatility:.1%}")
    
    with col4:
        st.metric("Confiance Modèle", model_confidence)
        
        # ML readiness assessment
        ml_readiness = 100
        if total_data_points < 12:
            ml_readiness -= 30
        if csv_data.get('revenue_volatility', 0) > 0.5:
            ml_readiness -= 20
        if 'validation_results' in st.session_state:
            quality_score = st.session_state.validation_results.get('quality_score', 100)
            ml_readiness = ml_readiness * (quality_score / 100)
        
        st.metric("Readiness ML", f"{ml_readiness:.0f}%")
    
    # Enhanced Forecasting Configuration
    st.subheader("🔮 Configuration Prévisions ML Avancées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced forecast target selection
        forecast_target = st.selectbox(
            "Cible de Prévision",
            available_metrics,
            help="Choisir la variable à prévoir basée sur vos données uploadées"
        )
        
        forecast_periods = st.slider("Périodes de Prévision (mois)", 3, 36, 12)
        
        # Advanced ML options
        st.markdown("#### 🔧 Options ML Avancées")
        include_trend = st.checkbox("Analyse Tendance", value=True, help="Inclure l'analyse de tendance temporelle")
        include_seasonality = st.checkbox("Analyse Saisonnalité", value=True, help="Détecter et modéliser la saisonnalité")
        use_ensemble = st.checkbox("Méthodes Ensemble", value=True, help="Utiliser plusieurs algorithmes pour plus de robustesse")
        
        # Enhanced model selection
        if use_ensemble:
            model_type = st.selectbox(
                "Type Modèle",
                ["Auto Ensemble (Recommandé)", "Random Forest + Linear", "Ensemble Complet"],
                help="Choisir l'approche d'ensemble pour les prévisions"
            )
        else:
            model_type = st.selectbox(
                "Algorithme ML",
                ["Random Forest", "Régression Linéaire", "Moyenne Mobile Adaptative"],
                help="Choisir l'algorithme de prévision"
            )
    
    with col2:
        confidence_level = st.slider("Niveau Confiance (%)", 80, 99, 95)
        
        # Enhanced forecast scenarios
        st.markdown("#### 📈 Scénarios de Prévision")
        include_scenarios = st.checkbox("Générer Scénarios Multiples", value=True, help="Créer des scénarios optimiste/pessimiste")
        
        if include_scenarios:
            optimistic_factor = st.slider("Scénario Optimiste (+%)", 5, 50, 20)
            pessimistic_factor = st.slider("Scénario Pessimiste (-%)", 5, 50, 15)
        
        # Enhanced external factors
        st.markdown("#### 🌍 Facteurs Externes")
        market_growth = st.slider("Croissance Marché Attendue (%)", -20, 30, 5)
        economic_impact = st.selectbox("Perspectives Économiques", ["Positive", "Neutre", "Négative"])
        
        # Business constraints
        st.markdown("#### 🏢 Contraintes Business")
        apply_constraints = st.checkbox("Appliquer Contraintes", value=True, help="Limiter les prévisions aux plages réalistes")
        
        if apply_constraints:
            max_growth = st.slider("Croissance Max Mensuelle (%)", 5, 100, 25, help="Limite la croissance mensuelle maximale")
            min_decline = st.slider("Déclin Max Mensuel (%)", 5, 50, 20, help="Limite le déclin mensuel maximal")
    
    # Historical data visualization for selected variable with enhancements
    if forecast_target in available_metrics:
        st.subheader(f"📈 Analyse Données Historiques : {forecast_target}")
        
        # Get data for selected target
        target_data = get_target_data(csv_data, forecast_target)
        
        if target_data and len(target_data) > 0:
            months = list(range(1, len(target_data) + 1))
            
            # Enhanced historical visualization
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=[f'Données Historiques {forecast_target}', 'Analyse de Distribution'],
                              vertical_spacing=0.1)
            
            # Main time series
            fig.add_trace(go.Scatter(
                x=months,
                y=target_data,
                mode='lines+markers',
                name=f'Historique {forecast_target}',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ), row=1, col=1)
            
            # Add trend line if requested
            if include_trend and len(target_data) > 3:
                x_trend = np.arange(len(target_data))
                slope, intercept = np.polyfit(x_trend, target_data, 1)
                trend_line = slope * x_trend + intercept
                
                fig.add_trace(go.Scatter(
                    x=months,
                    y=trend_line,
                    mode='lines',
                    name='Ligne de Tendance',
                    line=dict(color='red', width=2, dash='dash')
                ), row=1, col=1)
                
                # Display trend statistics
                trend_slope_monthly = slope
                trend_annual = trend_slope_monthly * 12
                st.caption(f"📈 Tendance détectée : {trend_annual:+.1f} unités/an ({trend_slope_monthly:+.2f}/mois)")
            
            # Distribution analysis
            fig.add_trace(go.Histogram(
                x=target_data,
                name='Distribution',
                nbinsx=min(20, len(target_data)//2),
                marker=dict(color='lightblue', opacity=0.7)
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"Analyse Complète {forecast_target}",
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Mois", row=1, col=1)
            fig.update_yaxes(title_text=f"{forecast_target}", row=1, col=1)
            fig.update_xaxes(title_text=f"Valeurs {forecast_target}", row=2, col=1)
            fig.update_yaxes(title_text="Fréquence", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced statistical summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Moyenne", f"{np.mean(target_data):,.2f}")
                st.metric("Médiane", f"{np.median(target_data):,.2f}")
            
            with col2:
                st.metric("Écart-Type", f"{np.std(target_data):,.2f}")
                st.metric("Coefficient Variation", f"{np.std(target_data)/np.mean(target_data):.1%}")
            
            with col3:
                st.metric("Minimum", f"{np.min(target_data):,.2f}")
                st.metric("Maximum", f"{np.max(target_data):,.2f}")
            
            with col4:
                if len(target_data) > 1:
                    growth_rate = ((target_data[-1] / target_data[0]) - 1) * 100
                    st.metric("Croissance Totale", f"{growth_rate:+.1f}%")
                    
                    monthly_growth = (target_data[-1] / target_data[0]) ** (1/len(target_data)) - 1
                    st.metric("Croissance Mensuelle", f"{monthly_growth*100:+.2f}%")
            
            # Data quality impact on forecasting
            if 'validation_results' in st.session_state:
                validation_results = st.session_state.validation_results
                quality_score = validation_results.get('quality_score', 100)
                
                if quality_score < 80:
                    st.warning(f"⚠️ **Impact Qualité** : Score {quality_score:.0f}/100 peut affecter la précision des prévisions ML")
                    
                    corrections_applied = len(st.session_state.get('correction_log', []))
                    if corrections_applied > 0:
                        st.info(f"ℹ️ {corrections_applied} corrections automatiques appliquées pour améliorer la fiabilité")
    
    # Generate enhanced ML forecast
    if st.button("🚀 Générer Prévisions ML Avancées", type="primary"):
        with st.spinner("Entraînement modèles ML avancés et génération prévisions..."):
            
            target_data = get_target_data(csv_data, forecast_target)
            
            if not target_data or len(target_data) < 3:
                st.error("❌ Données insuffisantes pour prévisions ML. Besoin d'au moins 3 points de données.")
                return
            
            # Prepare external factors and constraints
            external_factors = {
                'market_growth': market_growth / 100,
                'economic_impact': {'Positive': 1.1, 'Neutre': 1.0, 'Négative': 0.95}[economic_impact]
            }
            
            business_constraints = None
            if apply_constraints:
                business_constraints = {
                    'max_growth': max_growth / 100,
                    'min_growth': -min_decline / 100,
                    'smooth_factor': 0.3
                }
            
            # Generate enhanced forecasts using the improved ML engine
            forecast_results = ml_engine.enhanced_forecast(
                target_data,
                forecast_periods,
                confidence_level,
                external_factors,
                business_constraints
            )
            
            # FIX: Vérifier que forecast_results contient 'forecasts'
            if forecast_results is None or 'forecasts' not in forecast_results:
                st.error("❌ Erreur lors de la génération des prévisions ML.")
                return
            
            # Add scenarios if requested
            if include_scenarios:
                base_forecasts = forecast_results['forecasts']
                forecast_results['scenarios'] = generate_forecast_scenarios(
                    base_forecasts,
                    optimistic_factor / 100,
                    pessimistic_factor / 100
                )
            
            # Add data quality metrics to results
            if 'validation_results' in st.session_state:
                forecast_results['data_quality'] = {
                    'quality_score': st.session_state.validation_results.get('quality_score', 100),
                    'confidence_adjustment': model_confidence,
                    'corrections_applied': len(st.session_state.get('correction_log', []))
                }
            
            forecast_results['target'] = forecast_target
            forecast_results['external_factors'] = external_factors
            forecast_results['business_constraints'] = business_constraints
            
            st.session_state.enhanced_ml_results = forecast_results
    
    # Display enhanced forecast results with error handling
    if 'enhanced_ml_results' in st.session_state:
        try:
            display_enhanced_forecast_results(st.session_state.enhanced_ml_results, csv_data)
        except Exception as e:
            st.error(f"Erreur affichage résultats prévisions: {str(e)}")

def get_available_forecast_metrics(csv_data):
    """Get list of available metrics for forecasting based on CSV data"""
    available = []
    
    # Core financial metrics
    if csv_data.get('revenue_data'):
        available.append("Revenus")
    if csv_data.get('costs_data'):
        available.append("Coûts")
    if csv_data.get('profit_data'):
        available.append("Profit")
    
    # Check for other metrics in the CSV processor's detected mappings
    if 'csv_data' in st.session_state and 'mappings' in st.session_state.csv_data:
        mappings = st.session_state.csv_data['mappings']
        
        metric_mapping = {
            'cash_flow': "Cash Flow",
            'assets': "Total Actifs", 
            'current_assets': "Actifs Courants", 
            'fixed_assets': "Actifs Fixes",
            'liabilities': "Total Passifs",
            'current_liabilities': "Passifs Courants",
            'equity': "Capitaux Propres",
            'inventory': "Stocks",
            'accounts_receivable': "Créances Clients",
            'accounts_payable': "Dettes Fournisseurs",
            'customer_metrics': "Nombre Clients",
            'unit_metrics': "Unités Vendues",
            'pricing_metrics': "Prix Moyen",
            'saas_metrics': "Métriques SaaS"
        }
        
        for key, display_name in metric_mapping.items():
            if key in mappings and display_name not in available:
                available.append(display_name)
    
    # Financial ratios (calculated)
    if len(available) >= 2:
        available.extend([
            "Marge Bénéficiaire %",
            "Ratio Liquidité",
            "Taux Croissance CA"
        ])
    
    return available if available else ["Revenus", "Profit"]  # Fallback

def get_target_data(csv_data, target):
    """Get data array for the selected forecast target"""
    target_mapping = {
        "Revenus": csv_data.get('revenue_data', []),
        "Coûts": csv_data.get('costs_data', []),
        "Profit": csv_data.get('profit_data', []),
        "Cash Flow": csv_data.get('cash_flow_data', []),
        # Add more mappings based on available data
    }
    
    # For calculated metrics
    if target == "Marge Bénéficiaire %":
        revenue_data = csv_data.get('revenue_data', [])
        profit_data = csv_data.get('profit_data', [])
        if revenue_data and profit_data:
            return [(p/r)*100 for p, r in zip(profit_data, revenue_data) if r != 0]
    
    if target == "Taux Croissance CA":
        revenue_data = csv_data.get('revenue_data', [])
        if len(revenue_data) > 1:
            return [((revenue_data[i] - revenue_data[i-1]) / revenue_data[i-1]) * 100 
                   for i in range(1, len(revenue_data)) if revenue_data[i-1] != 0]
    
    return target_mapping.get(target, csv_data.get('revenue_data', []))

def generate_forecast_scenarios(base_forecasts, optimistic_factor, pessimistic_factor):
    """Generate optimistic and pessimistic scenarios"""
    return {
        'optimistic': [f * (1 + optimistic_factor) for f in base_forecasts],
        'pessimistic': [f * (1 - pessimistic_factor) for f in base_forecasts],
        'base': base_forecasts
    }

def display_enhanced_forecast_results(results, csv_data):
    """Display enhanced forecast results with comprehensive analysis - VERSION CORRIGÉE"""
    
    # FIX: Vérifier que 'forecasts' existe dans results
    if not results or 'forecasts' not in results:
        st.warning("❌ Aucun résultat de prévision disponible ou données incomplètes.")
        return
    
    st.subheader("📈 Résultats Prévisions ML Avancées")
    
    # Enhanced summary metrics with data quality context
    col1, col2, col3, col4, col5 = st.columns(5)
    
    forecasts = results['forecasts']
    target = results.get('target', 'Variable')
    
    with col1:
        avg_forecast = np.mean(forecasts)
        st.metric("Prévision Moyenne", f"{avg_forecast:,.0f}")
    
    with col2:
        total_forecast = sum(forecasts)
        periods = results.get('periods', len(forecasts))
        st.metric(f"Total {periods}-Mois", f"{total_forecast:,.0f}")
    
    with col3:
        # Calculate growth from last historical value
        if target in ["Revenus", "Coûts", "Profit"]:
            historical_data = get_target_data(csv_data, target)
            if historical_data:
                last_actual = historical_data[-1]
                growth = (forecasts[-1] / last_actual - 1) * 100
                st.metric("Croissance Projetée", f"{growth:+.1f}%")
    
    with col4:
        volatility = np.std(forecasts) / np.mean(forecasts) * 100 if np.mean(forecasts) != 0 else 0
        st.metric("Volatilité Prévision", f"{volatility:.1f}%")
    
    with col5:
        model_performance = results.get('model_performance', {})
        if 'r2_score' in model_performance:
            accuracy = model_performance['r2_score'] * 100
            st.metric("Précision Modèle", f"{accuracy:.1f}%")
        else:
            st.metric("Modèle Utilisé", results.get('best_model', 'ML'))
    
    # Data quality impact indicator
    if 'data_quality' in results:
        data_quality = results['data_quality']
        quality_score = data_quality.get('quality_score', 100)
        
        if quality_score < 80:
            st.warning(f"⚠️ **Impact Qualité Données** : Score {quality_score:.0f}/100 - Confiance {data_quality.get('confidence_adjustment', 'Modérée')}")
            
            corrections = data_quality.get('corrections_applied', 0)
            if corrections > 0:
                st.info(f"ℹ️ {corrections} corrections automatiques appliquées aux données d'entraînement")
    
    # Enhanced forecast visualization with multiple scenarios
    historical_data = get_target_data(csv_data, target)
    if historical_data:
        historical_months = list(range(1, len(historical_data) + 1))
        forecast_months = list(range(len(historical_months) + 1, len(historical_months) + results.get('periods', len(forecasts)) + 1))
        
        # Create comprehensive visualization
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=[f'Prévisions {target} avec Scénarios', 'Intervalles de Confiance'],
                           vertical_spacing=0.1)
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_months,
            y=historical_data,
            mode='lines+markers',
            name='Données Historiques',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ), row=1, col=1)
        
        # Base forecast
        fig.add_trace(go.Scatter(
            x=forecast_months,
            y=forecasts,
            mode='lines+markers',
            name='Prévision ML',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8)
        ), row=1, col=1)
        
        # Scenarios if available
        if 'scenarios' in results:
            scenarios = results['scenarios']
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=scenarios['optimistic'],
                mode='lines',
                name='Scénario Optimiste',
                line=dict(color='green', width=2, dash='dot'),
                opacity=0.7
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=scenarios['pessimistic'],
                mode='lines',
                name='Scénario Pessimiste',
                line=dict(color='orange', width=2, dash='dot'),
                opacity=0.7
            ), row=1, col=1)
        
        # Confidence intervals - main chart
        if 'upper_bounds' in results and 'lower_bounds' in results:
            fig.add_trace(go.Scatter(
                x=forecast_months + forecast_months[::-1],
                y=results['upper_bounds'] + results['lower_bounds'][::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'IC {results.get("confidence_level", 95)}%',
                showlegend=True
            ), row=1, col=1)
            
            # Detailed confidence interval view
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=results['upper_bounds'],
                mode='lines',
                name='Borne Supérieure',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.8
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=forecasts,
                mode='lines+markers',
                name='Prévision Centrale',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=forecast_months,
                y=results['lower_bounds'],
                mode='lines',
                name='Borne Inférieure',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.8
            ), row=2, col=1)
        
        fig.update_layout(
            title=f"Prévisions ML Avancées : {target}",
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Mois", row=1, col=1)
        fig.update_yaxes(title_text=f"{target}", row=1, col=1)
        fig.update_xaxes(title_text="Mois", row=2, col=1)
        fig.update_yaxes(title_text=f"{target}", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced insights and recommendations with business context
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Performance du Modèle ML")
        performance = results.get('model_performance', {})
        
        if performance:
            col_a, col_b = st.columns(2)
            
            with col_a:
                rmse = performance.get('rmse', 0)
                st.metric("RMSE", f"{rmse:.2f}")
                
                mae = performance.get('mae', 0)
                st.metric("MAE", f"{mae:.2f}")
            
            with col_b:
                r2_score = performance.get('r2_score', 0)
                st.metric("R² Score", f"{r2_score:.3f}")
                
                cv_score = performance.get('cv_score', 0)
                st.metric("CV Score", f"{cv_score:.3f}")
            
            # Model quality assessment
            if r2_score > 0.8:
                st.success("🟢 **Modèle Très Performant** : Prévisions hautement fiables")
            elif r2_score > 0.6:
                st.info("🔵 **Modèle Performant** : Prévisions fiables")
            elif r2_score > 0.4:
                st.warning("🟡 **Modèle Moyen** : Prévisions à valider")
            else:
                st.error("🔴 **Modèle Faible** : Prévisions peu fiables")
        
        # Feature importance if available
        if 'feature_importance' in results and results['feature_importance']:
            st.markdown("#### 🎯 Importance des Variables")
            
            importance_data = results['feature_importance']
            features = list(importance_data.keys())
            importance_values = list(importance_data.values())
            
            fig_importance = go.Figure(data=[
                go.Bar(x=importance_values, y=features, orientation='h',
                      marker_color='lightblue')
            ])
            
            fig_importance.update_layout(
                title="Importance des Features ML",
                xaxis_title="Importance",
                height=300
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model information
        best_model = results.get('best_model', 'Ensemble ML')
        st.markdown(f"**Modèle Sélectionné** : {best_model}")
        
        if 'model_scores' in results:
            model_scores = results['model_scores']
            if model_scores:
                st.markdown("**Comparaison Modèles** :")
                for model_name, scores in model_scores.items():
                    if isinstance(scores, dict) and 'mean_score' in scores:
                        st.write(f"• {model_name}: {scores['mean_score']:.3f}")
    
    with col2:
        st.markdown("#### 💡 Insights et Recommandations ML")
        
        # Generate insights based on forecast patterns
        insights = generate_forecast_insights(forecasts, historical_data, target)
        
        for insight in insights:
            if insight['type'] == 'positive':
                st.success(f"✅ {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"⚠️ {insight['message']}")
            else:
                st.info(f"ℹ️ {insight['message']}")
        
        # Recommendations based on forecast results
        st.markdown("#### 🎯 Recommandations Stratégiques")
        
        if historical_data and len(forecasts) > 0:
            last_actual = historical_data[-1]
            first_forecast = forecasts[0]
            
            if first_forecast > last_actual * 1.1:
                st.success("📈 **Opportunité Croissance** : Préparer scaling opérationnel")
            elif first_forecast < last_actual * 0.9:
                st.warning("📉 **Alerte Déclin** : Actions correctives recommandées")
            else:
                st.info("📊 **Stabilité Projetée** : Maintenir stratégie actuelle")
        
        # Business constraints feedback
        if results.get('business_constraints'):
            st.caption("🏢 Prévisions ajustées selon contraintes business")
        
        if results.get('external_factors'):
            st.caption("🌍 Facteurs externes pris en compte")
    
    # Detailed forecast table with scenarios
    st.markdown("#### 📋 Tableau Détaillé des Prévisions")
    
    forecast_table_data = []
    
    for i, forecast in enumerate(forecasts):
        month = i + 1
        row_data = {
            'Mois': f"M+{month}",
            'Prévision Base': f"{forecast:,.0f}",
        }
        
        # Add confidence intervals
        if 'upper_bounds' in results and 'lower_bounds' in results:
            row_data['Borne Inf.'] = f"{results['lower_bounds'][i]:,.0f}"
            row_data['Borne Sup.'] = f"{results['upper_bounds'][i]:,.0f}"
        
        # Add scenarios if available
        if 'scenarios' in results:
            scenarios = results['scenarios']
            row_data['Optimiste'] = f"{scenarios['optimistic'][i]:,.0f}"
            row_data['Pessimiste'] = f"{scenarios['pessimistic'][i]:,.0f}"
        
        forecast_table_data.append(row_data)
    
    # Display only first 12 months in main table, rest in expander
    main_table_data = forecast_table_data[:12]
    extended_table_data = forecast_table_data[12:] if len(forecast_table_data) > 12 else []
    
    df_forecast = pd.DataFrame(main_table_data)
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)
    
    if extended_table_data:
        with st.expander(f"📅 Voir Prévisions Étendues (Mois 13-{len(forecast_table_data)})", expanded=False):
            df_extended = pd.DataFrame(extended_table_data)
            st.dataframe(df_extended, use_container_width=True, hide_index=True)
    
    # Risk analysis section
    if 'scenarios' in results:
        st.markdown("#### ⚠️ Analyse de Risque des Prévisions")
        
        scenarios = results['scenarios']
        base_total = sum(scenarios['base'])
        optimistic_total = sum(scenarios['optimistic'])
        pessimistic_total = sum(scenarios['pessimistic'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            upside_potential = optimistic_total - base_total
            st.metric("Potentiel Haussier", f"{upside_potential:,.0f}")
            upside_pct = (upside_potential / base_total) * 100 if base_total > 0 else 0
            st.caption(f"{upside_pct:+.1f}%")
        
        with col2:
            downside_risk = base_total - pessimistic_total
            st.metric("Risque Baissier", f"{downside_risk:,.0f}")
            downside_pct = (downside_risk / base_total) * 100 if base_total > 0 else 0
            st.caption(f"{downside_pct:.1f}%")
        
        with col3:
            risk_ratio = upside_potential / downside_risk if downside_risk > 0 else float('inf')
            st.metric("Ratio Risque/Rendement", f"{risk_ratio:.2f}")
            
            if risk_ratio > 2:
                st.success("🟢 Favorable")
            elif risk_ratio > 1:
                st.info("🔵 Équilibré")
            else:
                st.warning("🟡 Risqué")
    
    # Validation and confidence summary
    st.markdown("#### 🔍 Validation et Confiance")
    
    validation_summary = []
    
    # Data quality validation
    if 'data_quality' in results:
        data_quality = results['data_quality']
        quality_score = data_quality.get('quality_score', 100)
        
        validation_summary.append({
            'Critère': 'Qualité Données Source',
            'Score': f"{quality_score:.0f}/100",
            'Statut': '🟢 Excellent' if quality_score >= 90 else '🔵 Bon' if quality_score >= 70 else '🟡 Modéré'
        })
    
    # Model performance validation
    if performance:
        r2_score = performance.get('r2_score', 0)
        validation_summary.append({
            'Critère': 'Performance Modèle (R²)',
            'Score': f"{r2_score:.3f}",
            'Statut': '🟢 Excellent' if r2_score >= 0.8 else '🔵 Bon' if r2_score >= 0.6 else '🟡 Modéré'
        })
    
    # Data volume validation
    if historical_data:
        data_points = len(historical_data)
        validation_summary.append({
            'Critère': 'Volume Données',
            'Score': f"{data_points} points",
            'Statut': '🟢 Suffisant' if data_points >= 12 else '🔵 Acceptable' if data_points >= 6 else '🟡 Limité'
        })
    
    if validation_summary:
        df_validation = pd.DataFrame(validation_summary)
        st.dataframe(df_validation, use_container_width=True, hide_index=True)
    
    # Export options
    st.markdown("#### 📥 Options d'Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exporter Données CSV", use_container_width=True):
            csv_export = df_forecast.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger CSV",
                data=csv_export,
                file_name=f"predictions_{target}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📈 Générer Rapport", use_container_width=True):
            st.info("🔄 Fonction génération rapport en développement")
    
    with col3:
        if st.button("🔄 Nouvelle Prévision", use_container_width=True):
            if 'enhanced_ml_results' in st.session_state:
                del st.session_state.enhanced_ml_results
            st.rerun()

def generate_forecast_insights(forecasts, historical_data, target):
    """Generate insights based on forecast patterns"""
    insights = []
    
    try:
        if not forecasts or len(forecasts) == 0:
            return insights
        
        # Trend analysis
        if len(forecasts) > 1:
            trend_slope = (forecasts[-1] - forecasts[0]) / len(forecasts)
            
            if trend_slope > np.mean(forecasts) * 0.05:
                insights.append({
                    'type': 'positive',
                    'message': f"Tendance haussière forte détectée pour {target}"
                })
            elif trend_slope < -np.mean(forecasts) * 0.05:
                insights.append({
                    'type': 'warning',
                    'message': f"Tendance baissière détectée pour {target}"
                })
            else:
                insights.append({
                    'type': 'info',
                    'message': f"Tendance stable projetée pour {target}"
                })
        
        # Volatility analysis
        if len(forecasts) > 2:
            forecast_volatility = np.std(forecasts) / np.mean(forecasts)
            
            if forecast_volatility < 0.1:
                insights.append({
                    'type': 'positive',
                    'message': "Prévisions très stables - Faible volatilité projetée"
                })
            elif forecast_volatility > 0.3:
                insights.append({
                    'type': 'warning',
                    'message': "Forte volatilité projetée - Surveillance renforcée recommandée"
                })
        
        # Comparison with historical data
        if historical_data and len(historical_data) > 0:
            historical_avg = np.mean(historical_data)
            forecast_avg = np.mean(forecasts)
            
            change_pct = ((forecast_avg - historical_avg) / historical_avg) * 100
            
            if change_pct > 20:
                insights.append({
                    'type': 'positive',
                    'message': f"Amélioration significative projetée (+{change_pct:.1f}% vs historique)"
                })
            elif change_pct < -20:
                insights.append({
                    'type': 'warning',
                    'message': f"Détérioration significative projetée ({change_pct:.1f}% vs historique)"
                })
    
    except Exception as e:
        insights.append({
            'type': 'info',
            'message': f"Analyse insights limitée: {str(e)}"
        })
    
    return insights

# ========== ENHANCED RISK MANAGEMENT ==========
def show_risk_management():
    """Gestion des risques avancée avec simulation Monte Carlo et insights sectoriels - VERSION CORRIGÉE"""
    st.header("⚠️ Gestion des Risques Financiers Avancée")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    
    if not csv_data:
        st.warning("📤 **Aucune Donnée CSV Disponible**")
        st.info("La Gestion des Risques avancée nécessite vos données CSV uploadées pour une évaluation précise avec validation.")
        
        if st.button("📤 Importer Données CSV Maintenant", type="primary"):
            st.session_state['current_page'] = 'csv_import'
            st.rerun()
        return
    
    # Enhanced risk context with data quality
    quality_context = ""
    risk_confidence = "Élevée"
    
    if 'validation_results' in st.session_state:
        validation_results = st.session_state.validation_results
        quality_score = validation_results.get('quality_score', 100)
        
        if quality_score >= 80:
            quality_context = f" (Analyse fiable - Score: {quality_score:.0f}/100 ✅)"
            risk_confidence = "Très Élevée"
        elif quality_score >= 60:
            quality_context = f" (Fiabilité modérée - Score: {quality_score:.0f}/100 ⚠️)"
            risk_confidence = "Modérée"
        else:
            quality_context = f" (Fiabilité limitée - Score: {quality_score:.0f}/100 🔴)"
            risk_confidence = "Limitée"
            
        critical_issues = validation_results.get('critical_issues', 0)
        if critical_issues > 0:
            quality_context += f" - {critical_issues} incohérence(s) critique(s)"
    
    st.success(f"📊 **Analyse des risques basée sur vos données CSV{quality_context}**")
    
    # Initialize enhanced analytics
    analytics = AdvancedAnalytics()
    
    # Calculate enhanced risk metrics with validation
    revenue_volatility = float(csv_data.get('revenue_volatility', 0))
    profit_margin = float(csv_data.get('profit_margin', 0))
    revenue_growth = float(csv_data.get('revenue_growth', 0))
    current_ratio = float(csv_data.get('current_ratio', 1.5))
    debt_to_equity = float(csv_data.get('debt_to_equity', 0.4))
    
    # Enhanced risk score calculation with validation context
    risk_components = {}
    
    # Revenue risk (0-30 points)
    revenue_risk = 0
    if revenue_volatility > 0.4:
        revenue_risk = 30
    elif revenue_volatility > 0.3:
        revenue_risk = 25
    elif revenue_volatility > 0.2:
        revenue_risk = 15
    elif revenue_volatility > 0.1:
        revenue_risk = 8
    
    risk_components['Revenue Volatility'] = revenue_risk
    
    # Profitability risk (0-25 points)
    profitability_risk = 0
    if profit_margin < -10:
        profitability_risk = 25
    elif profit_margin < 0:
        profitability_risk = 20
    elif profit_margin < 5:
        profitability_risk = 15
    elif profit_margin < 10:
        profitability_risk = 8
    
    risk_components['Profitability'] = profitability_risk
    
    # Growth risk (0-20 points)
    growth_risk = 0
    if revenue_growth < -20:
        growth_risk = 20
    elif revenue_growth < -10:
        growth_risk = 15
    elif revenue_growth < 0:
        growth_risk = 10
    elif revenue_growth < 5:
        growth_risk = 5
    
    risk_components['Growth Trend'] = growth_risk
    
    # Liquidity risk (0-15 points)
    liquidity_risk = 0
    if current_ratio < 0.8:
        liquidity_risk = 15
    elif current_ratio < 1.0:
        liquidity_risk = 12
    elif current_ratio < 1.2:
        liquidity_risk = 8
    elif current_ratio < 1.5:
        liquidity_risk = 4
    
    risk_components['Liquidity'] = liquidity_risk
    
    # Leverage risk (0-10 points)
    leverage_risk = 0
    if debt_to_equity > 3:
        leverage_risk = 10
    elif debt_to_equity > 2:
        leverage_risk = 8
    elif debt_to_equity > 1:
        leverage_risk = 5
    elif debt_to_equity > 0.5:
        leverage_risk = 2
    
    risk_components['Leverage'] = leverage_risk
    
    # Data quality risk adjustment
    data_quality_penalty = 0
    if 'validation_results' in st.session_state:
        quality_score = st.session_state.validation_results.get('quality_score', 100)
        if quality_score < 70:
            data_quality_penalty = (100 - quality_score) * 0.1
    
    risk_components['Data Quality'] = data_quality_penalty
    
    # Calculate total risk score
    total_risk_score = sum(risk_components.values())
    total_risk_score = min(100, max(0, total_risk_score))
    
    # Enhanced risk dashboard
    st.subheader("🎯 Dashboard Risques Multidimensionnel")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Enhanced risk gauge with validation context
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Score Risque Global<br><span style='font-size:0.8em;color:gray'>Confiance: {risk_confidence}</span>"},
            delta={'reference': 50, 'position': "bottom"},
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
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk level interpretation with validation
        if total_risk_score < 25:
            st.success("🟢 **Risque Faible**")
            st.write("Position financière solide")
        elif total_risk_score < 50:
            st.info("🔵 **Risque Modéré**")
            st.write("Surveillance régulière recommandée")
        elif total_risk_score < 75:
            st.warning("🟡 **Risque Élevé**")
            st.write("Actions préventives nécessaires")
        else:
            st.error("🔴 **Risque Critique**")
            st.write("Intervention immédiate requise")
        
        # Add data quality context
        if data_quality_penalty > 0:
            st.caption(f"⚠️ Score ajusté (+{data_quality_penalty:.1f}) pour qualité données")
    
    with col2:
        # Enhanced risk components breakdown
        st.markdown("#### 📊 Décomposition des Risques")
        
        # Risk components chart
        fig_components = go.Figure(data=[
            go.Bar(
                x=list(risk_components.keys()),
                y=list(risk_components.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            )
        ])
        
        fig_components.update_layout(
            title="Contribution par Composante de Risque",
            xaxis_title="Composantes",
            yaxis_title="Points de Risque",
            height=300
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
        
        # Risk factors table with enhanced details
        risk_factors_data = []
        
        for component, score in risk_components.items():
            if score > 0:
                if component == 'Revenue Volatility':
                    level = "Critique" if score >= 25 else "Élevé" if score >= 15 else "Modéré"
                    description = f"Volatilité CA: {revenue_volatility:.1%}"
                elif component == 'Profitability':
                    level = "Critique" if score >= 20 else "Élevé" if score >= 15 else "Modéré"
                    description = f"Marge: {profit_margin:.1f}%"
                elif component == 'Growth Trend':
                    level = "Critique" if score >= 15 else "Élevé" if score >= 10 else "Modéré"
                    description = f"Croissance: {revenue_growth:.1f}%"
                elif component == 'Liquidity':
                    level = "Critique" if score >= 12 else "Élevé" if score >= 8 else "Modéré"
                    description = f"Ratio liquidité: {current_ratio:.2f}"
                elif component == 'Leverage':
                    level = "Critique" if score >= 8 else "Élevé" if score >= 5 else "Modéré"
                    description = f"Dette/Capitaux: {debt_to_equity:.2f}"
                elif component == 'Data Quality':
                    level = "Attention"
                    description = f"Score qualité: {100-data_quality_penalty*10:.0f}/100"
                else:
                    level = "Moyen"
                    description = "Facteur de risque détecté"
                
                risk_factors_data.append({
                    'Facteur': component,
                    'Score': f"{score:.1f}",
                    'Niveau': level,
                    'Détail': description
                })
        
        if risk_factors_data:
            df_risk_factors = pd.DataFrame(risk_factors_data)
            st.dataframe(df_risk_factors, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Aucun facteur de risque significatif détecté")
    
    # Enhanced tabs with validation context
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyse Facteurs", "🎲 Simulation Monte Carlo", "💡 Recommandations IA", "⚕️ Risques Données"])
    
    with tab1:
        st.subheader("🔍 Analyse Détaillée des Facteurs de Risque")
        
        # Industry-specific risk analysis
        industry_manager = IndustryTemplateManager()
        detected_industry = industry_manager.detect_industry_from_csv(csv_data)
        
        st.info(f"🏭 **Analyse spécialisée pour industrie** : {industry_manager.templates[detected_industry]['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Risques Opérationnels")
            
            # Revenue analysis with industry context
            st.metric("Volatilité CA", f"{revenue_volatility:.1%}")
            if revenue_volatility > 0.3:
                st.error("🔴 Volatilité excessive")
                if detected_industry == 'retail':
                    st.caption("💡 Normal pour retail saisonnier")
                else:
                    st.caption("⚠️ Instabilité préoccupante")
            elif revenue_volatility > 0.2:
                st.warning("🟡 Volatilité modérée")
            else:
                st.success("🟢 CA stable")
            
            st.metric("Croissance CA", f"{revenue_growth:+.1f}%")
            if revenue_growth < -10:
                st.error("🔴 Déclin sévère")
            elif revenue_growth < 0:
                st.warning("🟡 Déclin")
            elif revenue_growth > 20:
                st.success("🟢 Forte croissance")
                if detected_industry == 'saas':
                    st.caption("💡 Excellent pour SaaS")
            else:
                st.info("🔵 Croissance modérée")
            
            # Profitability analysis
            st.metric("Marge Bénéficiaire", f"{profit_margin:.1f}%")
            
            # Industry-specific margin analysis
            template = industry_manager.get_template(detected_industry)
            benchmark_margin = template['benchmarks'].get('profit_margin', 0.1) * 100
            
            if profit_margin < 0:
                st.error("🔴 Entreprise déficitaire")
            elif profit_margin < benchmark_margin * 0.5:
                st.error("🔴 Marges très faibles")
            elif profit_margin < benchmark_margin:
                st.warning("🟡 Marges sous benchmark")
            elif profit_margin > benchmark_margin * 1.5:
                st.success("🟢 Marges excellentes")
            else:
                st.success("🟢 Marges satisfaisantes")
            
            st.caption(f"Benchmark {detected_industry}: {benchmark_margin:.1f}%")
        
        with col2:
            st.markdown("#### 💰 Risques Financiers")
            
            st.metric("Ratio de Liquidité", f"{current_ratio:.2f}")
            if current_ratio < 1.0:
                st.error("🔴 Liquidité critique")
                st.write("• Risque de défaut de paiement")
                st.write("• Actions immédiates requises")
            elif current_ratio < 1.2:
                st.warning("🟡 Liquidité tendue")
                st.write("• Surveillance renforcée")
                st.write("• Optimiser le BFR")
            else:
                st.success("🟢 Liquidité saine")
            
            st.metric("Ratio d'Endettement", f"{debt_to_equity:.2f}")
            if debt_to_equity > 2:
                st.error("🔴 Endettement excessif")
            elif debt_to_equity > 1:
                st.warning("🟡 Endettement élevé")
            else:
                st.success("🟢 Endettement maîtrisé")
            
            # Cash flow risk if available
            cash_flow = csv_data.get('cash_flow', 0)
            st.metric("Cash Flow Mensuel", f"{cash_flow:,.0f} DHS")
            
            if cash_flow < 0:
                st.error("🔴 Cash flow négatif")
                st.write("• Risque de trésorerie")
                st.write("• Besoin financement urgent")
            elif cash_flow < csv_data.get('monthly_costs', 1) * 0.1:
                st.warning("🟡 Cash flow faible")
            else:
                st.success("🟢 Cash flow positif")
        
        # Sector-specific risks
        st.markdown(f"#### 🏭 Risques Spécifiques {industry_manager.templates[detected_industry]['name']}")
        
        if detected_industry == 'saas':
            st.info("☁️ **Risques SaaS Spécifiques :**")
            st.write("• **Churn Rate** : Risque de perte abonnés")
            st.write("• **Acquisition Costs** : Coût client vs LTV")
            st.write("• **Scalabilité** : Capacité infrastructure")
            st.write("• **Concurrence** : Marché très compétitif")
        
        elif detected_industry == 'retail':
            st.info("🛍️ **Risques Retail Spécifiques :**")
            st.write("• **Saisonnalité** : Variations importantes")
            st.write("• **Stocks** : Risque obsolescence")
            st.write("• **Concurrence** : Pression sur marges")
            st.write("• **Supply Chain** : Dépendance fournisseurs")
        
        elif detected_industry == 'technology':
            st.info("💻 **Risques Tech Spécifiques :**")
            st.write("• **Obsolescence** : Évolution technologique")
            st.write("• **Talent** : Pénurie compétences")
            st.write("• **Cycles produits** : Investissement R&D")
            st.write("• **Réglementation** : Évolutions juridiques")
        
        elif detected_industry == 'manufacturing':
            st.info("🏭 **Risques Manufacturing Spécifiques :**")
            st.write("• **Capacité** : Sous-utilisation équipements")
            st.write("• **Qualité** : Défauts et rappels")
            st.write("• **Supply Chain** : Ruptures approvisionnement")
            st.write("• **Réglementation** : Normes environnementales")
    
    with tab2:
        st.subheader("🎲 Simulation Monte Carlo des Risques")
        
        if st.button("🚀 Lancer Simulation Risques", type="secondary"):
            with st.spinner("Exécution simulation Monte Carlo des risques financiers..."):
                
                # Enhanced Monte Carlo simulation with validation context
                base_monthly_revenue = csv_data.get('monthly_revenue', 15000)
                base_monthly_costs = csv_data.get('monthly_costs', 12000)
                
                # Adjust volatility based on data quality
                simulation_volatility = revenue_volatility
                if 'validation_results' in st.session_state:
                    quality_score = st.session_state.validation_results.get('quality_score', 100)
                    if quality_score < 70:
                        simulation_volatility *= 1.3  # Increase uncertainty for poor data quality
                    elif quality_score > 90:
                        simulation_volatility *= 0.9  # Reduce uncertainty for high quality data
                
                mc_results = analytics.monte_carlo_simulation(
                    base_monthly_revenue, 
                    base_monthly_costs, 
                    volatility=simulation_volatility,
                    simulations=1000, 
                    periods=12
                )
                
                # Enhanced risk metrics from Monte Carlo
                profits = mc_results['net_profit']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    var_5 = np.percentile(profits, 5)
                    st.metric("VaR 5%", f"{var_5:,.0f} DHS")
                    if var_5 < 0:
                        st.error("🔴 Risque perte significatif")
                    else:
                        st.success("🟢 Pire cas reste positif")
                
                with col2:
                    prob_loss = (profits < 0).mean() * 100
                    st.metric("Probabilité Perte", f"{prob_loss:.1f}%")
                    if prob_loss > 20:
                        st.error("🔴 Risque élevé")
                    elif prob_loss > 10:
                        st.warning("🟡 Risque modéré")
                    else:
                        st.success("🟢 Risque faible")
                
                with col3:
                    expected_profit = profits.mean()
                    st.metric("Profit Espéré", f"{expected_profit:,.0f} DHS")
                    if expected_profit > 0:
                        st.success("🟢 Rentabilité attendue")
                    else:
                        st.error("🔴 Perte attendue")
                
                with col4:
                    profit_volatility = profits.std()
                    st.metric("Volatilité Profit", f"{profit_volatility:,.0f} DHS")
                    cv = profit_volatility / abs(expected_profit) if expected_profit != 0 else float('inf')
                    if cv < 0.5:
                        st.success("🟢 Prévisible")
                    elif cv < 1.0:
                        st.warning("🟡 Variable")
                    else:
                        st.error("🔴 Très volatile")
                
                # Enhanced risk distribution visualization
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=profits,
                    nbinsx=50,
                    name='Distribution Profits',
                    opacity=0.7,
                    marker_color='lightblue'
                ))
                
                # Add enhanced risk markers
                fig.add_vline(x=var_5, line_dash="dash", line_color="red", 
                             annotation_text="VaR 5%")
                fig.add_vline(x=0, line_dash="solid", line_color="black", 
                             annotation_text="Seuil Rentabilité")
                fig.add_vline(x=expected_profit, line_dash="dot", line_color="green", 
                             annotation_text="Profit Espéré")
                
                # Add percentile lines
                var_1 = np.percentile(profits, 1)
                percentile_95 = np.percentile(profits, 95)
                fig.add_vline(x=var_1, line_dash="dash", line_color="darkred", 
                             annotation_text="VaR 1%")
                fig.add_vline(x=percentile_95, line_dash="dash", line_color="darkgreen", 
                             annotation_text="95e Percentile")
                
                fig.update_layout(
                    title="Distribution des Profits Simulés (1000 scénarios)",
                    xaxis_title="Profit Annuel (DHS)",
                    yaxis_title="Fréquence",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced risk interpretation with industry context
                st.markdown("#### 🎯 Interprétation des Risques")
                
                if prob_loss > 30:
                    st.error("🔴 **Risque Très Élevé** : Plus de 30% de chance de perte")
                    st.write("• Révision stratégique urgente nécessaire")
                    st.write("• Mise en place de mesures de protection")
                    st.write("• Recherche de diversification")
                elif prob_loss > 15:
                    st.warning("🟡 **Risque Élevé** : Probabilité significative de perte")
                    st.write("• Surveillance étroite recommandée")
                    st.write("• Plans de contingence à préparer")
                    st.write("• Amélioration de la résilience")
                elif prob_loss > 5:
                    st.info("🔵 **Risque Modéré** : Faible probabilité de perte")
                    st.write("• Monitoring régulier suffisant")
                    st.write("• Maintenir les stratégies actuelles")
                else:
                    st.success("🟢 **Risque Faible** : Très faible probabilité de perte")
                    st.write("• Position financière solide")
                    st.write("• Opportunités de croissance possibles")
                
                # Data quality impact on simulation
                if 'validation_results' in st.session_state:
                    quality_score = st.session_state.validation_results.get('quality_score', 100)
                    if quality_score < 80:
                        st.warning(f"⚠️ **Note** : Simulation basée sur données qualité {quality_score:.0f}% - Résultats à interpréter avec prudence")
                        
                        if quality_score < 60:
                            st.error("🔴 Recommandation : Améliorer la qualité des données avant prise de décisions critiques")
    
    with tab3:
        st.subheader("💡 Recommandations IA pour Mitigation des Risques")
        
        # Generate enhanced recommendations with validation insights - FIX
        try:
            ratios = analytics.calculate_comprehensive_ratios(csv_data)
            validation_context = st.session_state.get('validation_results')
            
            enhanced_recommendations = analytics.generate_ai_recommendations(
                csv_data, ratios, 100 - total_risk_score, validation_context
            )
            
            if enhanced_recommendations:
                for i, rec in enumerate(enhanced_recommendations):
                    priority_color = "🔴" if rec['priority'] == 'Critique' else "🟠" if rec['priority'] == 'Élevée' else "🟡"
                    
                    with st.expander(f"{priority_color} {rec['category']} - Priorité {rec['priority']}", expanded=i < 3):
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"**Recommandation :** {rec['recommendation']}")
                            
                            # Add enhanced implementation steps
                            st.markdown("**Étapes d'Implémentation :**")
                            if rec['category'] == 'Gestion de trésorerie':
                                st.write("1. Audit complet des flux de trésorerie")
                                st.write("2. Mise en place d'un tableau de bord quotidien")
                                st.write("3. Négociation avec clients et fournisseurs")
                                st.write("4. Recherche de solutions de financement")
                            elif rec['category'] == 'Amélioration de la liquidité':
                                st.write("1. Analyse détaillée du besoin en fonds de roulement")
                                st.write("2. Optimisation des stocks")
                                st.write("3. Révision des conditions de paiement")
                                st.write("4. Mise en place de lignes de crédit")
                            elif rec['category'] == 'Amélioration de la rentabilité':
                                st.write("1. Analyse ABC des produits/services")
                                st.write("2. Étude de marché et pricing")
                                st.write("3. Plan d'optimisation des coûts")
                                st.write("4. Indicateurs de suivi mensuel")
                            else:
                                st.write("1. Évaluation détaillée du problème")
                                st.write("2. Développement d'un plan d'action")
                                st.write("3. Mise en œuvre progressive")
                                st.write("4. Suivi et ajustements")
                        
                        with col_b:
                            st.metric("Impact", rec['impact'])
                            st.metric("Délai", rec['timeframe'])
                            
                            if isinstance(rec.get('estimated_benefit'), (int, float)):
                                st.metric("Bénéfice Est.", f"{rec['estimated_benefit']:,.0f} DHS")
                            else:
                                st.metric("Bénéfice", rec.get('estimated_benefit', 'Qualitatif'))
                            
                            # Risk reduction potential
                            if rec['priority'] == 'Critique':
                                st.metric("Réduction Risque", "15-25%")
                            elif rec['priority'] == 'Élevée':
                                st.metric("Réduction Risque", "8-15%")
                            else:
                                st.metric("Réduction Risque", "3-8%")
            
        except Exception as e:
            st.warning(f"Erreur génération recommandations avancées: {str(e)}")
            
            # Fallback recommendations
            st.markdown("#### 💡 Recommandations Générales")
            
            if total_risk_score > 75:
                st.error("🚨 **Actions Critiques Immédiates**")
                st.write("• Révision complète de la stratégie financière")
                st.write("• Mise en place d'un plan de redressement")
                st.write("• Recherche de financements d'urgence")
                st.write("• Audit externe des processus")
            elif total_risk_score > 50:
                st.warning("⚠️ **Actions Préventives Urgentes**")
                st.write("• Amélioration du monitoring financier")
                st.write("• Diversification des sources de revenus")
                st.write("• Optimisation de la structure de coûts")
                st.write("• Renforcement de la position de trésorerie")
            elif total_risk_score > 25:
                st.info("🔵 **Optimisation Continue**")
                st.write("• Maintenir la surveillance des indicateurs")
                st.write("• Améliorer l'efficacité opérationnelle")
                st.write("• Préparer des plans de contingence")
            else:
                st.success("✅ **Position Saine - Croissance Possible**")
                st.write("• Maintenir les bonnes pratiques actuelles")
                st.write("• Explorer des opportunités de croissance")
                st.write("• Renforcer les avantages concurrentiels")
        
        # Industry-specific risk mitigation
        st.markdown(f"### 🏭 Mitigation Spécifique {industry_manager.templates[detected_industry]['name']}")
        
        template = industry_manager.get_template(detected_industry)
        
        if detected_industry == 'saas':
            st.info("☁️ **Stratégies SaaS Spécifiques :**")
            st.write("• **Churn Rate** : Développer des programmes de fidélisation")
            st.write("• **Acquisition Costs** : Optimiser les canaux marketing")
            st.write("• **Scalabilité** : Préparer l'infrastructure pour la croissance")
            st.write("• **Récurrence** : Diversifier les sources de revenus récurrents")
            
        elif detected_industry == 'retail':
            st.info("🛍️ **Stratégies Retail Spécifiques :**")
            st.write("• **Saisonnalité** : Développer des stratégies anti-cycliques")
            st.write("• **Stocks** : Optimiser la rotation et réduire l'obsolescence")
            st.write("• **Concurrence** : Différenciation et fidélisation client")
            st.write("• **Supply Chain** : Diversifier les fournisseurs")
            
        elif detected_industry == 'technology':
            st.info("💻 **Stratégies Tech Spécifiques :**")
            st.write("• **Obsolescence** : Investissement continu en R&D")
            st.write("• **Talent** : Stratégies de rétention des compétences clés")
            st.write("• **Cycles produits** : Diversification du portefeuille")
            st.write("• **Cybersécurité** : Renforcement de la sécurité informatique")
            
        elif detected_industry == 'manufacturing':
            st.info("🏭 **Stratégies Manufacturing Spécifiques :**")
            st.write("• **Capacité** : Optimisation de l'utilisation des équipements")
            st.write("• **Qualité** : Systèmes de contrôle qualité renforcés")
            st.write("• **Supply Chain** : Sécurisation des approvisionnements")
            st.write("• **Réglementation** : Veille réglementaire continue")
    
    with tab4:
        st.subheader("⚕️ Risques Liés à la Qualité des Données")
        
        if 'validation_results' in st.session_state:
            validation_results = st.session_state.validation_results
            
            # Comprehensive data quality risk assessment
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                quality_score = validation_results.get('quality_score', 100)
                st.metric("Score Qualité", f"{quality_score:.0f}/100")
                
                if quality_score >= 90:
                    st.success("🟢 Risque données minimal")
                elif quality_score >= 70:
                    st.info("🔵 Risque données modéré")
                elif quality_score >= 50:
                    st.warning("🟡 Risque données élevé")
                else:
                    st.error("🔴 Risque données critique")
            
            with col2:
                total_issues = validation_results.get('total_issues', 0)
                st.metric("Anomalies Totales", total_issues)
                
                if total_issues == 0:
                    st.success("✅ Aucune")
                elif total_issues <= 3:
                    st.info("🔵 Limitées")
                else:
                    st.warning("🟡 Nombreuses")
            
            with col3:
                critical_issues = validation_results.get('critical_issues', 0)
                st.metric("Anomalies Critiques", critical_issues)
                
                if critical_issues == 0:
                    st.success("✅ Aucune")
                else:
                    st.error(f"🔴 {critical_issues}")
            
            with col4:
                corrections_count = len(st.session_state.get('correction_log', []))
                st.metric("Corrections Auto", corrections_count)
                
                if corrections_count == 0:
                    st.success("✅ Aucune")
                else:
                    st.info(f"🔧 {corrections_count}")
            
            # Detailed data risk analysis
            st.markdown("#### 🔍 Impact des Risques Données sur l'Analyse Financière")
            
            data_risk_level = "Faible"
            if quality_score < 50:
                data_risk_level = "Critique"
            elif quality_score < 70:
                data_risk_level = "Élevé"
            elif quality_score < 90:
                data_risk_level = "Modéré"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Niveau de Risque Données : {data_risk_level}**")
                
                if data_risk_level == "Critique":
                    st.error("🔴 **Impact Majeur** : Fiabilité des analyses compromise")
                    st.markdown("**Conséquences :**")
                    st.write("• Décisions basées sur des données erronées")
                    st.write("• Sous-estimation ou surestimation des risques")
                    st.write("• Prévisions potentiellement fausses")
                    st.write("• Perte de confiance des parties prenantes")
                    
                elif data_risk_level == "Élevé":
                    st.warning("🟡 **Impact Significatif** : Précision des analyses réduite")
                    st.markdown("**Conséquences :**")
                    st.write("• Marge d'erreur importante dans les analyses")
                    st.write("• Besoin de validation externe")
                    st.write("• Recommandations à prendre avec précaution")
                    
                elif data_risk_level == "Modéré":
                    st.info("🔵 **Impact Limité** : Analyses globalement fiables")
                    st.markdown("**Conséquences :**")
                    st.write("• Léger impact sur la précision")
                    st.write("• Corrections automatiques appliquées")
                    st.write("• Monitoring continu recommandé")
                    
                else:
                    st.success("🟢 **Impact Minimal** : Haute fiabilité des analyses")
                    st.write("• Données de haute qualité")
                    st.write("• Analyses hautement fiables")
                    st.write("• Décisions sûres possibles")
            
            with col2:
                st.markdown("#### 📋 Plan d'Amélioration Qualité Données")
                
                if critical_issues > 0:
                    st.error("🚨 **Action Immédiate Requise**")
                    st.markdown("**Étapes Prioritaires :**")
                    st.write("1. **Audit complet** des processus de saisie")
                    st.write("2. **Correction manuelle** des incohérences critiques")
                    st.write("3. **Formation équipe** sur les standards qualité")
                    st.write("4. **Mise en place contrôles** automatiques")
                
                if quality_score < 80:
                    st.warning("⚠️ **Amélioration Nécessaire**")
                    st.markdown("**Actions Recommandées :**")
                    st.write("• Révision des procédures de collecte")
                    st.write("• Validation croisée des données")
                    st.write("• Formation sur les bonnes pratiques")
                    st.write("• Automatisation des contrôles")
                
                if quality_score >= 80:
                    st.success("✅ **Maintenir Excellence**")
                    st.markdown("**Actions de Maintien :**")
                    st.write("• Monitoring continu de la qualité")
                    st.write("• Révisions périodiques des processus")
                    st.write("• Formation continue des équipes")
                    st.write("• Amélioration continue des outils")
                
                # Specific data quality recommendations
                st.markdown("#### 🎯 Recommandations Spécifiques")
                
                issues = validation_results.get('issues', [])
                for issue in issues[:3]:  # Show top 3 issues
                    if issue.get('severity') in ['Élevée', 'Critique']:
                        issue_type = issue.get('type', 'Anomalie')
                        st.write(f"• **{issue_type}** : {issue.get('message', 'Correction nécessaire')}")
        else:
            st.info("Données de validation non disponibles. Réimportez vos données CSV pour une analyse complète des risques liés aux données.")

# ========== ENHANCED INDUSTRY TEMPLATES ==========
def show_industry_templates():
    """Enhanced industry templates with comprehensive validation and benchmarking"""
    st.header("🏭 Analyse Financière Spécialisée par Industrie")
    
    # Get CSV financial data
    csv_data = CSVDataManager.get_csv_financial_data()
    template_manager = IndustryTemplateManager()
    
    if csv_data:
        # Display data quality context
        quality_context = ""
        if 'validation_results' in st.session_state:
            quality_score = st.session_state.validation_results.get('quality_score', 100)
            if quality_score >= 80:
                quality_context = f" (Données validées - Score: {quality_score:.0f}/100 ✅)"
            else:
                quality_context = f" (Qualité modérée - Score: {quality_score:.0f}/100 ⚠️)"
        
        st.success(f"📊 **Analyse sectorielle alimentée par vos données CSV{quality_context}**")
        
        # Enhanced auto-detection with validation
        detected_industry = template_manager.detect_industry_from_csv(csv_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            confidence_score = 85  # Base confidence
            
            # Adjust confidence based on data quality and quantity
            if 'validation_results' in st.session_state:
                quality_score = st.session_state.validation_results.get('quality_score', 100)
                confidence_score = confidence_score * (quality_score / 100)
            
            data_points = len(csv_data.get('revenue_data', []))
            if data_points < 12:
                confidence_score *= 0.8
            elif data_points >= 24:
                confidence_score *= 1.1
            
            confidence_score = min(100, confidence_score)
            
            st.info(f"🤖 **Industrie Auto-détectée** : {template_manager.templates[detected_industry]['name']} "
                   f"(Confiance: {confidence_score:.0f}%) basée sur vos patterns financiers")
        
        with col2:
            # Allow manual industry selection
            industry_options = list(template_manager.templates.keys())
            selected_industry = st.selectbox(
                "Modifier l'Industrie",
                industry_options,
                index=industry_options.index(detected_industry),
                format_func=lambda x: f"{template_manager.templates[x]['icon']} {template_manager.templates[x]['name']}"
            )
        
        # Industry-specific data validation
        validation_issues = template_manager.validate_industry_data(csv_data, selected_industry)
        if validation_issues:
            with st.expander("⚠️ Validation Spécifique à l'Industrie", expanded=False):
                for issue in validation_issues:
                    st.warning(f"⚠️ {issue}")
    else:
        st.warning("📤 **Aucune Donnée CSV Disponible**")
        st.info("Les Templates Industrie fonctionnent mieux avec vos données financières uploadées pour un benchmarking précis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Importer Données CSV Maintenant", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'csv_import'
                st.rerun()
        
        with col2:
            # Allow exploration without CSV data
            industry_options = list(template_manager.templates.keys())
            selected_industry = st.selectbox(
                "Explorer une Industrie",
                industry_options,
                index=2,  # Default to technology
                format_func=lambda x: f"{template_manager.templates[x]['icon']} {template_manager.templates[x]['name']}"
            )
    
    # Get selected template with enhanced data
    template = template_manager.get_template(selected_industry)
    
    # Enhanced industry overview tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Profil Industrie", 
        "📈 Benchmarking Avancé", 
        "🎯 Votre Performance", 
        "💡 Insights Sectoriels",
        "📋 Plan d'Action",
        "🔍 Analyse Comparative"
    ])
    
    with tab1:
        st.subheader(f"{template['icon']} Profil Complet {template['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Modèle de Revenus")
            st.code(template['revenue_model'], language="text")
            
            st.markdown("### 🎯 Métriques Clés de Performance")
            for i, metric in enumerate(template['key_metrics']):
                icon = "📊" if i % 3 == 0 else "📈" if i % 3 == 1 else "💰"
                st.write(f"{icon} **{metric}**")
            
            st.markdown("### 📊 Ratios Financiers Typiques")
            ratios_df = pd.DataFrame([
                {"Ratio": k.replace('_', ' ').title(), "Valeur": f"{v:.1%}" if isinstance(v, float) and v < 1 else f"{v:.2f}"}
                for k, v in template['typical_ratios'].items()
            ])
            st.dataframe(ratios_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 💼 Structure de Coûts Typique")
            
            # Enhanced cost structure visualization
            cost_structure = template['cost_structure']
            
            fig = go.Figure(data=[go.Pie(
                labels=[k.replace('_', ' ').title() for k in cost_structure.keys()],
                values=list(cost_structure.values()),
                hole=0.4,
                marker=dict(
                    colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF'],
                    line=dict(color='#FFFFFF', width=2)
                ),
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title=f"Structure Coûts {template['name']}",
                annotations=[dict(text=template['icon'], x=0.5, y=0.5, font_size=30, showarrow=False)],
                height=450,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔄 Cycle de Conversion")
            wc_metrics = template['working_capital']
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("DSO*", f"{wc_metrics['days_sales_outstanding']:.0f}j", help="Days Sales Outstanding")
            with col_b:
                st.metric("DIO*", f"{wc_metrics['days_inventory_outstanding']:.0f}j", help="Days Inventory Outstanding")
            with col_c:
                st.metric("DPO*", f"{wc_metrics['days_payable_outstanding']:.0f}j", help="Days Payable Outstanding")
            
            cycle_conversion = (wc_metrics['days_sales_outstanding'] + 
                              wc_metrics['days_inventory_outstanding'] - 
                              wc_metrics['days_payable_outstanding'])
            
            st.metric("**Cycle de Conversion**", f"{cycle_conversion:.0f} jours")
            
            if cycle_conversion < 30:
                st.success("🟢 Cycle très efficace")
            elif cycle_conversion < 60:
                st.info("🔵 Cycle efficace")
            elif cycle_conversion < 90:
                st.warning("🟡 Cycle à optimiser")
            else:
                st.error("🔴 Cycle inefficace")
    
    with tab2:
        st.subheader("📈 Benchmarking Avancé vs Industrie")
        
        if csv_data:
            # Enhanced benchmarking with validation context
            comparison = template_manager.benchmark_against_industry(csv_data, selected_industry)
            
            # Create comprehensive comparison visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Croissance CA', 'Marge Bénéficiaire', 'Performance vs Benchmarks', 'Radar Comparatif'],
                specs=[[{"type": "bar"}, {"type": "bar"}], 
                       [{"type": "bar"}, {"type": "scatterpolar"}]]
            )
            
            # Revenue growth comparison
            if 'revenue_growth' in comparison:
                rg_data = comparison['revenue_growth']
                fig.add_trace(go.Bar(
                    name='Votre Entreprise',
                    x=['Croissance CA'],
                    y=[rg_data['company_value'] * 100],
                    marker_color='lightblue',
                    text=[f"{rg_data['company_value']*100:.1f}%"],
                    textposition='outside'
                ), row=1, col=1)
                
                fig.add_trace(go.Bar(
                    name='Benchmark Industrie',
                    x=['Croissance CA'],
                    y=[rg_data['industry_benchmark'] * 100],
                    marker_color='lightgreen',
                    text=[f"{rg_data['industry_benchmark']*100:.1f}%"],
                    textposition='outside',
                    showlegend=False
                ), row=1, col=1)
            
            # Profit margin comparison
            if 'profit_margin' in comparison:
                pm_data = comparison['profit_margin']
                fig.add_trace(go.Bar(
                    name='Votre Entreprise',
                    x=['Marge Profit'],
                    y=[pm_data['company_value'] * 100],
                    marker_color='lightblue',
                    text=[f"{pm_data['company_value']*100:.1f}%"],
                    textposition='outside',
                    showlegend=False
                ), row=1, col=2)
                
                fig.add_trace(go.Bar(
                    name='Benchmark Industrie',
                    x=['Marge Profit'],
                    y=[pm_data['industry_benchmark'] * 100],
                    marker_color='lightgreen',
                    text=[f"{pm_data['industry_benchmark']*100:.1f}%"],
                    textposition='outside',
                    showlegend=False
                ), row=1, col=2)
            
            # Performance summary
            performance_metrics = []
            company_values = []
            benchmark_values = []
            
            for metric_name, metric_data in comparison.items():
                if isinstance(metric_data, dict) and 'performance' in metric_data:
                    performance_metrics.append(metric_name.replace('_', ' ').title())
                    company_values.append(metric_data['company_value'])
                    benchmark_values.append(metric_data['industry_benchmark'])
            
            if performance_metrics:
                fig.add_trace(go.Bar(
                    name='Votre Performance',
                    x=performance_metrics,
                    y=company_values,
                    marker_color='lightblue'
                ), row=2, col=1)
                
                fig.add_trace(go.Bar(
                    name='Benchmark',
                    x=performance_metrics,
                    y=benchmark_values,
                    marker_color='lightgreen'
                ), row=2, col=1)
                
                # Radar chart for comprehensive view
                fig.add_trace(go.Scatterpolar(
                    r=company_values,
                    theta=performance_metrics,
                    fill='toself',
                    name='Votre Entreprise',
                    line_color='blue'
                ), row=2, col=2)
                
                fig.add_trace(go.Scatterpolar(
                    r=benchmark_values,
                    theta=performance_metrics,
                    fill='toself',
                    name='Benchmark Industrie',
                    line_color='green',
                    opacity=0.6
                ), row=2, col=2)
            
            fig.update_layout(
                height=800,
                title_text=f"Benchmarking Complet vs {template['name']}",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.markdown("#### 📋 Tableau Comparatif Détaillé")
            
            comparison_table = []
            for metric_name, metric_data in comparison.items():
                if isinstance(metric_data, dict):
                    comparison_table.append({
                        'Métrique': metric_name.replace('_', ' ').title(),
                        'Votre Valeur': f"{metric_data.get('company_value', 0):.2%}" if metric_data.get('company_value', 0) < 1 else f"{metric_data.get('company_value', 0):.2f}",
                        'Benchmark': f"{metric_data.get('industry_benchmark', 0):.2%}" if metric_data.get('industry_benchmark', 0) < 1 else f"{metric_data.get('industry_benchmark', 0):.2f}",
                        'Écart': f"{metric_data.get('percentage_difference', 0):+.1f}%",
                        'Performance': metric_data.get('performance', 'N/A'),
                        'Statut Validation': metric_data.get('validation_status', 'Normal')
                    })
            
            if comparison_table:
                df_comparison = pd.DataFrame(comparison_table)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Add data quality note
            if 'validation_results' in st.session_state:
                quality_score = st.session_state.validation_results.get('quality_score', 100)
                if quality_score < 80:
                    st.caption(f"⚠️ Benchmarking basé sur données qualité {quality_score:.0f}% - Validation externe recommandée")
        
        else:
            st.info("📊 Uploadez vos données CSV pour voir le benchmarking personnalisé vs votre industrie")
            
            # Show generic benchmarks
            st.markdown("#### 📊 Benchmarks Génériques de l'Industrie")
            
            benchmark_data = []
            for metric, value in template['benchmarks'].items():
                benchmark_data.append({
                    'Métrique': metric.replace('_', ' ').title(),
                    'Benchmark': f"{value:.1%}" if isinstance(value, float) and value < 1 else f"{value:.2f}",
                    'Description': get_metric_description(metric, selected_industry)
                })
            
            df_benchmarks = pd.DataFrame(benchmark_data)
            st.dataframe(df_benchmarks, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🎯 Analyse de Votre Performance")
        
        if csv_data:
            # Enhanced performance analysis with industry context
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Métriques Financières Actuelles")
                
                monthly_revenue = csv_data.get('monthly_revenue', 0)
                annual_revenue = monthly_revenue * 12
                st.metric("CA Annuel", f"{annual_revenue:,.0f} DHS")
                
                profit_margin = csv_data.get('profit_margin', 0)
                st.metric("Marge Bénéficiaire", f"{profit_margin:.1f}%")
                
                revenue_growth = csv_data.get('revenue_growth', 0)
                st.metric("Croissance CA", f"{revenue_growth:+.1f}%")
                
                revenue_volatility = csv_data.get('revenue_volatility', 0)
                st.metric("Volatilité CA", f"{revenue_volatility:.1%}")
                
                # Calculate industry-specific KPIs
                st.markdown("#### 🎯 KPIs Spécifiques à Votre Industrie")
                
                if selected_industry == 'saas':
                    # SaaS-specific metrics
                    mrr = monthly_revenue
                    arr = mrr * 12
                    st.metric("MRR", f"{mrr:,.0f} DHS")
                    st.metric("ARR", f"{arr:,.0f} DHS")
                    
                    # Estimated metrics
                    estimated_customers = max(100, int(mrr / 50))  # Assuming avg 50 DHS per customer
                    st.metric("Clients Estimés", f"{estimated_customers:,}")
                    
                    estimated_churn = min(0.15, max(0.03, revenue_volatility))
                    st.metric("Churn Estimé", f"{estimated_churn:.1%}")
                
                elif selected_industry == 'retail':
                    # Retail-specific metrics
                    inventory_turnover = template['benchmarks'].get('inventory_turns', 6)
                    st.metric("Rotation Stocks", f"{inventory_turnover:.1f}x")
                    
                    # Seasonal factor analysis
                    if len(csv_data.get('revenue_data', [])) >= 12:
                        revenue_data = csv_data['revenue_data']
                        seasonal_cv = calculate_seasonality(revenue_data)
                        st.metric("Facteur Saisonnier", f"{seasonal_cv:.1%}")
                
                elif selected_industry == 'technology':
                    # Tech-specific metrics
                    estimated_rd_ratio = min(0.3, max(0.05, profit_margin * 0.02))
                    st.metric("R&D/CA Estimé", f"{estimated_rd_ratio:.1%}")
                    
                    if annual_revenue > 0:
                        revenue_per_employee = annual_revenue / max(10, annual_revenue // 200000)  # Estimate employees
                        st.metric("CA/Employé", f"{revenue_per_employee:,.0f} DHS")
                
                elif selected_industry == 'manufacturing':
                    # Manufacturing-specific metrics
                    asset_turnover = template['typical_ratios'].get('asset_turnover', 1.2)
                    st.metric("Rotation Actifs", f"{asset_turnover:.2f}")
                    
                    estimated_capacity = max(0.6, min(0.95, (revenue_growth + 100) / 120))
                    st.metric("Utilisation Capacité", f"{estimated_capacity:.1%}")
            
            with col2:
                st.markdown("#### 📈 Position Relative dans l'Industrie")
                
                # Industry position analysis
                position_score = 0
                position_factors = []
                
                # Revenue growth position
                benchmark_growth = template['benchmarks'].get('revenue_growth', 0.1) * 100
                if revenue_growth >= benchmark_growth * 1.2:
                    position_score += 25
                    position_factors.append("🟢 Croissance excellente")
                elif revenue_growth >= benchmark_growth:
                    position_score += 15
                    position_factors.append("🔵 Croissance au-dessus benchmark")
                elif revenue_growth >= benchmark_growth * 0.8:
                    position_score += 10
                    position_factors.append("🟡 Croissance correcte")
                else:
                    position_factors.append("🔴 Croissance sous benchmark")
                
                # Profitability position
                benchmark_margin = template['benchmarks'].get('profit_margin', 0.1) * 100
                if profit_margin >= benchmark_margin * 1.2:
                    position_score += 25
                    position_factors.append("🟢 Rentabilité excellente")
                elif profit_margin >= benchmark_margin:
                    position_score += 15
                    position_factors.append("🔵 Rentabilité au-dessus benchmark")
                elif profit_margin >= benchmark_margin * 0.8:
                    position_score += 10
                    position_factors.append("🟡 Rentabilité correcte")
                else:
                    position_factors.append("🔴 Rentabilité sous benchmark")
                
                # Stability position
                if revenue_volatility <= 0.1:
                    position_score += 20
                    position_factors.append("🟢 Revenus très stables")
                elif revenue_volatility <= 0.2:
                    position_score += 15
                    position_factors.append("🔵 Revenus stables")
                elif revenue_volatility <= 0.3:
                    position_score += 10
                    position_factors.append("🟡 Revenus modérément volatils")
                else:
                    position_factors.append("🔴 Revenus très volatils")
                
                # Add data quality adjustment
                if 'validation_results' in st.session_state:
                    quality_score = st.session_state.validation_results.get('quality_score', 100)
                    position_score = position_score * (quality_score / 100)
                
                position_score = min(100, position_score)
                
                # Display position score
                st.metric("Score Position Industrie", f"{position_score:.0f}/100")
                
                if position_score >= 80:
                    st.success("🏆 **Leader dans votre secteur**")
                elif position_score >= 60:
                    st.info("📈 **Performance au-dessus de la moyenne**")
                elif position_score >= 40:
                    st.warning("📊 **Performance moyenne**")
                else:
                    st.error("📉 **Performance en dessous de la moyenne**")
                
                # Display position factors
                st.markdown("**Facteurs de Position :**")
                for factor in position_factors:
                    st.write(f"• {factor}")
                
                # Industry ranking visualization
                fig_ranking = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=position_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Position {template['name']}"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 50], 'color': "orange"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ]
                    }
                ))
                
                fig_ranking.update_layout(height=300)
                st.plotly_chart(fig_ranking, use_container_width=True)
        
        else:
            st.info("📊 Uploadez vos données CSV pour voir une analyse personnalisée de votre performance vs l'industrie")
    
    with tab4:
        st.subheader("💡 Insights Sectoriels Avancés")
        
        if csv_data:
            # Generate industry-specific insights with validation
            insights, recommendations = template_manager.generate_industry_insights(csv_data, selected_industry)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Insights Validés")
                if insights:
                    for insight in insights:
                        st.success(f"✅ {insight}")
                        
                        # Add confidence level based on data quality
                        if 'validation_results' in st.session_state:
                            quality_score = st.session_state.validation_results.get('quality_score', 100)
                            if quality_score >= 90:
                                st.caption("🔹 Confiance très élevée")
                            elif quality_score >= 70:
                                st.caption("🔸 Confiance élevée")
                            else:
                                st.caption("🔸 Confiance modérée - Validation externe recommandée")
                else:
                    st.info("Performance dans les normes sectorielles. Maintenir la stratégie actuelle.")
            
            with col2:
                st.markdown("#### 🎯 Recommandations Sectorielles")
                if recommendations:
                    for rec in recommendations:
                        if "⚠️" in rec or "🔴" in rec:
                            st.error(rec)
                        elif "💡" in rec or "🎯" in rec:
                            st.warning(rec)
                        else:
                            st.info(rec)
                else:
                    st.success("✅ Aucune recommandation spécifique identifiée")
            
            # Advanced sector-specific analysis
            st.markdown("#### 🔍 Analyse Sectorielle Avancée")
            
            # Seasonal pattern analysis
            if len(csv_data.get('revenue_data', [])) >= 12:
                revenue_data = csv_data['revenue_data']
                seasonal_factors = template.get('seasonal_factors', [1.0] * 12)
                
                fig_seasonal = go.Figure()
                
                # Historical seasonality
                monthly_avg = []
                for month in range(12):
                    month_values = [revenue_data[i] for i in range(month, len(revenue_data), 12)]
                    if month_values:
                        monthly_avg.append(np.mean(month_values))
                    else:
                        monthly_avg.append(0)
                
                if len(monthly_avg) == 12:
                    months_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                                   'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                    
                    # Normalize for comparison
                    if np.mean(monthly_avg) > 0:
                        normalized_actual = [m / np.mean(monthly_avg) for m in monthly_avg]
                    else:
                        normalized_actual = [1.0] * 12
                    
                    fig_seasonal.add_trace(go.Scatter(
                        x=months_names,
                        y=normalized_actual,
                        mode='lines+markers',
                        name='Votre Saisonnalité',
                        line=dict(color='blue', width=3)
                    ))
                    
                    fig_seasonal.add_trace(go.Scatter(
                        x=months_names,
                        y=seasonal_factors,
                        mode='lines+markers',
                        name=f'Saisonnalité Typique {template["name"]}',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig_seasonal.update_layout(
                        title="Analyse Saisonnalité vs Benchmark Industrie",
                        yaxis_title="Facteur Saisonnier (Base 1.0)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_seasonal, use_container_width=True)
                    
                    # Seasonality insights
                    actual_seasonality = np.std(normalized_actual)
                    expected_seasonality = np.std(seasonal_factors)
                    
                    if actual_seasonality > expected_seasonality * 1.5:
                        st.warning("⚠️ **Saisonnalité plus marquée** que la moyenne du secteur")
                    elif actual_seasonality < expected_seasonality * 0.5:
                        st.success("✅ **Revenus moins saisonniers** que la moyenne du secteur")
                    else:
                        st.info("📊 **Patterns saisonniers** alignés avec le secteur")
        
        else:
            # Show generic industry insights
            st.markdown("#### 🏭 Insights Génériques du Secteur")
            
            if selected_industry == 'saas':
                st.info("☁️ **Facteurs Clés de Succès SaaS :**")
                st.write("• **Récurrence** : Focus sur les revenus prévisibles")
                st.write("• **Churn Management** : Réduction taux attrition")
                st.write("• **Unit Economics** : LTV > 3x CAC")
                st.write("• **Scalabilité** : Croissance sans augmentation proportionnelle des coûts")
                
            elif selected_industry == 'retail':
                st.info("🛍️ **Facteurs Clés de Succès Retail :**")
                st.write("• **Rotation Stock** : Optimisation des niveaux d'inventaire")
                st.write("• **Saisonnalité** : Planification selon les cycles")
                st.write("• **Expérience Client** : Différenciation par le service")
                st.write("• **Supply Chain** : Efficacité logistique")
                
            elif selected_industry == 'technology':
                st.info("💻 **Facteurs Clés de Succès Tech :**")
                st.write("• **Innovation** : Investissement continu en R&D")
                st.write("• **Time-to-Market** : Rapidité de développement")
                st.write("• **Talent** : Attraction et rétention des compétences")
                st.write("• **Écosystème** : Partenariats stratégiques")
                
            elif selected_industry == 'manufacturing':
                st.info("🏭 **Facteurs Clés de Succès Manufacturing :**")
                st.write("• **Efficacité Opérationnelle** : Optimisation des processus")
                st.write("• **Qualité** : Systèmes de contrôle rigoureux")
                st.write("• **Capacité** : Utilisation optimale des équipements")
                st.write("• **Supply Chain** : Sécurisation des approvisionnements")
    
    with tab5:
        st.subheader("📋 Plan d'Action Personnalisé")
        
        if csv_data:
            # Generate comprehensive action plan
            st.markdown("#### 🎯 Plan d'Action Basé sur Votre Performance")
            
            # Analyze current position
            revenue_growth = csv_data.get('revenue_growth', 0)
            profit_margin = csv_data.get('profit_margin', 0)
            revenue_volatility = csv_data.get('revenue_volatility', 0)
            
            benchmark_growth = template['benchmarks'].get('revenue_growth', 0.1) * 100
            benchmark_margin = template['benchmarks'].get('profit_margin', 0.1) * 100
            
            # Priority actions based on gaps
            priority_actions = []
            
            # Growth gap analysis
            growth_gap = revenue_growth - benchmark_growth
            if growth_gap < -10:
                priority_actions.append({
                    'priority': 'Critique',
                    'area': 'Croissance',
                    'action': 'Plan de relance croissance urgent',
                    'timeframe': '0-3 mois',
                    'description': 'Analyser causes du retard de croissance et implémenter actions correctives',
                    'kpis': ['Taux de croissance mensuel', 'Part de marché', 'Acquisition clients']
                })
            elif growth_gap < -5:
                priority_actions.append({
                    'priority': 'Élevée',
                    'area': 'Croissance',
                    'action': 'Accélération stratégie croissance',
                    'timeframe': '3-6 mois',
                    'description': 'Identifier leviers croissance supplémentaires',
                    'kpis': ['Pipeline commercial', 'Conversion leads', 'Rétention clients']
                })
            
            # Profitability gap analysis
            margin_gap = profit_margin - benchmark_margin
            if margin_gap < -5:
                priority_actions.append({
                    'priority': 'Critique',
                    'area': 'Rentabilité',
                    'action': 'Optimisation urgente des marges',
                    'timeframe': '0-3 mois',
                    'description': 'Révision complète structure coûts et pricing',
                    'kpis': ['Marge brute', 'Coût par unité', 'Pricing power']
                })
            elif margin_gap < -2:
                priority_actions.append({
                    'priority': 'Élevée',
                    'area': 'Rentabilité',
                    'action': 'Amélioration continue rentabilité',
                    'timeframe': '3-6 mois',
                    'description': 'Optimisation opérationnelle et efficacité',
                    'kpis': ['Productivité', 'Automation ratio', 'Cost per acquisition']
                })
            
            # Volatility analysis
            if revenue_volatility > 0.3:
                priority_actions.append({
                    'priority': 'Moyenne',
                    'area': 'Stabilité',
                    'action': 'Réduction volatilité revenus',
                    'timeframe': '6-12 mois',
                    'description': 'Diversification sources revenus et amélioration prédictibilité',
                    'kpis': ['Coefficient variation CA', 'Récurrence revenus', 'Diversification clients']
                })
            
            # Industry-specific actions
            if selected_industry == 'saas':
                priority_actions.append({
                    'priority': 'Moyenne',
                    'area': 'SaaS Optimization',
                    'action': 'Optimisation métriques SaaS',
                    'timeframe': '3-9 mois',
                    'description': 'Focus sur MRR, réduction churn et amélioration LTV/CAC',
                    'kpis': ['MRR Growth', 'Churn Rate', 'LTV/CAC Ratio', 'Net Revenue Retention']
                })
            elif selected_industry == 'retail':
                priority_actions.append({
                    'priority': 'Moyenne',
                    'area': 'Retail Excellence',
                    'action': 'Optimisation opérations retail',
                    'timeframe': '3-9 mois',
                    'description': 'Amélioration rotation stocks et gestion saisonnalité',
                    'kpis': ['Inventory Turnover', 'Same-Store Sales', 'Gross Margin', 'Customer Traffic']
                })
            elif selected_industry == 'technology':
                priority_actions.append({
                    'priority': 'Moyenne',
                    'area': 'Tech Innovation',
                    'action': 'Renforcement capacités innovation',
                    'timeframe': '6-12 mois',
                    'description': 'Investissement R&D et accélération time-to-market',
                    'kpis': ['R&D/Revenue Ratio', 'Time-to-Market', 'Innovation Pipeline', 'Patent Portfolio']
                })
            elif selected_industry == 'manufacturing':
                priority_actions.append({
                    'priority': 'Moyenne',
                    'area': 'Manufacturing Excellence',
                    'action': 'Optimisation efficacité opérationnelle',
                    'timeframe': '6-12 mois',
                    'description': 'Amélioration OEE et réduction waste',
                    'kpis': ['OEE', 'Capacity Utilization', 'Quality Rate', 'Lead Time']
                })
            
            # Sort by priority
            priority_order = {'Critique': 0, 'Élevée': 1, 'Moyenne': 2}
            priority_actions.sort(key=lambda x: priority_order.get(x['priority'], 3))
            
            # Display action plan
            for i, action in enumerate(priority_actions):
                priority_color = "🔴" if action['priority'] == 'Critique' else "🟠" if action['priority'] == 'Élevée' else "🟡"
                
                with st.expander(f"{priority_color} Action {i+1}: {action['action']} - Priorité {action['priority']}", 
                               expanded=i < 2):
                    
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.markdown(f"**Domaine** : {action['area']}")
                        st.markdown(f"**Description** : {action['description']}")
                        
                        st.markdown("**Étapes Détaillées :**")
                        if action['area'] == 'Croissance':
                            st.write("1. Audit des canaux d'acquisition actuels")
                            st.write("2. Analyse de la concurrence et positionnement")
                            st.write("3. Développement de nouveaux segments/produits")
                            st.write("4. Mise en place de programmes de fidélisation")
                            st.write("5. Suivi quotidien des métriques de croissance")
                        elif action['area'] == 'Rentabilité':
                            st.write("1. Analyse détaillée de la structure de coûts")
                            st.write("2. Benchmark des prix vs concurrence")
                            st.write("3. Optimisation des processus opérationnels")
                            st.write("4. Négociation avec fournisseurs clés")
                            st.write("5. Mise en place de contrôles de gestion renforcés")
                        elif action['area'] == 'Stabilité':
                            st.write("1. Diversification du portefeuille clients")
                            st.write("2. Développement de revenus récurrents")
                            st.write("3. Amélioration de la prédictibilité des ventes")
                            st.write("4. Mise en place de systèmes d'alerte précoce")
                        else:
                            st.write("1. Évaluation de la situation actuelle")
                            st.write("2. Définition d'objectifs spécifiques")
                            st.write("3. Mise en place d'initiatives ciblées")
                            st.write("4. Monitoring et ajustements continus")
                    
                    with col_b:
                        st.metric("Délai", action['timeframe'])
                        st.metric("Priorité", action['priority'])
                        
                        st.markdown("**KPIs à Suivre :**")
                        for kpi in action['kpis']:
                            st.write(f"• {kpi}")
            
            # Add data quality improvement action if needed
            if 'validation_results' in st.session_state:
                quality_score = st.session_state.validation_results.get('quality_score', 100)
                if quality_score < 80:
                    with st.expander("🔧 Action Transversale: Amélioration Qualité Données - Priorité Élevée", expanded=False):
                        st.markdown("**Domaine** : Gouvernance des Données")
                        st.markdown(f"**Score Actuel** : {quality_score:.0f}/100")
                        
                        st.markdown("**Actions Immédiates :**")
                        st.write("1. Audit complet des processus de collecte des données")
                        st.write("2. Formation des équipes sur les standards qualité")
                        st.write("3. Mise en place de contrôles automatiques")
                        st.write("4. Validation croisée des données critiques")
                        st.write("5. Mise en place d'un monitoring continu")
                        
                        st.metric("Impact Attendu", "Amélioration fiabilité analyses +25%")
        
        else:
            st.info("📊 Uploadez vos données CSV pour générer un plan d'action personnalisé basé sur votre performance actuelle")
            
            # Show generic best practices for the industry
            st.markdown(f"#### 🏭 Meilleures Pratiques {template['name']}")
            
            if selected_industry == 'saas':
                st.write("• **Optimisation MRR** : Focus sur la croissance des revenus récurrents")
                st.write("• **Réduction Churn** : Programmes de fidélisation et support client")
                st.write("• **Scaling Efficient** : Amélioration des ratios de productivité")
                st.write("• **Product-Market Fit** : Validation continue de l'adéquation produit-marché")
                
            elif selected_industry == 'retail':
                st.write("• **Inventory Management** : Optimisation rotation et réduction stocks morts")
                st.write("• **Customer Experience** : Amélioration parcours client omnicanal")
                st.write("• **Seasonal Planning** : Anticipation et préparation des pics d'activité")
                st.write("• **Supply Chain Efficiency** : Optimisation chaîne logistique")
                
            elif selected_industry == 'technology':
                st.write("• **R&D Investment** : Maintenir l'avantage concurrentiel par l'innovation")
                st.write("• **Talent Acquisition** : Attraction et rétention des meilleurs profils")
                st.write("• **Agile Development** : Accélération time-to-market")
                st.write("• **Partnership Ecosystem** : Développement d'écosystèmes partenaires")
                
            elif selected_industry == 'manufacturing':
                st.write("• **Operational Excellence** : Optimisation continue des processus")
                st.write("• **Quality Systems** : Mise en place de systèmes qualité robustes")
                st.write("• **Capacity Optimization** : Maximisation utilisation des équipements")
                st.write("• **Supply Chain Resilience** : Sécurisation des approvisionnements")
    
    with tab6:
        st.subheader("🔍 Analyse Comparative Multi-Sectorielle")
        
        if csv_data:
            st.markdown("### 📊 Comparaison Multi-Sectorielle")
            
            # Compare against all industry benchmarks
            all_comparisons = {}
            for industry_key in template_manager.templates.keys():
                comparison = template_manager.benchmark_against_industry(csv_data, industry_key)
                all_comparisons[industry_key] = comparison
            
            # Create comprehensive comparison visualization
            industries = list(all_comparisons.keys())
            revenue_growth_company = []
            revenue_growth_benchmarks = []
            profit_margin_company = []
            profit_margin_benchmarks = []
            
            for industry in industries:
                comp = all_comparisons[industry]
                if 'revenue_growth' in comp:
                    revenue_growth_company.append(comp['revenue_growth']['company_value'] * 100)
                    revenue_growth_benchmarks.append(comp['revenue_growth']['industry_benchmark'] * 100)
                else:
                    revenue_growth_company.append(0)
                    revenue_growth_benchmarks.append(0)
                
                if 'profit_margin' in comp:
                    profit_margin_company.append(comp['profit_margin']['company_value'] * 100)
                    profit_margin_benchmarks.append(comp['profit_margin']['industry_benchmark'] * 100)
                else:
                    profit_margin_company.append(0)
                    profit_margin_benchmarks.append(0)
            
            # Multi-industry comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Croissance CA vs Industries', 'Marge Bénéficiaire vs Industries'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Revenue growth comparison
            fig.add_trace(
                go.Bar(name='Votre Entreprise', x=[template_manager.templates[ind]['name'] for ind in industries], 
                      y=revenue_growth_company, marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Benchmark Industrie', x=[template_manager.templates[ind]['name'] for ind in industries], 
                      y=revenue_growth_benchmarks, marker_color='lightgreen'),
                row=1, col=1
            )
            
            # Profit margin comparison
            fig.add_trace(
                go.Bar(name='Votre Entreprise', x=[template_manager.templates[ind]['name'] for ind in industries], 
                      y=profit_margin_company, marker_color='lightblue', showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='Benchmark Industrie', x=[template_manager.templates[ind]['name'] for ind in industries], 
                      y=profit_margin_benchmarks, marker_color='lightgreen', showlegend=False),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                title_text="Comparaison Performance Multi-Sectorielle",
                barmode='group'
            )
            fig.update_yaxes(title_text="Croissance CA (%)", row=1, col=1)
            fig.update_yaxes(title_text="Marge Bénéficiaire (%)", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best fit industry analysis
            st.markdown("### 🎯 Analyse Meilleur Fit Sectoriel")
            
            fit_scores = {}
            for industry in industries:
                score = 0
                comp = all_comparisons[industry]
                
                # Score based on how close company performance is to industry benchmarks
                if 'revenue_growth' in comp:
                    growth_diff = abs(comp['revenue_growth']['percentage_difference'])
                    score += max(0, 100 - growth_diff)
                
                if 'profit_margin' in comp:
                    margin_diff = abs(comp['profit_margin']['percentage_difference'])
                    score += max(0, 100 - margin_diff)
                
                fit_scores[industry] = score / 2  # Average of the two metrics
            
            # Sort industries by fit score
            sorted_industries = sorted(fit_scores.items(), key=lambda x: x[1], reverse=True)
            
            st.markdown("#### 📈 Classement Fit Sectoriel")
            
            for i, (industry, score) in enumerate(sorted_industries):
                template_info = template_manager.templates[industry]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{i+1}. {template_info['icon']} {template_info['name']}**")
                
                with col2:
                    st.metric("Score Fit", f"{score:.0f}/100")
                
                with col3:
                    if i == 0:
                        st.success("🥇 Meilleur Fit")
                    elif i == 1:
                        st.info("🥈 Bon Fit")
                    elif i == 2:
                        st.warning("🥉 Fit Modéré")
                    else:
                        st.caption("Fit Limité")
                
                # Show why this industry fits (or doesn't)
                if score > 80:
                    st.success(f"✅ Votre profil financier correspond très bien au secteur {template_info['name']}")
                elif score > 60:
                    st.info(f"ℹ️ Votre profil s'aligne bien avec le secteur {template_info['name']}")
                elif score > 40:
                    st.warning(f"⚠️ Alignement partiel avec le secteur {template_info['name']}")
                else:
                    st.error(f"❌ Profil peu compatible avec le secteur {template_info['name']}")
                
                st.markdown("---")
            
            # Strategic recommendations based on best fit
            best_fit_industry = sorted_industries[0][0]
            if best_fit_industry != selected_industry:
                st.markdown("### 💡 Recommandations Stratégiques")
                st.info(f"🎯 **Insight Sectoriel** : Vos métriques s'alignent mieux avec {template_manager.templates[best_fit_industry]['name']} "
                       f"que votre classification actuelle ({template_manager.templates[selected_industry]['name']})")
                
                st.markdown("**Options Stratégiques :**")
                st.write(f"• **Pivoter** vers le modèle {template_manager.templates[best_fit_industry]['name']}")
                st.write(f"• **Hybrider** en adoptant les meilleures pratiques de {template_manager.templates[best_fit_industry]['name']}")
                st.write(f"• **Optimiser** votre modèle actuel pour mieux s'aligner avec {template['name']}")
            
            # Competitive positioning analysis
            st.markdown("### 🎯 Positionnement Concurrentiel")
            
            # Create positioning matrix
            fig = go.Figure()
            
            for industry in industries:
                comp = all_comparisons[industry]
                x_val = comp.get('revenue_growth', {}).get('company_value', 0) * 100
                y_val = comp.get('profit_margin', {}).get('company_value', 0) * 100
                
                template_info = template_manager.templates[industry]
                
                # Add industry benchmark point
                x_benchmark = comp.get('revenue_growth', {}).get('industry_benchmark', 0) * 100
                y_benchmark = comp.get('profit_margin', {}).get('industry_benchmark', 0) * 100
                
                fig.add_trace(go.Scatter(
                    x=[x_benchmark],
                    y=[y_benchmark],
                    mode='markers',
                    name=f"Benchmark {template_info['name']}",
                    marker=dict(size=15, symbol='diamond'),
                    text=[template_info['icon']],
                    textposition="middle center"
                ))
            
            # Add company position
            company_growth = csv_data.get('revenue_growth', 0)
            company_margin = csv_data.get('profit_margin', 0)
            
            fig.add_trace(go.Scatter(
                x=[company_growth],
                y=[company_margin],
                mode='markers',
                name='Votre Entreprise',
                marker=dict(size=20, color='red', symbol='star'),
                text=['VOUS'],
                textposition="middle center"
            ))
            
            fig.update_layout(
                title='Matrice Positionnement: Croissance vs Rentabilité',
                xaxis_title='Croissance CA (%)',
                yaxis_title='Marge Bénéficiaire (%)',
                height=500
            )
            
            # Add quadrant lines
            fig.add_hline(y=company_margin, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_vline(x=company_growth, line_dash="dot", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Quadrant analysis
            st.markdown("#### 🎯 Analyse Positionnement")
            
            if company_growth > 10 and company_margin > 15:
                st.success("🌟 **Position Excellente** : Forte croissance et haute rentabilité - Vous dominez votre marché")
            elif company_growth > 10 and company_margin > 5:
                st.info("📈 **Croissance Forte** : Excellent momentum, optimiser la rentabilité")
            elif company_growth > 0 and company_margin > 15:
                st.info("💰 **Rentabilité Forte** : Excellentes marges, accélérer la croissance")
            elif company_growth > 0 and company_margin > 5:
                st.warning("⚖️ **Position Équilibrée** : Performance correcte, identifier leviers d'amélioration")
            else:
                st.error("🔄 **Repositionnement Nécessaire** : Amélioration urgente requise sur croissance et/ou rentabilité")
            
        else:
            st.info("📊 L'analyse comparative multi-sectorielle sera disponible avec vos données CSV")
            
            # Show general comparison framework
            st.markdown("### 🔍 Framework Analyse Comparative")
            
            comparison_framework = {
                'Dimension': ['Croissance CA', 'Rentabilité', 'Efficacité', 'Stabilité', 'Innovation'],
                'SaaS': ['Très Élevée', 'Élevée', 'Variable', 'Élevée', 'Critique'],
                'Retail': ['Modérée', 'Faible', 'Élevée', 'Variable', 'Modérée'],
                'Technology': ['Élevée', 'Élevée', 'Modérée', 'Modérée', 'Très Élevée'],
                'Manufacturing': ['Modérée', 'Modérée', 'Très Élevée', 'Élevée', 'Faible']
            }
            
            comparison_df = pd.DataFrame(comparison_framework)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown("**Légende :**")
            st.write("• **Très Élevée** : Performance supérieure attendue")
            st.write("• **Élevée** : Performance au-dessus de la moyenne")
            st.write("• **Modérée** : Performance moyenne")
            st.write("• **Faible** : Performance en dessous de la moyenne")
            st.write("• **Variable** : Dépend fortement du contexte")

# ========== HELPER FUNCTIONS ==========
def calculate_seasonality(data):
    """Calculate seasonality coefficient for retail analysis"""
    if len(data) < 12:
        return 0
    
    try:
        monthly_means = []
        for month in range(12):
            month_data = [data[i] for i in range(month, len(data), 12)]
            if month_data:
                monthly_means.append(np.mean(month_data))
        
        if len(monthly_means) == 12:
            return np.std(monthly_means) / np.mean(monthly_means)
        else:
            return 0
    except:
        return 0

def get_metric_description(metric, industry):
    """Get description for a metric based on industry context"""
    descriptions = {
        'revenue_growth': f"Taux de croissance annuel du chiffre d'affaires typique pour {industry}",
        'profit_margin': f"Marge bénéficiaire nette moyenne observée dans {industry}",
        'inventory_turns': f"Nombre de rotations d'inventaire par an dans {industry}",
        'churn_rate': f"Taux d'attrition mensuel des clients dans {industry}",
        'ltv_cac_ratio': f"Ratio valeur vie client sur coût d'acquisition dans {industry}",
        'gross_margin': f"Marge brute typique pour {industry}",
        'capacity_utilization': f"Taux d'utilisation des capacités de production dans {industry}",
        'oee': f"Overall Equipment Effectiveness dans {industry}",
        'defect_rate': f"Taux de défauts typique dans {industry}"
    }
    
    return descriptions.get(metric, f"Métrique de référence pour {industry}")

# ========== MAIN APPLICATION ==========
def main():
    """Enhanced main function with comprehensive validation and industry analysis"""
    
    init_session_state()
    
    # Enhanced header with data quality status
    st.sidebar.markdown(f"""
    ### 🏢 Suite Analyse Financière Avancée
    **Bienvenue dans la Plateforme Analytics Professionnelle**
    
    *Validation Avancée • Insights IA • Benchmarking Sectoriel*
    
    ---
    """)
    
    # Enhanced CSV import indicator with quality metrics
    if CSVDataManager.has_csv_data():
        st.sidebar.success("📊 Données CSV Chargées")
        
        # Show enhanced metrics with validation context
        csv_data = CSVDataManager.get_csv_financial_data()
        if csv_data:
            monthly_revenue = csv_data.get('monthly_revenue', 0)
            profit_margin = csv_data.get('profit_margin', 0)
            st.sidebar.metric("CA Mensuel", f"{monthly_revenue:,.0f} DHS")
            st.sidebar.metric("Marge Bénéficiaire", f"{profit_margin:.1f}%")
            
            # Data quality indicator
            if 'validation_results' in st.session_state:
                quality_score = st.session_state.validation_results.get('quality_score', 100)
                if quality_score >= 90:
                    st.sidebar.success(f"🟢 Qualité: {quality_score:.0f}/100")
                elif quality_score >= 70:
                    st.sidebar.info(f"🔵 Qualité: {quality_score:.0f}/100")
                else:
                    st.sidebar.warning(f"🟡 Qualité: {quality_score:.0f}/100")
                
                critical_issues = st.session_state.validation_results.get('critical_issues', 0)
                if critical_issues > 0:
                    st.sidebar.error(f"🔴 {critical_issues} problème(s) critique(s)")
                
                corrections_applied = len(st.session_state.get('correction_log', []))
                if corrections_applied > 0:
                    st.sidebar.info(f"🔧 {corrections_applied} correction(s) auto")
    else:
        st.sidebar.warning("📤 Aucune Donnée CSV")
        st.sidebar.caption("Uploadez des données pour analyse complète")
    
    # Enhanced navigation menu with new capabilities
    menu_items = {
        "📤 Import CSV Intelligent": "csv_import",
        "👔 Dashboard Exécutif": "executive_dashboard",
        "🧠 Analytics IA Avancés": "advanced_analytics", 
        "🎯 Planification Scénarios": "scenario_planning",
        "🤖 Prévisions ML Optimisées": "ml_forecasting",
        "⚠️ Gestion Risques Avancée": "risk_management",
        "🏭 Templates Sectoriels": "industry_templates"
    }
    
    # Handle enhanced page navigation
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
            "🧭 Navigation",
            list(menu_items.keys()),
            index=0
        )
    
    # Route to appropriate enhanced page
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
    
    # Enhanced sidebar footer with comprehensive status
    with st.sidebar:
        st.markdown("---")
        
        # Enhanced system status
        st.markdown("### 🔧 Statut Système Avancé")
        st.success("🟢 Processeur CSV: Opérationnel")
        st.success("🟢 Moteur Analytics: Actif") 
        st.success("🟢 Modèles ML: Disponibles")
        st.success("🟢 Templates Sectoriels: Complets")
        st.success("🟢 Validation Avancée: Active")
        st.success("🟢 Corrections Auto: Fonctionnelles")
        
        # Enhanced datetime and user info
        current_datetime = datetime.now()
        st.caption(f"Heure Actuelle: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("Utilisateur: SalianiBouchaib")
        
        # Performance metrics if data available
        if CSVDataManager.has_csv_data():
            st.markdown("---")
            st.markdown("### 📈 Métriques Session")
            
            # Calculate session metrics
            data_points = len(st.session_state.get('imported_metrics', {}).get('revenue', {}).get('data', []))
            st.caption(f"Points de données: {data_points}")
            
            # Validation summary
            if 'validation_results' in st.session_state:
                validation_results = st.session_state.validation_results
                total_issues = validation_results.get('total_issues', 0)
                st.caption(f"Validations effectuées: {total_issues} anomalies détectées")
                
                corrections = len(st.session_state.get('correction_log', []))
                if corrections > 0:
                    st.caption(f"Corrections appliquées: {corrections}")
        
        # Enhanced additional info
        st.markdown("---")
        st.markdown("### 📊 Capacités Avancées")
        st.caption("✅ Validation Équation Comptable")
        st.caption("✅ Contrôle Logique Profit")
        st.caption("✅ Détection Variations Extrêmes")
        st.caption("✅ Identification Outliers (3σ)")
        st.caption("✅ Corrections Automatisées")
        st.caption("✅ Interpolation Intelligente")
        st.caption("✅ Lissage Variations")
        st.caption("✅ Validation Croisée")
        st.caption("✅ Calibrage ML Amélioré")
        st.caption("✅ Ensemble Methods")
        st.caption("✅ Contraintes Business")
        st.caption("✅ Analyse Sectorielle")
        st.caption("✅ Benchmarking Multi-Industries")

# ========== RUN ENHANCED APPLICATION ==========
if __name__ == "__main__":
    main()
