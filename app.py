from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from datetime import timezone
from dotenv import load_dotenv
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class AITradingSystem:
    """نظام الذكاء الاصطناعي المتقدم للتداول"""
    
    def __init__(self):
        self.ml_models = {}
        self.neural_models = {}
        self.time_series_models = {}
        self.reinforcement_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.performance_history = []
        self.model_metrics = {}
        
        # إعدادات النماذج
        self.model_config = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1},
            'neural_network': {'hidden_layers': (100, 50), 'activation': 'relu'},
            'lstm': {'units': 50, 'dropout': 0.2, 'epochs': 50},
            'cnn': {'filters': 32, 'kernel_size': 3, 'epochs': 30}
        }
        
        # تحميل النماذج المحفوظة
        self.load_saved_models()
    
    def load_saved_models(self):
        """تحميل النماذج المحفوظة"""
        try:
            if os.path.exists('ai_models.pkl'):
                with open('ai_models.pkl', 'rb') as f:
                    saved_data = pickle.load(f)
                    self.ml_models = saved_data.get('ml_models', {})
                    self.neural_models = saved_data.get('neural_models', {})
                    self.scalers = saved_data.get('scalers', {})
                    self.model_metrics = saved_data.get('metrics', {})
                print("✅ تم تحميل النماذج المحفوظة بنجاح")
        except Exception as e:
            print(f"⚠️ خطأ في تحميل النماذج: {e}")
    
    def save_models(self):
        """حفظ النماذج المدربة"""
        try:
            save_data = {
                'ml_models': self.ml_models,
                'neural_models': self.neural_models,
                'scalers': self.scalers,
                'metrics': self.model_metrics
            }
            with open('ai_models.pkl', 'wb') as f:
                pickle.dump(save_data, f)
            print("✅ تم حفظ النماذج بنجاح")
        except Exception as e:
            print(f"❌ خطأ في حفظ النماذج: {e}")
    
    def prepare_training_data(self, price_data, indicators_data):
        """إعداد بيانات التدريب"""
        try:
            # دمج بيانات الأسعار والمؤشرات
            features = []
            labels = []
            
            for i in range(len(price_data) - 1):
                feature_vector = []
                
                # إضافة بيانات الأسعار
                current_price = price_data[i]
                next_price = price_data[i + 1]
                
                feature_vector.extend([
                    current_price['open'],
                    current_price['high'],
                    current_price['low'],
                    current_price['close']
                ])
                
                # إضافة المؤشرات
                for indicator_name, indicator_data in indicators_data.items():
                    if indicator_data and len(indicator_data) > i:
                        indicator_value = self._extract_indicator_value(indicator_data[i])
                        if indicator_value is not None:
                            feature_vector.append(indicator_value)
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.append(0)
                
                # حساب العلامة (النتيجة)
                price_change = (next_price['close'] - current_price['close']) / current_price['close']
                if price_change > 0.001:  # ارتفاع أكثر من 0.1%
                    label = 1  # BUY
                elif price_change < -0.001:  # انخفاض أكثر من 0.1%
                    label = 0  # SELL
                else:
                    label = 2  # HOLD
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
        except Exception as e:
            print(f"❌ خطأ في إعداد بيانات التدريب: {e}")
            return None, None
    
    def _extract_indicator_value(self, indicator_data):
        """استخراج قيمة المؤشر"""
        if isinstance(indicator_data, dict):
            for key in ['value', 'close', 'sma', 'ema', 'rsi', 'macd']:
                if key in indicator_data:
                    try:
                        return float(indicator_data[key])
                    except:
                        continue
        return None
    
    def train_ml_models(self, features, labels):
        """تدريب نماذج Machine Learning"""
        try:
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['main'] = scaler
            
            # تدريب Random Forest
            rf_model = RandomForestClassifier(**self.model_config['random_forest'])
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            self.ml_models['random_forest'] = rf_model
            self.model_metrics['random_forest'] = {'accuracy': rf_accuracy}
            
            # تدريب Gradient Boosting
            gb_model = GradientBoostingClassifier(**self.model_config['gradient_boosting'])
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            gb_accuracy = accuracy_score(y_test, gb_pred)
            self.ml_models['gradient_boosting'] = gb_model
            self.model_metrics['gradient_boosting'] = {'accuracy': gb_accuracy}
            
            # تدريب Neural Network
            nn_model = MLPClassifier(**self.model_config['neural_network'])
            nn_model.fit(X_train_scaled, y_train)
            nn_pred = nn_model.predict(X_test_scaled)
            nn_accuracy = accuracy_score(y_test, nn_pred)
            self.ml_models['neural_network'] = nn_model
            self.model_metrics['neural_network'] = {'accuracy': nn_accuracy}
            
            print(f"✅ تم تدريب نماذج ML - RF: {rf_accuracy:.3f}, GB: {gb_accuracy:.3f}, NN: {nn_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نماذج ML: {e}")
            return False
    
    def train_lstm_model(self, features, labels):
        """تدريب نموذج LSTM للسلاسل الزمنية"""
        try:
            # إعادة تشكيل البيانات للـ LSTM
            sequence_length = 10
            X_lstm, y_lstm = [], []
            
            for i in range(sequence_length, len(features)):
                X_lstm.append(features[i-sequence_length:i])
                y_lstm.append(labels[i])
            
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_lstm, y_lstm, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler_lstm = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = scaler_lstm.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_lstm.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            self.scalers['lstm'] = scaler_lstm
            
            # بناء نموذج LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[-1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            # تدريب النموذج
            model.fit(X_train_scaled, y_train, 
                     epochs=self.model_config['lstm']['epochs'], 
                     batch_size=32, 
                     validation_data=(X_test_scaled, y_test),
                     verbose=0)
            
            # تقييم النموذج
            lstm_loss, lstm_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.neural_models['lstm'] = model
            self.model_metrics['lstm'] = {'accuracy': lstm_accuracy, 'loss': lstm_loss}
            
            print(f"✅ تم تدريب نموذج LSTM - دقة: {lstm_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نموذج LSTM: {e}")
            return False
    
    def train_cnn_model(self, features, labels):
        """تدريب نموذج CNN للرؤية الحاسوبية"""
        try:
            # تحويل البيانات إلى تنسيق مناسب للـ CNN
            sequence_length = 20
            X_cnn, y_cnn = [], []
            
            for i in range(sequence_length, len(features)):
                X_cnn.append(features[i-sequence_length:i])
                y_cnn.append(labels[i])
            
            X_cnn = np.array(X_cnn)
            y_cnn = np.array(y_cnn)
            
            # إعادة تشكيل البيانات للـ CNN
            X_cnn = X_cnn.reshape(X_cnn.shape[0], X_cnn.shape[1], 1)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_cnn, y_cnn, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler_cnn = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            X_train_scaled = scaler_cnn.fit_transform(X_train_reshaped)
            X_test_scaled = scaler_cnn.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            self.scalers['cnn'] = scaler_cnn
            
            # بناء نموذج CNN
            model = Sequential([
                Conv1D(32, 3, activation='relu', input_shape=(sequence_length, 1)),
                MaxPooling1D(2),
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            # تدريب النموذج
            model.fit(X_train_scaled, y_train, 
                     epochs=self.model_config['cnn']['epochs'], 
                     batch_size=32, 
                     validation_data=(X_test_scaled, y_test),
                     verbose=0)
            
            # تقييم النموذج
            cnn_loss, cnn_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            self.neural_models['cnn'] = model
            self.model_metrics['cnn'] = {'accuracy': cnn_accuracy, 'loss': cnn_loss}
            
            print(f"✅ تم تدريب نموذج CNN - دقة: {cnn_accuracy:.3f}")
            return True
        except Exception as e:
            print(f"❌ خطأ في تدريب نموذج CNN: {e}")
            return False
    
    def predict_with_ensemble(self, features):
        """التنبؤ باستخدام مجموعة النماذج"""
        try:
            predictions = []
            probabilities = []
            
            # تطبيع البيانات
            if 'main' in self.scalers:
                features_scaled = self.scalers['main'].transform([features])
            else:
                features_scaled = [features]
            
            # تنبؤات نماذج ML
            for model_name, model in self.ml_models.items():
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(features_scaled)[0]
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                    probabilities.append(pred_proba)
            
            # تنبؤات نماذج Neural Networks
            for model_name, model in self.neural_models.items():
                if model_name == 'lstm':
                    # إعداد بيانات LSTM
                    sequence_length = 10
                    if len(features) >= sequence_length:
                        lstm_input = np.array([features[-sequence_length:]])
                        if 'lstm' in self.scalers:
                            lstm_input_reshaped = lstm_input.reshape(-1, lstm_input.shape[-1])
                            lstm_input_scaled = self.scalers['lstm'].transform(lstm_input_reshaped)
                            lstm_input_scaled = lstm_input_scaled.reshape(lstm_input.shape)
                        else:
                            lstm_input_scaled = lstm_input
                        
                        pred_proba = model.predict(lstm_input_scaled, verbose=0)[0]
                        pred = np.argmax(pred_proba)
                        predictions.append(pred)
                        probabilities.append(pred_proba)
                
                elif model_name == 'cnn':
                    # إعداد بيانات CNN
                    sequence_length = 20
                    if len(features) >= sequence_length:
                        cnn_input = np.array([features[-sequence_length:]])
                        cnn_input = cnn_input.reshape(cnn_input.shape[0], cnn_input.shape[1], 1)
                        
                        if 'cnn' in self.scalers:
                            cnn_input_reshaped = cnn_input.reshape(-1, cnn_input.shape[-1])
                            cnn_input_scaled = self.scalers['cnn'].transform(cnn_input_reshaped)
                            cnn_input_scaled = cnn_input_scaled.reshape(cnn_input.shape)
                        else:
                            cnn_input_scaled = cnn_input
                        
                        pred_proba = model.predict(cnn_input_scaled, verbose=0)[0]
                        pred = np.argmax(pred_proba)
                        predictions.append(pred)
                        probabilities.append(pred_proba)
            
            # حساب المتوسط المرجح للتنبؤات
            if probabilities:
                avg_probabilities = np.mean(probabilities, axis=0)
                final_prediction = np.argmax(avg_probabilities)
                confidence = np.max(avg_probabilities) * 100
                
                # تحويل التوقع إلى نص
                signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
                signal = signal_map.get(final_prediction, 'HOLD')
                
                return {
                    'signal': signal,
                    'confidence': round(confidence, 2),
                    'probabilities': {
                        'BUY': round(avg_probabilities[1] * 100, 2),
                        'SELL': round(avg_probabilities[0] * 100, 2),
                        'HOLD': round(avg_probabilities[2] * 100, 2)
                    },
                    'model_predictions': len(predictions),
                    'ensemble_used': True
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                    'model_predictions': 0,
                    'ensemble_used': False
                }
        except Exception as e:
            print(f"❌ خطأ في التنبؤ: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                'model_predictions': 0,
                'ensemble_used': False
            }
    
    def analyze_chart_patterns(self, price_data):
        """تحليل الأنماط في الرسوم البيانية باستخدام الرؤية الحاسوبية"""
        try:
            # إنشاء رسم بياني
            plt.figure(figsize=(10, 6))
            prices = [p['close'] for p in price_data[-50:]]  # آخر 50 نقطة
            plt.plot(prices)
            plt.title('Price Chart Analysis')
            plt.xlabel('Time')
            plt.ylabel('Price')
            
            # حفظ الرسم كصورة
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # تحويل إلى صورة OpenCV
            img_data = img_buffer.getvalue()
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # تحليل الأنماط
            patterns = self._detect_chart_patterns(img)
            
            return {
                'patterns_detected': patterns,
                'chart_analysis': True
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الأنماط: {e}")
            return {
                'patterns_detected': [],
                'chart_analysis': False
            }
    
    def _detect_chart_patterns(self, img):
        """كشف الأنماط في الرسم البياني"""
        try:
            # تحويل إلى تدرج رمادي
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # كشف الحواف
            edges = cv2.Canny(gray, 50, 150)
            
            # كشف الخطوط
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            patterns = []
            if lines is not None:
                # تحليل اتجاه الخطوط
                upward_lines = 0
                downward_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if -45 < angle < 45:  # خط أفقي
                        continue
                    elif angle > 0:  # خط صاعد
                        upward_lines += 1
                    else:  # خط هابط
                        downward_lines += 1
                
                # تحديد الأنماط
                if upward_lines > downward_lines * 1.5:
                    patterns.append('Uptrend')
                elif downward_lines > upward_lines * 1.5:
                    patterns.append('Downtrend')
                else:
                    patterns.append('Sideways')
            
            return patterns
        except Exception as e:
            print(f"❌ خطأ في كشف الأنماط: {e}")
            return []
    
    def reinforcement_learning_update(self, action, reward, state):
        """تحديث نموذج التعلم المعزز"""
        try:
            # تطبيق خوارزمية Q-Learning مبسطة
            if not hasattr(self, 'q_table'):
                self.q_table = {}
            
            state_key = str(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            # تحديث قيمة Q
            learning_rate = 0.1
            discount_factor = 0.9
            
            old_value = self.q_table[state_key][action]
            self.q_table[state_key][action] = old_value + learning_rate * (reward - old_value)
            
            return True
        except Exception as e:
            print(f"❌ خطأ في تحديث التعلم المعزز: {e}")
            return False
    
    def get_reinforcement_prediction(self, state):
        """الحصول على تنبؤ من التعلم المعزز"""
        try:
            state_key = str(state)
            if state_key in self.q_table:
                q_values = self.q_table[state_key]
                best_action = max(q_values, key=q_values.get)
                confidence = abs(q_values[best_action]) / 10  # تحويل إلى نسبة مئوية
                return {
                    'signal': best_action,
                    'confidence': min(confidence * 100, 100),
                    'q_values': q_values
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'q_values': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                }
        except Exception as e:
            print(f"❌ خطأ في تنبؤ التعلم المعزز: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'q_values': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            }
    
    def get_performance_metrics(self):
        """الحصول على مقاييس الأداء"""
        return {
            'model_metrics': self.model_metrics,
            'performance_history': self.performance_history[-10:],  # آخر 10 قياسات
            'total_models': len(self.ml_models) + len(self.neural_models),
            'models_loaded': len(self.ml_models) > 0 or len(self.neural_models) > 0
        }


class TradingAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.twelvedata.com/time_series"
        self.is_running = False
        self.analysis_thread = None
        self.latest_results = {}
        
        # إضافة نظام الذكاء الاصطناعي
        self.ai_system = AITradingSystem()
        self.training_data = []
        self.ai_enabled = True
        
        # عداد الطلبات
        self.api_requests_count = 0
        self.api_requests_limit = 8  # الحد الأقصى للطلبات في الدقيقة
        self.last_reset_time = datetime.now()
        
        # جميع أزواج العملات المتاحة
        self.available_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF',
            'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'NZD/USD',
            'USD/CAD', 'EUR/AUD', 'GBP/AUD', 'EUR/CAD', 'GBP/CAD',
            'AUD/CAD', 'NZD/JPY', 'CHF/JPY', 'EUR/NZD', 'GBP/NZD'
        ]
        
        # المؤشرات المتاحة حسب المجموعات
        self.available_indicators = {
            # Trend Indicators (مؤشرات الاتجاه)
            'trend': {
                'sma': 'Simple Moving Average',
                'ema': 'Exponential Moving Average', 
                'wma': 'Weighted Moving Average',
                'dema': 'Double Exponential Moving Average',
                'tema': 'Triple Exponential Moving Average',
                'kama': 'Kaufman Adaptive Moving Average',
                'hma': 'Hull Moving Average',
                't3': 'T3 Moving Average'
            },
            # Momentum Indicators (مؤشرات الزخم)
            'momentum': {
                'rsi': 'Relative Strength Index',
                'stoch': 'Stochastic Oscillator',
                'stochrsi': 'Stochastic RSI',
                'willr': 'Williams %R',
                'macd': 'MACD',
                'ppo': 'Percentage Price Oscillator',
                'adx': 'Average Directional Index',
                'cci': 'Commodity Channel Index',
                'mom': 'Momentum',
                'roc': 'Rate of Change'
            },
            # Volatility Indicators (مؤشرات التذبذب)
            'volatility': {
                'bbands': 'Bollinger Bands',
                'atr': 'Average True Range',
                'stdev': 'Standard Deviation',
                'donchian': 'Donchian Channels'
            },
            # Volume Indicators (مؤشرات الحجم)
            'volume': {
                'obv': 'On Balance Volume',
                'cmf': 'Chaikin Money Flow',
                'ad': 'Accumulation/Distribution',
                'mfi': 'Money Flow Index',
                'emv': 'Ease of Movement',
                'fi': 'Force Index'
            },
            # Price Indicators (مؤشرات السعر)
            'price': {
                'avgprice': 'Average Price',
                'medprice': 'Median Price',
                'typprice': 'Typical Price',
                'wcprice': 'Weighted Close Price'
            },
            # Misc / Other Indicators (مؤشرات أخرى)
            'misc': {
                'sar': 'Parabolic SAR',
                'ultosc': 'Ultimate Oscillator',
                'tsi': 'True Strength Index'
            }
        }
    
    def set_api_key(self, api_key):
        """تعيين مفتاح API"""
        self.api_key = api_key
    
    def _check_api_limit(self):
        """التحقق من حد الطلبات"""
        current_time = datetime.now()
        
        # إعادة تعيين العداد كل دقيقة
        if (current_time - self.last_reset_time).seconds >= 60:
            self.api_requests_count = 0
            self.last_reset_time = current_time
        
        return self.api_requests_count < self.api_requests_limit
    
    def _increment_api_count(self):
        """زيادة عداد الطلبات"""
        self.api_requests_count += 1
    
    def get_api_status(self):
        """الحصول على حالة API"""
        current_time = datetime.now()
        time_remaining = 60 - (current_time - self.last_reset_time).seconds
        
        return {
            'requests_used': self.api_requests_count,
            'requests_limit': self.api_requests_limit,
            'requests_remaining': self.api_requests_limit - self.api_requests_count,
            'time_remaining': max(0, time_remaining),
            'can_make_request': self._check_api_limit()
        }
    
   
    #---------------------------------------------------
    def fetch_indicator_data(self, pair, indicator, interval='1min', **params):
        """جلب بيانات مؤشر من API TwelveData"""
        try:
            # التحقق من حد الطلبات
            if not self._check_api_limit():
                print(f"تم الوصول للحد الأقصى من الطلبات ({self.api_requests_limit})")
                return None
            
            # إعداد المعاملات الأساسية
            api_params = {
                'symbol': pair,
                'interval': interval,
                'apikey': self.api_key,
                **params
            }
            
            # بناء URL المؤشر
            indicator_url = f"https://api.twelvedata.com/{indicator}"
            
            # زيادة عداد الطلبات
            self._increment_api_count()
            print(f"طلب API #{self.api_requests_count}/{self.api_requests_limit}: {indicator} لـ {pair}")
            
            response = requests.get(indicator_url, params=api_params, timeout=10)
            
            if response.status_code != 200:
                print(f"خطأ HTTP عند جلب {indicator} لـ {pair}: حالة {response.status_code}")
                return None
            
            data = response.json()
            
            # تحقق من وجود خطأ في API
            if 'status' in data and data['status'] == 'error':
                error_msg = data.get('message', 'خطأ غير معروف')
                print(f"خطأ من API عند جلب {indicator} لـ {pair}: {error_msg}")
                
                # إذا كان الخطأ بسبب استنفاد الرصيد، انتظر دقيقة
                if 'API credits' in error_msg or 'limit' in error_msg:
                    print("انتظار 60 ثانية لتجنب استنفاد الرصيد...")
                    import time
                    time.sleep(60)
                
                return None
            
            if 'values' not in data or not data['values']:
                print(f"لا توجد بيانات لـ {indicator} لـ {pair}")
                return None
            
            # طباعة هيكل البيانات للتحقق
            if data['values'] and len(data['values']) > 0:
                print(f"هيكل بيانات {indicator}: {list(data['values'][0].keys())}")
            
            return data['values']
            
        except requests.exceptions.RequestException as e:
            print(f"خطأ في الاتصال عند جلب {indicator} لـ {pair}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"خطأ في فك JSON عند جلب {indicator} لـ {pair}: {str(e)}")
            return None
        except Exception as e:
            print(f"خطأ غير متوقع عند جلب {indicator} لـ {pair}: {str(e)}")
            return None

    def fetch_price_data(self, pair, interval='1min', outputsize=250):
        """جلب بيانات الأسعار الأساسية"""
        try:
            # التحقق من حد الطلبات
            if not self._check_api_limit():
                print(f"تم الوصول للحد الأقصى من الطلبات ({self.api_requests_limit})")
                return None
            
            params = {
                'symbol': pair,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            # زيادة عداد الطلبات
            self._increment_api_count()
            print(f"طلب API #{self.api_requests_count}/{self.api_requests_limit}: أسعار {pair}")
        
            response = requests.get(self.base_url, params=params, timeout=10)
        
            if response.status_code != 200:
                print(f"خطأ HTTP عند جلب {pair}: حالة {response.status_code}")
                return None
        
            data = response.json()
        
            if 'status' in data and data['status'] == 'error':
                print(f"خطأ من API عند جلب {pair}: {data.get('message', 'خطأ غير معروف')}")
                return None
        
            if 'values' not in data or not data['values']:
                print(f"لا توجد بيانات لـ {pair}")
                return None
        
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True) + timedelta(hours=2)  # Convert to Palestine time (UTC+2)
            df = df.sort_values('datetime')
        
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
            return df
    
        except requests.exceptions.RequestException as e:
            print(f"خطأ في الاتصال عند جلب {pair}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"خطأ في فك JSON عند جلب {pair}: {str(e)}")
            return None
        except Exception as e:
            print(f"خطأ غير متوقع عند جلب {pair}: {str(e)}")
            return None

    #------------------------------------------------------
    def fetch_indicators_data(self, pair, selected_indicators, interval='1min'):
        """جلب بيانات المؤشرات المختارة من API"""
        indicators_data = {}
        
        for category, indicators in selected_indicators.items():
            if not indicators:  # تخطي المجموعات الفارغة
                continue
                
            for indicator in indicators:
                try:
                    # إعداد المعاملات الخاصة بكل مؤشر
                    params = self._get_indicator_params(indicator)
                    
                    # جلب بيانات المؤشر
                    data = self.fetch_indicator_data(pair, indicator, interval, **params)
                    
                    if data:
                        indicators_data[indicator] = data
                        print(f"تم جلب {indicator} لـ {pair}")
                    else:
                        print(f"فشل جلب {indicator} لـ {pair}")
                    
                    # تأخير لتجنب استنفاد رصيد API (8 طلبات في الدقيقة)
                    import time
                    time.sleep(8)  # 8 ثواني بين كل طلب
                        
                except Exception as e:
                    print(f"خطأ في جلب {indicator} لـ {pair}: {str(e)}")
                    continue
        
        return indicators_data

    def _get_indicator_params(self, indicator):
        """إرجاع المعاملات الافتراضية لكل مؤشر"""
        default_params = {
            'sma': {'time_period': 20},
            'ema': {'time_period': 20},
            'wma': {'time_period': 20},
            'dema': {'time_period': 20},
            'tema': {'time_period': 20},
            'kama': {'time_period': 20},
            'hma': {'time_period': 20},
            't3': {'time_period': 20},
            'rsi': {'time_period': 14},
            'stoch': {'time_period': 14},
            'stochrsi': {'time_period': 14},
            'willr': {'time_period': 14},
            'macd': {'time_period': 14},
            'ppo': {'time_period': 14},
            'adx': {'time_period': 14},
            'cci': {'time_period': 14},
            'mom': {'time_period': 14},
            'roc': {'time_period': 14},
            'bbands': {'time_period': 20, 'series_type': 'close'},
            'atr': {'time_period': 14},
            'stdev': {'time_period': 20},
            'donchian': {'time_period': 20},
            'obv': {},
            'cmf': {'time_period': 20},
            'ad': {},
            'mfi': {'time_period': 14},
            'emv': {},
            'fi': {'time_period': 14},
            'avgprice': {},
            'medprice': {},
            'typprice': {},
            'wcprice': {},
            'sar': {'time_period': 14},
            'ultosc': {'time_period': 14},
            'tsi': {'time_period': 14}
        }
        
        return default_params.get(indicator, {})

    @staticmethod
    def safe_float(val):
        try:
            if pd.isna(val):
                return None
            return float(val)
        except:
            return None

    def _get_indicator_value(self, data, keys):
        """استخراج قيمة المؤشر من البيانات مع محاولة مفاتيح متعددة"""
        if not data or len(data) == 0:
            return None
            
        for key in keys:
            if key in data[0]:
                try:
                    return float(data[0][key])
                except:
                    continue
        return None

    def _get_indicator_values(self, data, keys, count=2):
        """استخراج قيم متعددة من المؤشر"""
        if not data or len(data) < count:
            return None
            
        values = []
        for i in range(count):
            for key in keys:
                if key in data[i]:
                    try:
                        values.append(float(data[i][key]))
                        break
                    except:
                        continue
            else:
                return None
        return values

    def analyze_trend_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الاتجاه"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Moving Averages
        for ma_type in ['sma', 'ema', 'wma', 'dema', 'tema', 'kama', 'hma', 't3']:
            if ma_type in indicators_data:
                ma_data = indicators_data[ma_type]
                ma_values = self._get_indicator_values(ma_data, ['value', ma_type, 'sma', 'ema'])
                
                if ma_values and len(ma_values) >= 2:
                    current_ma = ma_values[0]
                    prev_ma = ma_values[1]
                    current_price = float(price_data[0]['close'])
                    
                    # اتجاه المتوسط المتحرك
                    if current_ma > prev_ma:
                        signals.append(0.6)
                        details.append(f"📈 {ma_type.upper()} صاعد ({current_ma:.5f})")
                    elif current_ma < prev_ma:
                        signals.append(-0.6)
                        details.append(f"📉 {ma_type.upper()} هابط ({current_ma:.5f})")
                    
                    # موقع السعر من المتوسط
                    if current_price > current_ma:
                        signals.append(0.4)
                        details.append(f"📊 السعر فوق {ma_type.upper()}")
                    else:
                        signals.append(-0.4)
                        details.append(f"📊 السعر تحت {ma_type.upper()}")
                else:
                    print(f"لا يمكن استخراج قيم {ma_type}")
        
        return signals, details

    def analyze_momentum_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الزخم"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل RSI
        if 'rsi' in indicators_data:
            rsi_data = indicators_data['rsi']
            rsi_values = self._get_indicator_values(rsi_data, ['value', 'rsi'])
            
            if rsi_values and len(rsi_values) >= 2:
                current_rsi = rsi_values[0]
                prev_rsi = rsi_values[1]
                
                # تشبع شراء/بيع
                if current_rsi < 30:
                    if current_rsi < 20:
                        signals.append(1)
                        details.append(f"✅ RSI={current_rsi:.1f} (تشبع بيع شديد)")
                    else:
                        signals.append(0.7)
                        details.append(f"✅ RSI={current_rsi:.1f} (تشبع بيع)")
                elif current_rsi > 70:
                    if current_rsi > 80:
                        signals.append(-1)
                        details.append(f"❌ RSI={current_rsi:.1f} (تشبع شراء شديد)")
                    else:
                        signals.append(-0.7)
                        details.append(f"❌ RSI={current_rsi:.1f} (تشبع شراء)")
                elif current_rsi > 50 and prev_rsi <= 50:
                    signals.append(0.6)
                    details.append(f"📈 RSI={current_rsi:.1f} (زخم صعودي)")
                elif current_rsi < 50 and prev_rsi >= 50:
                    signals.append(-0.6)
                    details.append(f"📉 RSI={current_rsi:.1f} (زخم هبوطي)")
            else:
                print("لا يمكن استخراج قيم RSI")
        
        # تحليل MACD
        if 'macd' in indicators_data:
            macd_data = indicators_data['macd']
            macd_values = self._get_indicator_values(macd_data, ['value', 'macd'])
            
            if macd_values and len(macd_values) >= 2:
                current_macd = macd_values[0]
                prev_macd = macd_values[1]
                current_signal = self._get_indicator_value(macd_data, ['signal', 'macd_signal'])
                prev_signal = self._get_indicator_value(macd_data[1:], ['signal', 'macd_signal']) if len(macd_data) > 1 else 0
                
                # تقاطع MACD مع خط الإشارة
                if current_macd > current_signal and prev_macd <= prev_signal:
                    signals.append(1)
                    details.append(f"✅ MACD عبر فوق خط الإشارة")
                elif current_macd < current_signal and prev_macd >= prev_signal:
                    signals.append(-1)
                    details.append(f"❌ MACD عبر تحت خط الإشارة")
                elif current_macd > current_signal:
                    signals.append(0.5)
                    details.append(f"📈 MACD إيجابي")
                else:
                    signals.append(-0.5)
                    details.append(f"📉 MACD سلبي")
        
        # تحليل Stochastic
        if 'stoch' in indicators_data:
            stoch_data = indicators_data['stoch']
            if len(stoch_data) >= 1 and ('k' in stoch_data[0] or 'value' in stoch_data[0]):
                current_k = float(stoch_data[0].get('k', stoch_data[0].get('value', 0)))
                current_d = float(stoch_data[0].get('d', 0))
                
                if current_k < 20:
                    signals.append(0.7)
                    details.append(f"✅ Stochastic={current_k:.1f} (تشبع بيع)")
                elif current_k > 80:
                    signals.append(-0.7)
                    details.append(f"❌ Stochastic={current_k:.1f} (تشبع شراء)")
        
        return signals, details

    def analyze_volatility_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات التذبذب"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Bollinger Bands
        if 'bbands' in indicators_data:
            bb_data = indicators_data['bbands']
            if len(bb_data) >= 1 and ('upper_band' in bb_data[0] or 'value' in bb_data[0]):
                current_price = float(price_data[0]['close'])
                upper_band = float(bb_data[0].get('upper_band', bb_data[0].get('value', 0)))
                middle_band = float(bb_data[0].get('middle_band', 0))
                lower_band = float(bb_data[0].get('lower_band', 0))
                
                # موقع السعر من البولينجر
                if current_price >= upper_band:
                    signals.append(-0.8)
                    details.append(f"❌ السعر عند الحد العلوي - احتمال ارتداد")
                elif current_price <= lower_band:
                    signals.append(0.8)
                    details.append(f"✅ السعر عند الحد السفلي - احتمال ارتداد")
                elif current_price > middle_band:
                    signals.append(0.3)
                    details.append(f"📊 السعر في المنطقة العليا")
                else:
                    signals.append(-0.3)
                    details.append(f"📊 السعر في المنطقة السفلى")
        
        # تحليل ATR
        if 'atr' in indicators_data:
            atr_data = indicators_data['atr']
            if len(atr_data) >= 2 and 'value' in atr_data[0] and 'value' in atr_data[1]:
                current_atr = float(atr_data[0]['value'])
                prev_atr = float(atr_data[1]['value'])
                atr_change = ((current_atr - prev_atr) / prev_atr * 100) if prev_atr != 0 else 0
                
                if atr_change > 10:
                    details.append(f"⚠️ ATR={current_atr:.5f} (تقلب متزايد +{atr_change:.1f}%)")
                elif atr_change < -10:
                    details.append(f"📊 ATR={current_atr:.5f} (تقلب متناقص {atr_change:.1f}%)")
                else:
                    details.append(f"📊 ATR={current_atr:.5f} (تقلب مستقر)")
        
        return signals, details

    def analyze_volume_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات الحجم"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Money Flow Index
        if 'mfi' in indicators_data:
            mfi_data = indicators_data['mfi']
            if len(mfi_data) >= 1 and 'value' in mfi_data[0]:
                current_mfi = float(mfi_data[0]['value'])
                
                if current_mfi < 20:
                    signals.append(0.7)
                    details.append(f"✅ MFI={current_mfi:.1f} (تشبع بيع)")
                elif current_mfi > 80:
                    signals.append(-0.7)
                    details.append(f"❌ MFI={current_mfi:.1f} (تشبع شراء)")
                elif current_mfi > 50:
                    signals.append(0.3)
                    details.append(f"📈 MFI={current_mfi:.1f} (زخم إيجابي)")
                else:
                    signals.append(-0.3)
                    details.append(f"📉 MFI={current_mfi:.1f} (زخم سلبي)")
        
        # تحليل Chaikin Money Flow
        if 'cmf' in indicators_data:
            cmf_data = indicators_data['cmf']
            if len(cmf_data) >= 1 and 'value' in cmf_data[0]:
                current_cmf = float(cmf_data[0]['value'])
                
                if current_cmf > 0.1:
                    signals.append(0.5)
                    details.append(f"📈 CMF={current_cmf:.3f} (تدفق أموال إيجابي)")
                elif current_cmf < -0.1:
                    signals.append(-0.5)
                    details.append(f"📉 CMF={current_cmf:.3f} (تدفق أموال سلبي)")
        
        return signals, details

    def analyze_price_indicators(self, indicators_data, price_data):
        """تحليل مؤشرات السعر"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Average Price
        if 'avgprice' in indicators_data:
            avgprice_data = indicators_data['avgprice']
            if len(avgprice_data) >= 1:
                current_avg = float(avgprice_data[0]['value'])
                current_close = float(price_data[0]['close'])
                
                if current_close > current_avg:
                    signals.append(0.4)
                    details.append(f"📈 السعر فوق المتوسط")
                else:
                    signals.append(-0.4)
                    details.append(f"📉 السعر تحت المتوسط")
        
        return signals, details

    def analyze_misc_indicators(self, indicators_data, price_data):
        """تحليل المؤشرات الأخرى"""
        signals = []
        details = []
        
        if not indicators_data:
            return signals, details
            
        # تحليل Parabolic SAR
        if 'sar' in indicators_data:
            sar_data = indicators_data['sar']
            if len(sar_data) >= 1:
                current_sar = float(sar_data[0]['value'])
                current_close = float(price_data[0]['close'])
                
                if current_close > current_sar:
                    signals.append(0.6)
                    details.append(f"📈 SAR={current_sar:.5f} (إشارة صعود)")
                else:
                    signals.append(-0.6)
                    details.append(f"📉 SAR={current_sar:.5f} (إشارة هبوط)")
        
        # تحليل Ultimate Oscillator
        if 'ultosc' in indicators_data:
            ultosc_data = indicators_data['ultosc']
            if len(ultosc_data) >= 1:
                current_ultosc = float(ultosc_data[0]['value'])
                
                if current_ultosc < 30:
                    signals.append(0.7)
                    details.append(f"✅ Ultimate={current_ultosc:.1f} (تشبع بيع)")
                elif current_ultosc > 70:
                    signals.append(-0.7)
                    details.append(f"❌ Ultimate={current_ultosc:.1f} (تشبع شراء)")
        
        return signals, details
    #--------------------------------------------------
    def train_ai_models(self, pair, historical_data):
        """تدريب نماذج الذكاء الاصطناعي"""
        try:
            if not historical_data or len(historical_data) < 100:
                print(f"بيانات غير كافية لتدريب النماذج لـ {pair}")
                return False
            
            # إعداد بيانات التدريب
            price_data = []
            indicators_data = {}
            
            for data_point in historical_data:
                price_data.append({
                    'open': float(data_point.get('open', 0)),
                    'high': float(data_point.get('high', 0)),
                    'low': float(data_point.get('low', 0)),
                    'close': float(data_point.get('close', 0))
                })
            
            # تحضير بيانات المؤشرات
            features, labels = self.ai_system.prepare_training_data(price_data, indicators_data)
            
            if features is None or len(features) < 50:
                print(f"بيانات تدريب غير كافية لـ {pair}")
                return False
            
            # تدريب النماذج
            print(f"🤖 بدء تدريب النماذج لـ {pair}...")
            
            # تدريب نماذج ML
            ml_success = self.ai_system.train_ml_models(features, labels)
            
            # تدريب نماذج Neural Networks
            lstm_success = self.ai_system.train_lstm_model(features, labels)
            cnn_success = self.ai_system.train_cnn_model(features, labels)
            
            # حفظ النماذج
            self.ai_system.save_models()
            
            print(f"✅ تم تدريب النماذج لـ {pair} - ML: {ml_success}, LSTM: {lstm_success}, CNN: {cnn_success}")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تدريب النماذج لـ {pair}: {e}")
            return False

    def generate_signal(self, indicators_data, price_data, selected_indicators):
        """توليد إشارة التداول مع الذكاء الاصطناعي"""
        if not indicators_data or not price_data:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'بيانات غير كافية',
                'indicators': {},
                'ai_prediction': None
            }

        all_signals = []
        all_details = []

        # تحليل كل مجموعة مؤشرات
        for category, indicators in selected_indicators.items():
            if not indicators:  # تخطي المجموعات الفارغة
                continue

            category_indicators = {ind: indicators_data.get(ind, []) for ind in indicators if ind in indicators_data}

            if category == 'trend':
                signals, details = self.analyze_trend_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'momentum':
                signals, details = self.analyze_momentum_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volatility':
                signals, details = self.analyze_volatility_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'volume':
                signals, details = self.analyze_volume_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'price':
                signals, details = self.analyze_price_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)
            elif category == 'misc':
                signals, details = self.analyze_misc_indicators(category_indicators, price_data)
                all_signals.extend(signals)
                all_details.extend(details)

        # تحليل الذكاء الاصطناعي
        ai_prediction = None
        if self.ai_enabled and (self.ai_system.ml_models or self.ai_system.neural_models):
            try:
                # إعداد بيانات للتنبؤ
                current_features = []
                current_price = price_data[0]
                current_features.extend([
                    current_price['open'],
                    current_price['high'],
                    current_price['low'],
                    current_price['close']
                ])
                
                # إضافة قيم المؤشرات
                for category, indicators in selected_indicators.items():
                    for indicator in indicators:
                        if indicator in indicators_data and indicators_data[indicator]:
                            indicator_value = self.ai_system._extract_indicator_value(indicators_data[indicator][0])
                            current_features.append(indicator_value if indicator_value is not None else 0)
                        else:
                            current_features.append(0)
                
                # الحصول على تنبؤ الذكاء الاصطناعي
                ai_prediction = self.ai_system.predict_with_ensemble(current_features)
                
                # تحليل الأنماط في الرسم البياني
                chart_analysis = self.ai_system.analyze_chart_patterns(price_data)
                ai_prediction['chart_patterns'] = chart_analysis
                
                # تنبؤ التعلم المعزز
                state_vector = current_features[:10]  # أول 10 قيم كحالة
                rl_prediction = self.ai_system.get_reinforcement_prediction(state_vector)
                ai_prediction['reinforcement_learning'] = rl_prediction
                
            except Exception as e:
                print(f"❌ خطأ في تنبؤ الذكاء الاصطناعي: {e}")
                ai_prediction = {
                    'signal': 'HOLD',
                    'confidence': 0,
                    'probabilities': {'BUY': 0, 'SELL': 0, 'HOLD': 100},
                    'ensemble_used': False
                }

        # دمج النتائج التقليدية مع الذكاء الاصطناعي
        traditional_avg = np.mean(all_signals) if all_signals else 0
        threshold = 0.35
        
        # إذا كان لدينا تنبؤ من الذكاء الاصطناعي، ندمجه مع النتائج التقليدية
        if ai_prediction and ai_prediction.get('ensemble_used', False):
            ai_signal = ai_prediction['signal']
            ai_confidence = ai_prediction['confidence']
            
            # دمج الإشارات (وزن 70% للذكاء الاصطناعي، 30% للتحليل التقليدي)
            if ai_signal == 'BUY':
                ai_weight = 0.7
                traditional_weight = 0.3
                final_signal = 'CALL'
                signal_text = f'صعود (CALL) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence * ai_weight + abs(traditional_avg) * 100 * traditional_weight
            elif ai_signal == 'SELL':
                ai_weight = 0.7
                traditional_weight = 0.3
                final_signal = 'PUT'
                signal_text = f'هبوط (PUT) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence * ai_weight + abs(traditional_avg) * 100 * traditional_weight
            else:
                final_signal = 'HOLD'
                signal_text = f'انتظار (HOLD) 🤖 AI: {ai_confidence:.1f}%'
                confidence = ai_confidence
        else:
            # استخدام التحليل التقليدي فقط
            confidence = min(abs(traditional_avg) * 100, 100)
            
            if traditional_avg > threshold:
                final_signal = 'CALL'
                signal_text = 'صعود (CALL) 🟢'
            elif traditional_avg < -threshold:
                final_signal = 'PUT'
                signal_text = 'هبوط (PUT) 🔴'
            else:
                final_signal = 'HOLD'
                signal_text = 'انتظار (HOLD) ⚪'

        # جمع قيم المؤشرات
        indicators_values = {}
        for category, indicators in selected_indicators.items():
            for indicator in indicators:
                if indicator in indicators_data and indicators_data[indicator]:
                    latest_data = indicators_data[indicator][0]
                    if isinstance(latest_data, dict):
                        for key, value in latest_data.items():
                            if key != 'datetime':
                                indicators_values[f"{indicator}_{key}"] = self.safe_float(value)

        # إضافة تفاصيل الذكاء الاصطناعي
        ai_details = []
        if ai_prediction:
            if ai_prediction.get('probabilities'):
                ai_details.append(f"🤖 AI احتمالات: BUY {ai_prediction['probabilities'].get('BUY', 0):.1f}%, SELL {ai_prediction['probabilities'].get('SELL', 0):.1f}%, HOLD {ai_prediction['probabilities'].get('HOLD', 0):.1f}%")
            
            if ai_prediction.get('chart_patterns', {}).get('patterns_detected'):
                patterns = ai_prediction['chart_patterns']['patterns_detected']
                ai_details.append(f"📊 أنماط الرسم: {', '.join(patterns)}")
            
            if ai_prediction.get('reinforcement_learning', {}).get('signal'):
                rl_signal = ai_prediction['reinforcement_learning']['signal']
                rl_conf = ai_prediction['reinforcement_learning']['confidence']
                ai_details.append(f"🧠 التعلم المعزز: {rl_signal} ({rl_conf:.1f}%)")

        # دمج التفاصيل
        all_details.extend(ai_details)

        return {
            'signal': final_signal,
            'signal_text': signal_text,
            'confidence': round(confidence, 1),
            'reason': ' | '.join(all_details[:4]) if all_details else 'لا توجد إشارات',
            'all_details': all_details,
            'indicators': indicators_values,
            'price': {
                'open': self.safe_float(price_data[0]['open']),
                'high': self.safe_float(price_data[0]['high']),
                'low': self.safe_float(price_data[0]['low']),
                'close': self.safe_float(price_data[0]['close'])
            },
            'last_candle_time': price_data[0]['datetime'],
            'ai_prediction': ai_prediction,
            'ai_enabled': self.ai_enabled
        }
    def analyze_pair(self, pair, period, selected_indicators, interval='1min'):
        """تحليل زوج واحد"""
        try:
            # جلب بيانات الأسعار
            price_df = self.fetch_price_data(pair, interval, period)
            if price_df is None or len(price_df) == 0:
                print(f"لا توجد بيانات أسعار لـ {pair}")
                return None

            # جلب بيانات المؤشرات
            indicators_data = self.fetch_indicators_data(pair, selected_indicators, interval)
            if not indicators_data:
                print(f"لا توجد بيانات مؤشرات لـ {pair}")
                return None

            # تحويل بيانات الأسعار إلى تنسيق مناسب
            price_data = []
            for _, row in price_df.tail(50).iterrows():
                price_data.append({
                    'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'open': self.safe_float(row['open']),
                    'high': self.safe_float(row['high']),
                    'low': self.safe_float(row['low']),
                    'close': self.safe_float(row['close'])
                })

            if not price_data:
                print(f"بيانات الأسعار فارغة لـ {pair}")
                return None

            # توليد الإشارة
            analysis = self.generate_signal(indicators_data, price_data, selected_indicators)
            
            # إضافة بيانات الشموع للرسم البياني
            chart_data = []
            for row in price_data:
                chart_data.append({
                    'time': row['datetime'].split(' ')[1][:5],  # الوقت فقط
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close']
                })
            analysis['chart_data'] = chart_data
            return analysis
            
        except Exception as e:
            print(f"خطأ في تحليل {pair}: {str(e)}")
            return None
    def start_analysis(self, pairs, period, interval_minutes, selected_indicators):
        """تخزين الإعدادات دون تشغيل تحليل مستمر"""
        self.latest_results = {}  # إعادة تعيين النتائج
        self.is_running = True
        self.analysis_config = {
            'pairs': pairs,
            'period': period,
            'interval': interval_minutes,
            'indicators': selected_indicators
        }
        return True


    def stop_analysis(self):
        """إيقاف التحليل"""
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=2)
        return True

load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")
analyzer = TradingAnalyzer(API_KEY)

app = Flask(__name__)

CORS(app)

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/pairs')
def get_pairs():
    """الحصول على قائمة الأزواج المتاحة"""
    return jsonify(analyzer.available_pairs)

@app.route('/api/indicators')
def get_indicators():
    """الحصول على قائمة المؤشرات المتاحة"""
    return jsonify(analyzer.available_indicators)

@app.route('/api/config', methods=['POST'])
def set_config():
    """تعيين إعدادات API"""
    data = request.json
    analyzer.set_api_key(data.get('api_key', ''))
    return jsonify({'status': 'success'})

@app.route('/api/start', methods=['POST'])
def start_analysis():
    """بدء التحليل"""
    data = request.json
    pairs = data.get('pairs', [])
    period = data.get('period', 250)
    interval = data.get('interval', 1)
    selected_indicators = data.get('indicators', ['RSI', 'MACD'])
    
    if analyzer.start_analysis(pairs, period, interval, selected_indicators):
        return jsonify({'status': 'success', 'message': 'تم بدء التحليل'})
    else:
        return jsonify({'status': 'error', 'message': 'التحليل يعمل بالفعل'})

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    """إيقاف التحليل"""
    if analyzer.stop_analysis():
        return jsonify({'status': 'success', 'message': 'تم إيقاف التحليل'})
    else:
        return jsonify({'status': 'error', 'message': 'فشل إيقاف التحليل'})

@app.route('/api/status')
def get_status():
    """الحصول على حالة التحليل"""
    return jsonify({
        'is_running': analyzer.is_running,
        'results': analyzer.latest_results
    })

@app.route('/api/requests-status')
def get_requests_status():
    """الحصول على حالة طلبات API"""
    return jsonify(analyzer.get_api_status())

@app.route('/api/ai/train', methods=['POST'])
def train_ai_models():
    """تدريب نماذج الذكاء الاصطناعي"""
    try:
        data = request.json
        pair = data.get('pair', 'EUR/USD')
        period = data.get('period', 500)
        
        # جلب بيانات تاريخية للتدريب
        price_df = analyzer.fetch_price_data(pair, '1min', period)
        if price_df is None:
            return jsonify({'status': 'error', 'message': 'فشل في جلب البيانات التاريخية'})
        
        # تحويل البيانات
        historical_data = []
        for _, row in price_df.iterrows():
            historical_data.append({
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            })
        
        # تدريب النماذج
        success = analyzer.train_ai_models(pair, historical_data)
        
        if success:
            return jsonify({
                'status': 'success', 
                'message': f'تم تدريب النماذج بنجاح لـ {pair}',
                'metrics': analyzer.ai_system.get_performance_metrics()
            })
        else:
            return jsonify({'status': 'error', 'message': 'فشل في تدريب النماذج'})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في التدريب: {str(e)}'})

@app.route('/api/ai/status')
def get_ai_status():
    """الحصول على حالة نظام الذكاء الاصطناعي"""
    try:
        metrics = analyzer.ai_system.get_performance_metrics()
        return jsonify({
            'status': 'success',
            'ai_enabled': analyzer.ai_enabled,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على حالة AI: {str(e)}'})

@app.route('/api/ai/toggle', methods=['POST'])
def toggle_ai():
    """تفعيل/إلغاء تفعيل الذكاء الاصطناعي"""
    try:
        data = request.json
        enabled = data.get('enabled', True)
        analyzer.ai_enabled = enabled
        
        return jsonify({
            'status': 'success',
            'message': f'تم {"تفعيل" if enabled else "إلغاء تفعيل"} الذكاء الاصطناعي',
            'ai_enabled': analyzer.ai_enabled
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في تغيير حالة AI: {str(e)}'})

@app.route('/api/ai/performance')
def get_ai_performance():
    """الحصول على أداء النماذج"""
    try:
        performance = analyzer.ai_system.get_performance_metrics()
        return jsonify({
            'status': 'success',
            'performance': performance
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'خطأ في الحصول على الأداء: {str(e)}'})


@app.route('/api/results')
def get_results():
    """تشغيل التحليل عند الطلب"""
    try:
        if not analyzer.is_running or not hasattr(analyzer, 'analysis_config'):
            return jsonify({'status': 'error', 'message': 'التحليل غير مفعل حالياً'})

        config = analyzer.analysis_config
        results = {}
        current_time = datetime.now(timezone.utc) + timedelta(hours=2)

        for pair in config['pairs']:
            try:
                # تحويل الفترة إلى دقائق
                interval_str = f"{config['interval']}min"
                analysis = analyzer.analyze_pair(
                    pair,
                    config['period'],
                    config['indicators'],
                    interval_str
                )
                if analysis:
                    next_candle = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
                    end_time = next_candle + timedelta(minutes=config['interval'])

                    analysis['trade_timing'] = {
                        'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_time': next_candle.strftime('%H:%M:%S'),
                        'exit_time': end_time.strftime('%H:%M:%S'),
                        'duration': config['interval']
                    }

                    results[pair] = analysis
                else:
                    print(f"فشل تحليل {pair}")
            except Exception as e:
                print(f"خطأ في تحليل {pair}: {str(e)}")
                continue

        analyzer.latest_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis': results,
            'selected_indicators': config['indicators'],
            'status': 'success'
        }
        return jsonify(analyzer.latest_results)
        
    except Exception as e:
        print(f"خطأ في API results: {str(e)}")
        return jsonify({'status': 'error', 'message': f'خطأ في الخادم: {str(e)}'})
if __name__ == '__main__':
    print("🚀 بدء خادم التداول المتقدم...")
    print("📱 افتح المتصفح على: http://localhost:5000")
    print("📊 المؤشرات المتاحة:")
    print("   🔄 Trend: SMA, EMA, WMA, DEMA, TEMA, KAMA, HMA, T3")
    print("   ⚡ Momentum: RSI, STOCH, STOCHRSI, WILLR, MACD, PPO, ADX, CCI, MOM, ROC")
    print("   📈 Volatility: BBANDS, ATR, STDEV, DONCHIAN")
    print("   📊 Volume: OBV, CMF, AD, MFI, EMV, FI")
    print("   💰 Price: AVGPRICE, MEDPRICE, TYPPRICE, WCPRICE")
    print("   🎯 Misc: SAR, ULTOSC, TSI")
    app.run(debug=True, host='0.0.0.0', port=5000)