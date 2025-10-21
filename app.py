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


class TradingAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.twelvedata.com/time_series"
        self.is_running = False
        self.analysis_thread = None
        self.latest_results = {}
        
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
    def generate_signal(self, indicators_data, price_data, selected_indicators):
        """توليد إشارة التداول"""
        if not indicators_data or not price_data:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'بيانات غير كافية',
                'indicators': {}
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

        # الاستراتيجية الافتراضية (متوازنة)
        avg_signal = np.mean(all_signals) if all_signals else 0
        threshold = 0.35
        confidence_multiplier = 1.0

        # حساب الإشارة النهائية
        if len(all_signals) == 0:
            avg_signal = 0

        confidence = min(abs(avg_signal) * 100 * confidence_multiplier, 100)

        if avg_signal > threshold:
            final_signal = 'CALL'
            signal_text = 'صعود (CALL) 🟢'
        elif avg_signal < -threshold:
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
            'last_candle_time': price_data[0]['datetime']
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