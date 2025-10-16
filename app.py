from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
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
        
        # جميع أزواج العملات المتاحة
        self.available_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CHF',
            'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'NZD/USD',
            'USD/CAD', 'EUR/AUD', 'GBP/AUD', 'EUR/CAD', 'GBP/CAD',
            'AUD/CAD', 'NZD/JPY', 'CHF/JPY', 'EUR/NZD', 'GBP/NZD'
        ]
        
        # المؤشرات المتاحة
        self.available_indicators = {
            'EMA50': 'EMA50',
            'EMA200': 'EMA200',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'BB': 'Bollinger Bands',
            'ATR': 'ATR'
        }
    
    def set_api_key(self, api_key):
        """تعيين مفتاح API"""
        self.api_key = api_key
    
   
    #---------------------------------------------------
    def fetch_data(self, pair, outputsize=250):
        """جلب بيانات من API مع معالجة الأخطاء"""
        try:
            params = {
                'symbol': pair,
                'interval': '1min',
                'apikey': self.api_key,
                'outputsize': outputsize
            }
        
            response = requests.get(self.base_url, params=params, timeout=10)
        
            # تحقق من حالة الاستجابة
            if response.status_code != 200:
                print(f"خطأ HTTP عند جلب {pair}: حالة {response.status_code}")
                return None
        
            data = response.json()
        
        # تحقق إذا كانت هناك رسالة خطأ في الـ API
            if 'status' in data and data['status'] == 'error':
                print(f"خطأ من API عند جلب {pair}: {data.get('message', 'خطأ غير معروف')}")
                return None
        
            if 'values' not in data or not data['values']:
                print(f"لا توجد بيانات لـ {pair}")
                return None
        
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
            return df
    
        except requests.exceptions.RequestException as e:
        # مشاكل الشبكة أو انتهاء المهلة
            print(f"خطأ في الاتصال عند جلب {pair}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            print(f"خطأ في فك JSON عند جلب {pair}: {str(e)}")
            return None
        except Exception as e:
            print(f"خطأ غير متوقع عند جلب {pair}: {str(e)}")
            return None

    #------------------------------------------------------
    def calculate_indicators(self, df, selected_indicators):
        """حساب المؤشرات المختارة فقط"""
        if df is None or len(df) < 200:
            return None
            
        df = df.copy()
        
        # EMA50
        if 'EMA50' in selected_indicators:
            ema50 = EMAIndicator(close=df['close'], window=50)
            df['EMA50'] = ema50.ema_indicator()
        
        # EMA200
        if 'EMA200' in selected_indicators:
            ema200 = EMAIndicator(close=df['close'], window=200)
            df['EMA200'] = ema200.ema_indicator()
        
        # RSI
        if 'RSI' in selected_indicators:
            rsi = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi.rsi()
        
        # MACD
        if 'MACD' in selected_indicators:
            macd = MACD(close=df['close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        if 'BB' in selected_indicators:
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = bb.bollinger_wband()
            df['BB_pband'] = bb.bollinger_pband()
        
        # ATR
        if 'ATR' in selected_indicators:
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ATR'] = atr.average_true_range()
        
        return df

    @staticmethod
    def safe_float(val):
        try:
            if pd.isna(val):
                return None
            return float(val)
        except:
            return None

    def analyze_ema(self, df, latest, prev):
        """تحليل EMA"""
        signals = []
        details = []
        
        if 'EMA50' in df.columns and 'EMA200' in df.columns:
            ema50 = latest['EMA50']
            ema200 = latest['EMA200']
            price = latest['close']
            
            # Golden Cross / Death Cross
            if latest['EMA50'] > latest['EMA200'] and prev['EMA50'] <= prev['EMA200']:
                signals.append(1)
                details.append("✅ Golden Cross: EMA50 عبر فوق EMA200 (إشارة صعود قوية جداً)")
            elif latest['EMA50'] < latest['EMA200'] and prev['EMA50'] >= prev['EMA200']:
                signals.append(-1)
                details.append("❌ Death Cross: EMA50 عبر تحت EMA200 (إشارة هبوط قوية جداً)")
            
            # موقع السعر من EMA
            if price > ema50 > ema200:
                signals.append(0.7)
                details.append(f"📈 السعر فوق EMA50 و EMA200 (اتجاه صعودي قوي)")
            elif price < ema50 < ema200:
                signals.append(-0.7)
                details.append(f"📉 السعر تحت EMA50 و EMA200 (اتجاه هبوطي قوي)")
            elif price > ema50 and ema50 < ema200:
                signals.append(0.4)
                details.append(f"📊 السعر فوق EMA50 لكن تحت EMA200 (اتجاه صعودي ضعيف)")
            elif price < ema50 and ema50 > ema200:
                signals.append(-0.4)
                details.append(f"📊 السعر تحت EMA50 لكن فوق EMA200 (اتجاه هبوطي ضعيف)")
        
        elif 'EMA50' in df.columns:
            if latest['close'] > latest['EMA50'] and prev['close'] <= prev['EMA50']:
                signals.append(0.8)
                details.append("✅ السعر عبر فوق EMA50 (إشارة صعود)")
            elif latest['close'] < latest['EMA50'] and prev['close'] >= prev['EMA50']:
                signals.append(-0.8)
                details.append("❌ السعر عبر تحت EMA50 (إشارة هبوط)")
            elif latest['close'] > latest['EMA50']:
                signals.append(0.5)
                details.append(f"📈 السعر فوق EMA50")
            else:
                signals.append(-0.5)
                details.append(f"📉 السعر تحت EMA50")
        
        return signals, details

    def analyze_rsi(self, df, latest, prev):
        """تحليل RSI"""
        signals = []
        details = []
        
        if 'RSI' in df.columns:
            rsi = latest['RSI']
            prev_rsi = prev['RSI']
            
            # تشبع شراء/بيع
            if rsi < 30:
                if rsi < 20:
                    signals.append(1)
                    details.append(f"✅ RSI={rsi:.1f} (تشبع بيع شديد - فرصة شراء قوية)")
                else:
                    signals.append(0.7)
                    details.append(f"✅ RSI={rsi:.1f} (تشبع بيع - فرصة شراء)")
            elif rsi > 70:
                if rsi > 80:
                    signals.append(-1)
                    details.append(f"❌ RSI={rsi:.1f} (تشبع شراء شديد - فرصة بيع قوية)")
                else:
                    signals.append(-0.7)
                    details.append(f"❌ RSI={rsi:.1f} (تشبع شراء - فرصة بيع)")
            
            # اختراق مستويات
            elif rsi > 50 and prev_rsi <= 50:
                signals.append(0.6)
                details.append(f"📈 RSI={rsi:.1f} (اختراق فوق 50 - زخم صعودي)")
            elif rsi < 50 and prev_rsi >= 50:
                signals.append(-0.6)
                details.append(f"📉 RSI={rsi:.1f} (كسر تحت 50 - زخم هبوطي)")
            
            # زخم عام
            elif rsi > 55:
                signals.append(0.4)
                details.append(f"📊 RSI={rsi:.1f} (زخم إيجابي)")
            elif rsi < 45:
                signals.append(-0.4)
                details.append(f"📊 RSI={rsi:.1f} (زخم سلبي)")
            else:
                signals.append(0)
                details.append(f"⚪ RSI={rsi:.1f} (محايد)")
        
        return signals, details

    def analyze_macd(self, df, latest, prev):
        """تحليل MACD"""
        signals = []
        details = []
        
        if 'MACD' in df.columns:
            macd = latest['MACD']
            signal = latest['MACD_signal']
            diff = latest['MACD_diff']
            
            prev_macd = prev['MACD']
            prev_signal = prev['MACD_signal']
            prev_diff = prev['MACD_diff']
            
            # تقاطع MACD مع خط الإشارة
            if macd > signal and prev_macd <= prev_signal:
                signals.append(1)
                details.append(f"✅ MACD عبر فوق خط الإشارة (صعودي قوي)")
            elif macd < signal and prev_macd >= prev_signal:
                signals.append(-1)
                details.append(f"❌ MACD عبر تحت خط الإشارة (هبوطي قوي)")
            
            # قوة الاتجاه من histogram
            elif diff > 0:
                if diff > prev_diff:
                    signals.append(0.6)
                    details.append(f"📈 MACD إيجابي ومتزايد (قوة={abs(diff):.5f})")
                else:
                    signals.append(0.3)
                    details.append(f"📊 MACD إيجابي (قوة={abs(diff):.5f})")
            else:
                if diff < prev_diff:
                    signals.append(-0.6)
                    details.append(f"📉 MACD سلبي ومتناقص (قوة={abs(diff):.5f})")
                else:
                    signals.append(-0.3)
                    details.append(f"📊 MACD سلبي (قوة={abs(diff):.5f})")
        
        return signals, details

    def analyze_bollinger(self, df, latest, prev):
        """تحليل Bollinger Bands"""
        signals = []
        details = []
        
        if 'BB_upper' in df.columns:
            price = latest['close']
            upper = latest['BB_upper']
            middle = latest['BB_middle']
            lower = latest['BB_lower']
            bb_pband = latest['BB_pband']
            width = latest['BB_width']
            
            # موقع السعر من البولينجر
            if price >= upper:
                signals.append(-0.8)
                details.append(f"❌ السعر عند/فوق الحد العلوي ({bb_pband:.0f}%) - احتمال ارتداد هبوطي")
            elif price <= lower:
                signals.append(0.8)
                details.append(f"✅ السعر عند/تحت الحد السفلي ({bb_pband:.0f}%) - احتمال ارتداد صعودي")
            
            # اختراق من خارج الحدود
            elif prev['close'] >= prev['BB_upper'] and price < upper:
                signals.append(-0.5)
                details.append(f"📉 ارتداد من الحد العلوي ({bb_pband:.0f}%)")
            elif prev['close'] <= prev['BB_lower'] and price > lower:
                signals.append(0.5)
                details.append(f"📈 ارتداد من الحد السفلي ({bb_pband:.0f}%)")
            
            # موقع السعر النسبي
            elif bb_pband > 70:
                signals.append(-0.4)
                details.append(f"📊 السعر في المنطقة العليا ({bb_pband:.0f}%)")
            elif bb_pband < 30:
                signals.append(0.4)
                details.append(f"📊 السعر في المنطقة السفلى ({bb_pband:.0f}%)")
            else:
                signals.append(0)
                details.append(f"⚪ السعر في الوسط ({bb_pband:.0f}%)")
            
            # تحليل عرض البولينجر (التقلب)
            if width < 0.01:
                details.append(f"⚠️ البولينجر ضيق جداً (تقلب منخفض - توقع حركة قوية)")
            elif width > 0.05:
                details.append(f"⚠️ البولينجر واسع (تقلب عالي)")
        
        return signals, details

    def analyze_atr(self, df, latest, prev):
        """تحليل ATR"""
        signals = []
        details = []
        
        if 'ATR' in df.columns:
            atr = latest['ATR']
            prev_atr = prev['ATR']
            atr_change = ((atr - prev_atr) / prev_atr * 100) if prev_atr != 0 else 0
            
            # ATR لا يعطي إشارة اتجاه ولكن يعطي قوة التقلب
            if atr_change > 10:
                details.append(f"⚠️ ATR={atr:.5f} (تقلب متزايد +{atr_change:.1f}% - حذر)")
            elif atr_change < -10:
                details.append(f"📊 ATR={atr:.5f} (تقلب متناقص {atr_change:.1f}%)")
            else:
                details.append(f"📊 ATR={atr:.5f} (تقلب مستقر)")
            
            # لا نضيف signal لأن ATR مؤشر تقلب وليس اتجاه
            signals.append(0)
        
        return signals, details
    #--------------------------------------------------
    def generate_signal(self, df, selected_indicators, strategy='default'):
        """توليد إشارة التداول بناءً على الاستراتيجية المختارة"""
        # نتأكد من وجود 200 شمعة على الأقل
        if df is None or len(df) < 200:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'reason': 'بيانات غير كافية',
                'indicators': {}
            }

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        all_signals = []
        all_details = []

        # تحليل كل مؤشر مختار على البيانات الكاملة
        if 'EMA50' in selected_indicators or 'EMA200' in selected_indicators:
            signals, details = self.analyze_ema(df, latest, prev)
            all_signals.extend(signals)
            all_details.extend(details)

        if 'RSI' in selected_indicators:
            signals, details = self.analyze_rsi(df, latest, prev)
            all_signals.extend(signals)
            all_details.extend(details)

        if 'MACD' in selected_indicators:
            signals, details = self.analyze_macd(df, latest, prev)
            all_signals.extend(signals)
            all_details.extend(details)

        if 'BB' in selected_indicators:
            signals, details = self.analyze_bollinger(df, latest, prev)
            all_signals.extend(signals)
            all_details.extend(details)

        if 'ATR' in selected_indicators:
            signals, details = self.analyze_atr(df, latest, prev)
            all_signals.extend(signals)
            all_details.extend(details)

        # تطبيق منطق الاستراتيجية المختارة
        if strategy == 'conservative':
            # استراتيجية محافظة - تحتاج إشارات أقوى
            threshold = 0.6
            confidence_multiplier = 0.8
        elif strategy == 'aggressive':
            # استراتيجية عدوانية - تتفاعل مع إشارات أضعف
            threshold = 0.2
            confidence_multiplier = 1.2
        elif strategy == 'trend_following':
            # متابعة الاتجاه - التركيز على EMA و MACD
            ema_signals = []
            macd_signals = []
            if 'EMA50' in selected_indicators or 'EMA200' in selected_indicators:
                ema_signals, _ = self.analyze_ema(df, latest, prev)
            if 'MACD' in selected_indicators:
                macd_signals, _ = self.analyze_macd(df, latest, prev)

            # المتوسط المرجح للاتجاه
            trend_signals = ema_signals + macd_signals
            if trend_signals:
                avg_signal = np.mean(trend_signals) * 1.2  # تعزيز إشارات الاتجاه
            else:
                avg_signal = np.mean(all_signals) if all_signals else 0

            threshold = 0.4
            confidence_multiplier = 1.0
        elif strategy == 'mean_reversion':
            # العودة للوسط - التركيز على RSI و Bollinger Bands
            reversion_signals = []
            if 'RSI' in selected_indicators:
                rsi_signals, _ = self.analyze_rsi(df, latest, prev)
                reversion_signals.extend(rsi_signals)
            if 'BB' in selected_indicators:
                bb_signals, _ = self.analyze_bollinger(df, latest, prev)
                reversion_signals.extend(bb_signals)

            if reversion_signals:
                avg_signal = np.mean(reversion_signals) * 1.1
            else:
                avg_signal = np.mean(all_signals) if all_signals else 0

            threshold = 0.5
            confidence_multiplier = 1.0
        elif strategy == 'breakout':
            # استراتيجية الاختراق - التركيز على Bollinger Bands و ATR
            breakout_signals = []
            if 'BB' in selected_indicators:
                bb_signals, _ = self.analyze_bollinger(df, latest, prev)
                breakout_signals.extend(bb_signals)
            if 'ATR' in selected_indicators:
                atr_signals, _ = self.analyze_atr(df, latest, prev)
                breakout_signals.extend(atr_signals)

            if breakout_signals:
                avg_signal = np.mean(breakout_signals) * 0.9  # إشارات الاختراق أقل موثوقية
            else:
                avg_signal = np.mean(all_signals) if all_signals else 0

            threshold = 0.45
            confidence_multiplier = 0.9
        else:
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

        # إضافة اسم الاستراتيجية للتفاصيل
        strategy_names = {
            'default': 'الاستراتيجية الافتراضية (متوازنة)',
            'conservative': 'استراتيجية محافظة',
            'aggressive': 'استراتيجية عدوانية',
            'trend_following': 'متابعة الاتجاه',
            'mean_reversion': 'العودة للوسط',
            'breakout': 'استراتيجية الاختراق'
        }

        if all_details:
            all_details.insert(0, f"🎯 {strategy_names.get(strategy, 'استراتيجية غير معروفة')}")

        # جمع قيم المؤشرات من latest
        indicators_values = {}
        for ind in selected_indicators:
            if ind == 'EMA50' and 'EMA50' in df.columns:
                indicators_values['EMA50'] = self.safe_float(latest['EMA50'])
            elif ind == 'EMA200' and 'EMA200' in df.columns:
                indicators_values['EMA200'] = self.safe_float(latest['EMA200'])
            elif ind == 'RSI' and 'RSI' in df.columns:
                indicators_values['RSI'] = self.safe_float(latest['RSI'])
            elif ind == 'MACD' and 'MACD' in df.columns:
                indicators_values['MACD'] = self.safe_float(latest['MACD'])
                indicators_values['MACD_signal'] = self.safe_float(latest['MACD_signal'])
                indicators_values['MACD_diff'] = self.safe_float(latest['MACD_diff'])
            elif ind == 'BB' and 'BB_upper' in df.columns:
                indicators_values['BB_upper'] = self.safe_float(latest['BB_upper'])
                indicators_values['BB_middle'] = self.safe_float(latest['BB_middle'])
                indicators_values['BB_lower'] = self.safe_float(latest['BB_lower'])
                indicators_values['BB_pband'] = self.safe_float(latest['BB_pband'])
            elif ind == 'ATR' and 'ATR' in df.columns:
                indicators_values['ATR'] = self.safe_float(latest['ATR'])

        return {
            'signal': final_signal,
            'signal_text': signal_text,
            'confidence': round(confidence, 1),
            'reason': ' | '.join(all_details[:4]) if all_details else 'لا توجد إشارات',
            'all_details': all_details,
            'indicators': indicators_values,
            'price': {
                'open': self.safe_float(latest['open']),
                'high': self.safe_float(latest['high']),
                'low': self.safe_float(latest['low']),
                'close': self.safe_float(latest['close'])
            },
            'last_candle_time': latest['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy
        }
    def analyze_pair(self, pair, period, selected_indicators, strategy='default'):
        """تحليل زوج واحد"""
        df = self.fetch_data(pair, period)
        if df is None:
            return None

        df = self.calculate_indicators(df, selected_indicators)
        if df is None:
            return None

        analysis = self.generate_signal(df, selected_indicators, strategy)
        # إضافة بيانات الشموع للرسم البياني
        chart_data = []
        for _, row in df.tail(50).iterrows():
            chart_data.append({
                'time': row['datetime'].strftime('%H:%M'),
                'open': self.safe_float(row['open']),
                'high': self.safe_float(row['high']),
                'low': self.safe_float(row['low']),
                'close': self.safe_float(row['close'])
            })
        analysis['chart_data'] = chart_data
        return analysis
    def start_analysis(self, pairs, period, interval_minutes, selected_indicators, strategy='default'):
        """تخزين الإعدادات دون تشغيل تحليل مستمر"""
        self.latest_results = {}  # إعادة تعيين النتائج
        self.is_running = True
        self.analysis_config = {
            'pairs': pairs,
            'period': period,
            'interval': interval_minutes,
            'indicators': selected_indicators,
            'strategy': strategy
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


@app.route('/api/results')
def get_results():
    """تشغيل التحليل عند الطلب"""
    if not analyzer.is_running or not hasattr(analyzer, 'analysis_config'):
        return jsonify({'status': 'error', 'message': 'التحليل غير مفعل حالياً'})

    config = analyzer.analysis_config
    results = {}
    current_time = datetime.now(timezone.utc) + timedelta(hours=3)

    for pair in config['pairs']:
        analysis = analyzer.analyze_pair(pair, config['period'], config['indicators'], config.get('strategy', 'default'))
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

    analyzer.latest_results = {
        'timestamp': datetime.now().isoformat(),
        'analysis': results,
        'selected_indicators': config['indicators'],
        'status': 'success'
    }
    return jsonify(analyzer.latest_results)
if __name__ == '__main__':
    print("🚀 بدء خادم التداول المتقدم...")
    print("📱 افتح المتصفح على: http://localhost:5000")
    print("📊 المؤشرات المتاحة: EMA50, EMA200, RSI, MACD, Bollinger Bands, ATR")
    app.run(debug=True, host='0.0.0.0', port=5000)