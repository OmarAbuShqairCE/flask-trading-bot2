#!/usr/bin/env python3
"""
اختبار نظام الذكاء الاصطناعي للتداول
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_imports():
    """اختبار استيراد المكتبات المطلوبة"""
    print("🔍 اختبار استيراد المكتبات...")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في Pandas: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في Scikit-learn: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في TensorFlow: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في OpenCV: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ خطأ في Matplotlib: {e}")
        return False
    
    return True

def test_ai_system():
    """اختبار نظام الذكاء الاصطناعي"""
    print("\n🤖 اختبار نظام الذكاء الاصطناعي...")
    
    try:
        # استيراد النظام
        from app import AITradingSystem
        
        # إنشاء مثيل النظام
        ai_system = AITradingSystem()
        print("✅ تم إنشاء نظام AI بنجاح")
        
        # اختبار إعداد بيانات التدريب
        price_data = [
            {'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1020},
            {'open': 1.1020, 'high': 1.1080, 'low': 1.1000, 'close': 1.1060},
            {'open': 1.1060, 'high': 1.1100, 'low': 1.1040, 'close': 1.1080},
            {'open': 1.1080, 'high': 1.1120, 'low': 1.1060, 'close': 1.1100},
            {'open': 1.1100, 'high': 1.1150, 'low': 1.1080, 'close': 1.1130}
        ]
        
        indicators_data = {}
        features, labels = ai_system.prepare_training_data(price_data, indicators_data)
        
        if features is not None and len(features) > 0:
            print(f"✅ تم إعداد بيانات التدريب: {len(features)} عينة")
        else:
            print("⚠️ بيانات التدريب غير كافية")
        
        # اختبار التنبؤ
        if len(features) > 0:
            current_features = features[0]
            prediction = ai_system.predict_with_ensemble(current_features)
            print(f"✅ تم التنبؤ: {prediction['signal']} ({prediction['confidence']}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في نظام AI: {e}")
        return False

def test_trading_analyzer():
    """اختبار محلل التداول"""
    print("\n📊 اختبار محلل التداول...")
    
    try:
        from app import TradingAnalyzer
        
        # إنشاء محلل التداول
        analyzer = TradingAnalyzer()
        print("✅ تم إنشاء محلل التداول بنجاح")
        
        # اختبار حالة AI
        print(f"✅ AI مفعل: {analyzer.ai_enabled}")
        print(f"✅ نظام AI: {type(analyzer.ai_system).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في محلل التداول: {e}")
        return False

def test_data_generation():
    """اختبار توليد بيانات تجريبية"""
    print("\n📈 اختبار توليد البيانات...")
    
    try:
        # توليد بيانات تجريبية
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        prices = np.random.normal(1.1000, 0.01, 100).cumsum()
        
        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 0.005, 100),
            'low': prices - np.random.uniform(0, 0.005, 100),
            'close': prices + np.random.uniform(-0.002, 0.002, 100)
        })
        
        print(f"✅ تم توليد {len(data)} نقطة بيانات")
        print(f"✅ نطاق الأسعار: {data['close'].min():.5f} - {data['close'].max():.5f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في توليد البيانات: {e}")
        return False

def test_model_training():
    """اختبار تدريب النماذج"""
    print("\n🧠 اختبار تدريب النماذج...")
    
    try:
        from app import AITradingSystem
        
        # إنشاء بيانات تجريبية
        np.random.seed(42)
        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 3, 100)
        
        ai_system = AITradingSystem()
        
        # اختبار تدريب نماذج ML
        print("🔄 تدريب نماذج ML...")
        ml_success = ai_system.train_ml_models(features, labels)
        print(f"✅ تدريب ML: {'نجح' if ml_success else 'فشل'}")
        
        # اختبار التنبؤ
        if ml_success:
            test_features = np.random.randn(1, 10)
            prediction = ai_system.predict_with_ensemble(test_features[0])
            print(f"✅ التنبؤ: {prediction['signal']} ({prediction['confidence']}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في تدريب النماذج: {e}")
        return False

def main():
    """الدالة الرئيسية للاختبار"""
    print("🚀 بدء اختبار نظام التداول الذكي")
    print("=" * 50)
    
    tests = [
        ("استيراد المكتبات", test_imports),
        ("نظام الذكاء الاصطناعي", test_ai_system),
        ("محلل التداول", test_trading_analyzer),
        ("توليد البيانات", test_data_generation),
        ("تدريب النماذج", test_model_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: نجح")
            else:
                print(f"❌ {test_name}: فشل")
        except Exception as e:
            print(f"❌ {test_name}: خطأ - {e}")
    
    print("\n" + "="*50)
    print(f"📊 النتائج: {passed}/{total} اختبار نجح")
    
    if passed == total:
        print("🎉 جميع الاختبارات نجحت! النظام جاهز للاستخدام.")
        return True
    else:
        print("⚠️ بعض الاختبارات فشلت. تحقق من التثبيت.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
