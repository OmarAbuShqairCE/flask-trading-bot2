#!/usr/bin/env python3
"""
تشغيل نظام التداول الذكي
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """التحقق من المتطلبات"""
    print("🔍 التحقق من المتطلبات...")
    
    # التحقق من Python
    if sys.version_info < (3, 8):
        print("❌ يتطلب Python 3.8 أو أحدث")
        return False
    
    print(f"✅ Python: {sys.version}")
    
    # التحقق من الملفات المطلوبة
    required_files = ['app.py', 'templates/index.html', 'requirements.txt']
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ ملف مفقود: {file}")
            return False
        print(f"✅ {file}")
    
    return True

def install_requirements():
    """تثبيت المتطلبات"""
    print("\n📦 تثبيت المتطلبات...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ تم تثبيت المتطلبات بنجاح")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تثبيت المتطلبات: {e}")
        return False

def run_tests():
    """تشغيل الاختبارات"""
    print("\n🧪 تشغيل الاختبارات...")
    
    try:
        result = subprocess.run([sys.executable, 'test_ai_system.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ جميع الاختبارات نجحت")
            return True
        else:
            print("❌ بعض الاختبارات فشلت")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ خطأ في تشغيل الاختبارات: {e}")
        return False

def start_server():
    """تشغيل الخادم"""
    print("\n🚀 تشغيل خادم التداول الذكي...")
    
    try:
        # تشغيل التطبيق
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف الخادم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل الخادم: {e}")

def main():
    """الدالة الرئيسية"""
    print("🚀 نظام التداول الذكي المتقدم")
    print("=" * 40)
    
    # التحقق من المتطلبات
    if not check_requirements():
        print("❌ فشل في التحقق من المتطلبات")
        return False
    
    # تثبيت المتطلبات
    if not install_requirements():
        print("❌ فشل في تثبيت المتطلبات")
        return False
    
    # تشغيل الاختبارات
    if not run_tests():
        print("⚠️ تحذير: بعض الاختبارات فشلت، لكن يمكن المتابعة")
    
    # تشغيل الخادم
    print("\n🎯 بدء تشغيل النظام...")
    print("📱 افتح المتصفح على: http://localhost:5000")
    print("🤖 نظام الذكاء الاصطناعي جاهز!")
    print("=" * 40)
    
    start_server()
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف النظام")
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        sys.exit(1)
