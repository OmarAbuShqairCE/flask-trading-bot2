# 🛠️ دليل تثبيت نظام التداول الذكي

## متطلبات النظام

### الحد الأدنى

- **Python**: 3.8 أو أحدث
- **RAM**: 4GB على الأقل
- **Storage**: 2GB مساحة فارغة
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### الموصى به

- **Python**: 3.9-3.11
- **RAM**: 8GB أو أكثر
- **Storage**: 5GB مساحة فارغة
- **GPU**: NVIDIA GPU مع CUDA (اختياري)

## خطوات التثبيت

### 1. تثبيت Python

```bash
# Windows
# قم بتحميل Python من python.org

# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-pip
```

### 2. إنشاء بيئة افتراضية

```bash
# إنشاء بيئة افتراضية
python -m venv trading_ai_env

# تفعيل البيئة الافتراضية
# Windows
trading_ai_env\Scripts\activate

# macOS/Linux
source trading_ai_env/bin/activate
```

### 3. تثبيت المكتبات الأساسية

```bash
# تثبيت المكتبات الأساسية أولاً
pip install numpy pandas matplotlib scikit-learn

# تثبيت TensorFlow
pip install tensorflow==2.15.0

# تثبيت المكتبات الأخرى
pip install opencv-python seaborn scipy joblib

# تثبيت باقي المتطلبات
pip install -r requirements.txt
```

### 4. إعداد متغيرات البيئة

```bash
# إنشاء ملف .env
echo "TWELVEDATA_API_KEY=your_api_key_here" > .env
```

### 5. تشغيل النظام

```bash
python app.py
```

## حل المشاكل الشائعة

### مشكلة: خطأ في تثبيت TensorFlow

```bash
# حل بديل
pip install tensorflow-cpu==2.15.0
```

### مشكلة: خطأ في OpenCV

```bash
# تثبيت OpenCV
pip install opencv-python-headless
```

### مشكلة: خطأ في الذاكرة

```bash
# تقليل حجم النماذج في app.py
# تغيير معاملات النماذج
```

### مشكلة: بطء التدريب

```bash
# استخدام GPU إذا متوفر
pip install tensorflow-gpu==2.15.0
```

## التحقق من التثبيت

### اختبار المكتبات

```python
# إنشاء ملف test_install.py
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import sklearn
import matplotlib.pyplot as plt

print("✅ جميع المكتبات مثبتة بنجاح!")
print(f"TensorFlow: {tf.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

### اختبار النظام

```bash
# تشغيل الاختبار
python test_install.py
```

## تحسين الأداء

### استخدام GPU

```bash
# تثبيت CUDA Toolkit
# تثبيت cuDNN
# تثبيت TensorFlow GPU
pip install tensorflow-gpu==2.15.0
```

### تحسين الذاكرة

```python
# في app.py، تقليل معاملات النماذج
model_config = {
    'random_forest': {'n_estimators': 50, 'max_depth': 5},
    'lstm': {'units': 25, 'epochs': 20},
    'cnn': {'filters': 16, 'epochs': 15}
}
```

## استكشاف الأخطاء

### خطأ: "ModuleNotFoundError"

```bash
# تأكد من تفعيل البيئة الافتراضية
# أعد تثبيت المكتبة المفقودة
pip install package_name
```

### خطأ: "CUDA out of memory"

```bash
# تقليل حجم البيانات
# تقليل معاملات النماذج
# استخدام CPU بدلاً من GPU
```

### خطأ: "API key not found"

```bash
# تأكد من وجود ملف .env
# تحقق من صحة مفتاح API
```

## نصائح للاستخدام الأمثل

### 1. تدريب النماذج

- ابدأ ببيانات قليلة (100-200 نقطة)
- زد البيانات تدريجياً
- راقب استخدام الذاكرة

### 2. تحسين الأداء

- استخدم SSD للقرص الصلب
- زد ذاكرة النظام
- أغلق البرامج غير الضرورية

### 3. مراقبة النظام

- راقب استخدام CPU والذاكرة
- تحقق من سجلات الأخطاء
- احفظ النماذج بانتظام

## الدعم الفني

### سجلات النظام

```bash
# عرض سجلات التطبيق
tail -f app.log

# عرض أخطاء Python
python app.py 2>&1 | tee error.log
```

### معلومات النظام

```python
import platform
import psutil

print(f"OS: {platform.system()}")
print(f"Python: {platform.python_version()}")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"CPU: {psutil.cpu_count()} cores")
```

---

**ملاحظة**: إذا واجهت مشاكل، تأكد من تثبيت جميع المتطلبات بالترتيب الصحيح.
