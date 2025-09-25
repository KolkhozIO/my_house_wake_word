#!/usr/bin/env python3
"""
Простая конвертация SavedModel в TFLite для microWakeWord
"""

import sys
import os
sys.path.append('/home/microWakeWord')

import tensorflow as tf
from pathlib import Path

def convert_savedmodel_to_tflite():
    """Конвертирует SavedModel в TFLite"""
    
    # Пути
    savedmodel_dir = "/home/microWakeWord_data/trained_models/wakeword/non_stream"
    output_dir = "/home/microWakeWord_data/trained_models/wakeword"
    
    print("🔄 Конвертация SavedModel в TFLite...")
    
    try:
        # Загружаем SavedModel
        print(f"📁 Загрузка SavedModel из: {savedmodel_dir}")
        model = tf.saved_model.load(savedmodel_dir)
        
        # Получаем функцию инференса
        infer_func = model.signatures['serving_default']
        
        # Создаем конвертер
        converter = tf.lite.TFLiteConverter.from_concrete_functions([infer_func])
        
        # Настройки конвертера
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
        
        # Конвертируем
        print("🔄 Конвертация...")
        tflite_model = converter.convert()
        
        # Сохраняем
        tflite_path = os.path.join(output_dir, "model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite модель сохранена: {tflite_path}")
        print(f"📊 Размер модели: {len(tflite_model) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return False

if __name__ == "__main__":
    success = convert_savedmodel_to_tflite()
    if success:
        print("🎉 Конвертация завершена успешно!")
    else:
        print("❌ Конвертация завершилась с ошибками")
        sys.exit(1)