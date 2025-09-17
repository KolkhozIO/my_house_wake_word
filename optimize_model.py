#!/usr/bin/env python3
"""
Оптимизация модели для уменьшения размера
"""

import tensorflow as tf
import os

def optimize_model():
    """Оптимизирует существующую модель для уменьшения размера"""
    
    # Загружаем существующую модель
    model_path = "models/trained_model/wake_word_model.h5"
    if not os.path.exists(model_path):
        print("❌ Модель не найдена!")
        return
    
    print("📦 Загружаем модель...")
    model = tf.keras.models.load_model(model_path)
    
    # Создаем конвертер с квантизацией
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Включаем квантизацию для уменьшения размера
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("🔧 Конвертируем с квантизацией...")
    try:
        tflite_model = converter.convert()
        
        # Сохраняем оптимизированную модель
        optimized_path = "models/trained_model/wake_word_model_optimized.tflite"
        with open(optimized_path, 'wb') as f:
            f.write(tflite_model)
        
        # Проверяем размер
        original_size = os.path.getsize("models/trained_model/wake_word_model.tflite")
        optimized_size = os.path.getsize(optimized_path)
        
        print(f"✅ Оптимизированная модель сохранена: {optimized_path}")
        print(f"📊 Размер оригинальной модели: {original_size / 1024:.1f} KB")
        print(f"📊 Размер оптимизированной модели: {optimized_size / 1024:.1f} KB")
        print(f"📊 Уменьшение: {((original_size - optimized_size) / original_size * 100):.1f}%")
        
        # Заменяем оригинальную модель
        os.replace(optimized_path, "models/trained_model/wake_word_model.tflite")
        print("🔄 Заменена оригинальная модель")
        
    except Exception as e:
        print(f"❌ Ошибка при оптимизации: {e}")

if __name__ == "__main__":
    optimize_model()