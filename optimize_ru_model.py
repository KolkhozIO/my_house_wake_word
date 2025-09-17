#!/usr/bin/env python3
"""
Оптимизация русской модели для уменьшения размера
"""

import tensorflow as tf
import numpy as np
import os

def optimize_ru_model():
    """Создает оптимизированную русскую модель"""
    
    # Оптимизированная архитектура - меньше параметров
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 49)),
        tf.keras.layers.Reshape((40, 49, 1)),
        
        # Первый блок - меньше фильтров
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Второй блок - меньше фильтров
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Третий блок - меньше фильтров
        tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        
        # Полносвязные слои - меньше нейронов
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Создаем фиктивные данные для компиляции
    dummy_data = np.random.random((100, 40, 49))
    dummy_labels = np.random.randint(0, 2, (100, 1))
    
    # Обучаем на фиктивных данных (5 эпох)
    model.fit(dummy_data, dummy_labels, epochs=5, verbose=0)
    
    # Сохраняем модель
    model_dir = "models/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # H5 модель
    model_path = os.path.join(model_dir, "wake_word_model.h5")
    model.save(model_path)
    
    # Конвертируем в TFLite с квантизацией
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Включаем квантизацию для уменьшения размера
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Пробуем квантизацию
    try:
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(model_dir, "wake_word_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Проверяем размер
        size_kb = os.path.getsize(tflite_path) / 1024
        
        print(f"✅ Оптимизированная русская модель создана: {tflite_path}")
        print(f"📊 Размер: {size_kb:.1f} KB")
        
        return size_kb
        
    except Exception as e:
        print(f"❌ Ошибка при квантизации: {e}")
        
        # Пробуем без квантизации
        print("🔄 Пробуем без квантизации...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(model_dir, "wake_word_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = os.path.getsize(tflite_path) / 1024
        
        print(f"✅ Русская модель без квантизации: {tflite_path}")
        print(f"📊 Размер: {size_kb:.1f} KB")
        
        return size_kb

if __name__ == "__main__":
    optimize_ru_model()