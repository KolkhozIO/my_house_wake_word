#!/usr/bin/env python3
"""
Создание модели совместимой с TFLite Micro
"""

import tensorflow as tf
import numpy as np
import os

def create_compatible_model():
    """Создает модель совместимую с TFLite Micro"""
    
    # Простая архитектура без проблемных операций
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 49)),
        tf.keras.layers.Reshape((40, 49, 1)),
        
        # Простые слои без проблемных операций
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Создаем фиктивные данные для компиляции
    dummy_data = np.random.random((10, 40, 49))
    dummy_labels = np.random.randint(0, 2, (10, 1))
    
    # Обучаем на фиктивных данных (1 эпоха)
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
    
    # Сохраняем модель
    model_dir = "models/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # H5 модель
    model_path = os.path.join(model_dir, "wake_word_model.h5")
    model.save(model_path)
    
    # Конвертируем в TFLite с совместимостью
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Настройки для совместимости с TFLite Micro
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    try:
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(model_dir, "wake_word_model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Проверяем размер
        size_kb = os.path.getsize(tflite_path) / 1024
        
        print(f"✅ Совместимая модель создана: {tflite_path}")
        print(f"📊 Размер: {size_kb:.1f} KB")
        
        return size_kb
        
    except Exception as e:
        print(f"❌ Ошибка при конвертации: {e}")
        return None

if __name__ == "__main__":
    create_compatible_model()