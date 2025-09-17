#!/usr/bin/env python3
"""
Создание маленькой модели для ESP32
"""

import tensorflow as tf
import numpy as np
import os

def create_small_model():
    """Создает маленькую модель"""
    
    # Простая архитектура
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 49)),
        tf.keras.layers.Reshape((40, 49, 1)),
        
        # Меньше слоев и фильтров
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        
        # Меньше нейронов
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Создаем фиктивные данные для компиляции
    dummy_data = np.random.random((1, 40, 49))
    dummy_labels = np.array([1])
    
    # Обучаем на фиктивных данных (1 эпоха)
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
    
    # Сохраняем модель
    model_dir = "models/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # H5 модель
    model_path = os.path.join(model_dir, "wake_word_model.h5")
    model.save(model_path)
    
    # Конвертируем в TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(model_dir, "wake_word_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Проверяем размер
    size_kb = os.path.getsize(tflite_path) / 1024
    
    print(f"✅ Маленькая модель создана: {tflite_path}")
    print(f"📊 Размер: {size_kb:.1f} KB")
    
    return size_kb

if __name__ == "__main__":
    create_small_model()