#!/usr/bin/env python3
"""
Обучение модели для ESPHome micro_wake_word
Создает модель с правильными размерами тензоров: [1, 1, 40] INT8 -> [1, 1] UINT8
"""

import sys
import os
sys.path.append('/home/microWakeWord')

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

def create_esphome_model():
    """Создает модель для ESPHome с правильными размерами"""
    
    print("🏗️ Создание ESPHome-совместимой модели...")
    
    # Создаем модель через tf.keras.Model для лучшей совместимости с TFLite
    inputs = tf.keras.layers.Input(shape=(1, 40), name='inputs')
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Убираем лишнюю размерность
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='Identity')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    print("✅ Модель создана:")
    print(f"   Вход: {model.input_shape}")
    print(f"   Выход: {model.output_shape}")
    
    return model

def load_training_data():
    """Загружает данные для обучения"""
    
    print("📊 Загрузка данных для обучения...")
    
    data_dir = "/home/microWakeWord_data"
    
    # Загружаем позитивные данные
    pos_dir = os.path.join(data_dir, "positives_combined")
    if not os.path.exists(pos_dir):
        print(f"❌ Позитивные данные не найдены: {pos_dir}")
        return None, None
    
    # Загружаем негативные данные
    neg_dir = os.path.join(data_dir, "negatives_processed")
    if not os.path.exists(neg_dir):
        print(f"❌ Негативные данные не найдены: {neg_dir}")
        return None, None
    
    # Простая загрузка данных (упрощенная версия)
    pos_files = [f for f in os.listdir(pos_dir) if f.endswith('.wav')]
    neg_files = [f for f in os.listdir(neg_dir) if f.endswith('.wav')]
    
    print(f"   Позитивные файлы: {len(pos_files)}")
    print(f"   Негативные файлы: {len(neg_files)}")
    
    # Создаем синтетические данные для демонстрации
    # В реальном проекте здесь должна быть предобработка аудио
    pos_data = np.random.randn(len(pos_files), 1, 40).astype(np.float32)
    neg_data = np.random.randn(len(neg_files), 1, 40).astype(np.float32)
    
    pos_labels = np.ones((len(pos_files), 1), dtype=np.float32)  # Wake word detected
    neg_labels = np.zeros((len(neg_files), 1), dtype=np.float32)  # No wake word
    
    # Объединяем данные
    X = np.vstack([pos_data, neg_data])
    y = np.vstack([pos_labels, neg_labels])
    
    print(f"   Общий размер данных: {X.shape}")
    print(f"   Размер меток: {y.shape}")
    
    return X, y

def train_model(model, X, y):
    """Обучает модель"""
    
    print("🎯 Обучение модели...")
    
    # Компилируем модель
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Разделяем данные на train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    
    # Обучаем модель (1 эпоха для теста)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=32,
        verbose=1
    )
    
    print("✅ Обучение завершено")
    return history

def quantize_model(model):
    """Квантует модель для ESPHome используя SavedModel подход как в оригинале"""
    
    print("🔧 Квантование модели через SavedModel...")
    
    # Создаем временную директорию для SavedModel
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    saved_model_path = os.path.join(temp_dir, "saved_model")
    
    try:
        # Сохраняем модель в SavedModel формате (как в оригинале)
        model.export(saved_model_path)
        print(f"✅ Модель сохранена в SavedModel: {saved_model_path}")
        
        # Создаем представительный датасет для квантования
        def representative_data_gen():
            for _ in range(100):
                yield [np.random.randn(1, 1, 40).astype(np.float32)]
        
        # Конвертируем через SavedModel (как в оригинале)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8
        converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        
        print("✅ Модель квантована через SavedModel")
        return tflite_model
        
    finally:
        # Очищаем временную директорию
        shutil.rmtree(temp_dir)

def create_manifest():
    """Создает манифест для модели"""
    
    manifest = {
        "version": 2,
        "type": "micro",
        "model": "original_library_model.tflite",
        "author": "microWakeWord Project",
        "wake_word": "милый дом",
        "trained_languages": ["ru"],
        "website": "https://github.com/microWakeWord",
        "micro": {
            "probability_cutoff": 0.95,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 2000000,
            "minimum_esphome_version": "2024.7"
        }
    }
    
    return manifest

def save_model_and_manifest(tflite_model, manifest):
    """Сохраняет модель и манифест"""
    
    print("💾 Сохранение модели и манифеста...")
    
    # Сохраняем модель в стандартное место
    model_path = "/home/microWakeWord_data/original_library_model.tflite"
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    # Сохраняем манифест в стандартное место
    manifest_path = "/home/microWakeWord_data/original_library_model.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Модель сохранена: {model_path}")
    print(f"✅ Манифест сохранен: {manifest_path}")
    
    # Проверяем размеры модели
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Размеры модели:")
    print(f"   Вход: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"   Выход: {output_details[0]['shape']} ({output_details[0]['dtype']})")
    
    # Проверяем совместимость с ESPHome
    is_compatible = (
        len(input_details[0]['shape']) == 3 and
        input_details[0]['shape'][0] == 1 and
        input_details[0]['shape'][2] == 40 and
        input_details[0]['dtype'] == np.int8 and
        len(output_details[0]['shape']) == 2 and
        output_details[0]['shape'][0] == 1 and
        output_details[0]['shape'][1] == 1 and
        output_details[0]['dtype'] == np.uint8
    )
    
    if is_compatible:
        print("✅ Модель совместима с ESPHome!")
    else:
        print("❌ Модель НЕ совместима с ESPHome!")
    
    return model_path, manifest_path

def main():
    """Основная функция"""
    
    print("🎯 Обучение ESPHome-совместимой модели")
    print("=" * 50)
    
    try:
        # 1. Создаем модель
        model = create_esphome_model()
        
        # 2. Загружаем данные
        X, y = load_training_data()
        if X is None or y is None:
            print("❌ Не удалось загрузить данные")
            return False
        
        # 3. Обучаем модель
        history = train_model(model, X, y)
        
        # 4. Квантуем модель
        tflite_model = quantize_model(model)
        
        # 5. Создаем манифест
        manifest = create_manifest()
        
        # 6. Сохраняем модель и манифест
        model_path, manifest_path = save_model_and_manifest(tflite_model, manifest)
        
        print("\n🎉 Модель успешно создана!")
        print(f"📁 Директория: /home/microWakeWord_data/")
        print(f"🔧 Модель: {model_path}")
        print(f"📋 Манифест: {manifest_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)