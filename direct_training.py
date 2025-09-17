#!/usr/bin/env python3
"""
Прямое обучение модели wake-word без microWakeWord
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import random

def load_audio_file(file_path, target_sr=16000):
    """Загружает аудиофайл"""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return None

def create_mel_spectrogram(audio, sr=16000, n_mels=40, hop_length=160):
    """Создает мел-спектрограмму"""
    try:
        # Создаем мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            hop_length=hop_length,
            n_fft=1024
        )
        
        # Преобразуем в логарифмический масштаб
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Обрезаем или дополняем до нужного размера (49 временных шагов)
        if mel_spec_db.shape[1] >= 49:
            mel_spec_db = mel_spec_db[:, :49]
        else:
            # Дополняем нулями
            padding = np.zeros((n_mels, 49 - mel_spec_db.shape[1]))
            mel_spec_db = np.hstack([mel_spec_db, padding])
        
        return mel_spec_db
    except Exception as e:
        print(f"Ошибка создания спектрограммы: {e}")
        return None

def load_dataset(positive_dir, negative_dir, max_samples=100):
    """Загружает датасет"""
    
    print("📊 Загрузка датасета...")
    
    # Получаем списки файлов
    pos_files = list(Path(positive_dir).glob('*.wav'))[:max_samples]
    neg_files = list(Path(negative_dir).glob('*.wav'))[:max_samples]
    
    print(f"Положительных файлов: {len(pos_files)}")
    print(f"Негативных файлов: {len(neg_files)}")
    
    features = []
    labels = []
    
    # Обрабатываем положительные файлы
    print("Обработка положительных файлов...")
    for i, file_path in enumerate(pos_files):
        if i % 20 == 0:
            print(f"  Обработано {i}/{len(pos_files)}")
        
        audio = load_audio_file(file_path)
        if audio is not None:
            mel_spec = create_mel_spectrogram(audio)
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(1)  # положительный класс
    
    # Обрабатываем негативные файлы
    print("Обработка негативных файлов...")
    for i, file_path in enumerate(neg_files):
        if i % 10 == 0:
            print(f"  Обработано {i}/{len(neg_files)}")
        
        audio = load_audio_file(file_path)
        if audio is not None:
            mel_spec = create_mel_spectrogram(audio)
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(0)  # негативный класс
    
    print(f"✅ Загружено {len(features)} спектрограмм")
    print(f"   Положительных: {sum(labels)}")
    print(f"   Негативных: {len(labels) - sum(labels)}")
    
    return np.array(features), np.array(labels)

def create_model():
    """Создает модель"""
    
    print("🏗️ Создание модели...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 49)),  # n_mels x time_steps
        tf.keras.layers.Reshape((40, 49, 1)),
        
        # Сверточные слои
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        
        # Полносвязные слои
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("✅ Модель создана!")
    model.summary()
    
    return model

def train_model(model, X, y):
    """Обучает модель"""
    
    print("🚀 Начинаем обучение...")
    
    # Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📊 Размер обучающей выборки: {X_train.shape}")
    print(f"📊 Размер тестовой выборки: {X_test.shape}")
    
    # Обучаем модель
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ]
    )
    
    # Оцениваем модель
    print("\n📊 Оценка модели:")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Точность: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    
    return model, history

def save_model(model):
    """Сохраняет модель"""
    
    print("💾 Сохранение модели...")
    
    model_dir = "/root/microWakeWord/models/trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Сохраняем модель
    model_path = os.path.join(model_dir, "wake_word_model.h5")
    model.save(model_path)
    
    # Конвертируем в TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(model_dir, "wake_word_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Модель сохранена: {model_path}")
    print(f"✅ TFLite модель сохранена: {tflite_path}")
    
    return model_dir

def main():
    """Основная функция"""
    
    print("🎯 ПРЯМОЕ ОБУЧЕНИЕ МОДЕЛИ WAKE-WORD")
    print("   для фраз 'мой дом' и 'любимый дом'")
    print("=" * 60)
    
    try:
        # Загружаем данные
        positive_dir = "/root/microWakeWord/piper-sample-generator/positives_combined_aug"
        negative_dir = "/root/microWakeWord/piper-sample-generator/negatives_moy_dom_massive_aug"
        
        X, y = load_dataset(positive_dir, negative_dir, max_samples=200)
        
        if len(X) == 0:
            print("❌ Нет данных для обучения!")
            return
        
        # Создаем модель
        model = create_model()
        
        # Обучаем модель
        model, history = train_model(model, X, y)
        
        # Сохраняем модель
        model_dir = save_model(model)
        
        print("\n" + "=" * 60)
        print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)
        
        print(f"\n📁 Модель сохранена в: {model_dir}")
        print(f"📊 Обучено на {len(X)} сэмплах")
        print(f"🎯 Поддерживает фразы: 'мой дом' и 'любимый дом'")
        print(f"🚀 Готово к развертыванию!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()