#!/usr/bin/env python3
"""
Обучение модели на реальных данных
"""

import tensorflow as tf
import numpy as np
import os
import librosa
from pathlib import Path

def load_audio_file(file_path):
    """Загружает аудио файл"""
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        return audio
    except Exception as e:
        print(f"Ошибка загрузки {file_path}: {e}")
        return None

def create_mel_spectrogram(audio):
    """Создает mel-спектрограмму"""
    try:
        # Нормализация длины до 1.5 секунд (24000 сэмплов при 16kHz)
        target_length = 24000
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Создание mel-спектрограммы с правильными параметрами
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=40,
            hop_length=160,
            n_fft=512
        )
        
        # Логарифмическое масштабирование
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Обрезаем до нужного размера (40 x 49)
        if mel_spec.shape[1] > 49:
            mel_spec = mel_spec[:, :49]
        elif mel_spec.shape[1] < 49:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, 49 - mel_spec.shape[1])))
        
        return mel_spec
    except Exception as e:
        print(f"Ошибка создания спектрограммы: {e}")
        return None

def load_real_data():
    """Загружает реальные данные"""
    features = []
    labels = []
    
    # Положительные сэмплы
    pos_dir = Path("piper-sample-generator/positives_combined_aug")
    if pos_dir.exists():
        pos_files = list(pos_dir.glob("*.wav"))
        print(f"Найдено {len(pos_files)} положительных файлов")
        
        for i, file_path in enumerate(pos_files[:100]):  # Берем первые 100
            if i % 10 == 0:
                print(f"Обрабатываю положительные: {i}/{min(100, len(pos_files))}")
            
            audio = load_audio_file(file_path)
            if audio is not None:
                mel_spec = create_mel_spectrogram(audio)
                if mel_spec is not None:
                    features.append(mel_spec)
                    labels.append(1)  # положительный класс
    
    # Негативные сэмплы
    neg_dir = Path("piper-sample-generator/negatives_moy_dom_massive_aug")
    if neg_dir.exists():
        neg_files = list(neg_dir.glob("*.wav"))
        print(f"Найдено {len(neg_files)} негативных файлов")
        
        for i, file_path in enumerate(neg_files[:50]):  # Берем первые 50
            if i % 10 == 0:
                print(f"Обрабатываю негативные: {i}/{min(50, len(neg_files))}")
            
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
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(40, 49)),
        tf.keras.layers.Reshape((40, 49, 1)),
        
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
    
    return model

def main():
    """Основная функция"""
    print("📦 Загружаю реальные данные...")
    features, labels = load_real_data()
    
    if len(features) == 0:
        print("❌ Нет данных для обучения!")
        return
    
    print("🏗️ Создаю модель...")
    model = create_model()
    
    print("🎯 Обучаю модель...")
    model.fit(features, labels, epochs=10, validation_split=0.2, verbose=1)
    
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
    
    print(f"✅ Модель обучена на реальных данных!")
    print(f"📊 Размер: {size_kb:.1f} KB")
    print(f"📁 Сохранена: {tflite_path}")

if __name__ == "__main__":
    main()