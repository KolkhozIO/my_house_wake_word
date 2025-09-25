#!/usr/bin/env python3
"""
Тестирование обученной модели на примерах данных
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import glob

# Добавляем путь к библиотеке microWakeWord
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths

def load_model():
    """Загружает обученную модель"""
    model_path = "/home/microWakeWord_data/trained_models/wakeword_mixed_консервативный"
    
    # Загружаем конфигурацию
    config_path = os.path.join(model_path, "training_config.yaml")
    print(f"📋 Конфигурация: {config_path}")
    
    # Загружаем веса
    weights_path = os.path.join(model_path, "best_weights.weights.h5")
    print(f"⚖️ Веса модели: {weights_path}")
    
    # Создаем упрощенную модель для тестирования
    # Используем правильный размер входа на основе summary: (194, 40) -> 7760
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(194, 40)),
        tf.keras.layers.Flatten(),  # 194 * 40 = 7760
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Загружаем веса
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("✅ Модель загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки весов: {e}")
            print("🔄 Создаем модель без весов для тестирования")
            # Создаем модель без весов для демонстрации
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print("❌ Веса модели не найдены")
        return None
    
    return model

def preprocess_audio(file_path, target_sr=16000, duration_ms=1500):
    """Предобработка аудио файла"""
    try:
        # Загружаем аудио
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Обрезаем до нужной длительности
        target_samples = int(duration_ms * target_sr / 1000)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            # Дополняем нулями если короткий
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Создаем простую спектрограмму (упрощенная версия)
        # В реальной модели используется более сложная обработка
        stft = librosa.stft(audio, n_fft=512, hop_length=160)
        magnitude = np.abs(stft)
        
        # Обрезаем до нужного размера (194, 40)
        if magnitude.shape[0] > 40:
            magnitude = magnitude[:40, :]
        if magnitude.shape[1] > 194:
            magnitude = magnitude[:, :194]
        else:
            # Дополняем нулями
            pad_width = 194 - magnitude.shape[1]
            magnitude = np.pad(magnitude, ((0, 0), (0, pad_width)))
        
        return magnitude.T  # Транспонируем для соответствия (194, 40)
    
    except Exception as e:
        print(f"❌ Ошибка обработки {file_path}: {e}")
        return None

def test_model_on_files(model, file_paths, expected_label, label_name):
    """Тестирует модель на списке файлов"""
    print(f"\n🎯 Тестирование {label_name} данных:")
    print(f"📁 Файлов для тестирования: {len(file_paths)}")
    
    results = []
    
    for i, file_path in enumerate(file_paths[:10]):  # Тестируем первые 10 файлов
        print(f"  📄 {i+1}/10: {os.path.basename(file_path)}")
        
        # Предобрабатываем аудио
        spectrogram = preprocess_audio(file_path)
        if spectrogram is None:
            continue
        
        # Подготавливаем данные для модели
        input_data = np.expand_dims(spectrogram, axis=0)  # Добавляем batch dimension
        
        # Предсказание
        try:
            prediction = model.predict(input_data, verbose=0)
            confidence = float(prediction[0][0])
            
            # Определяем предсказанный класс
            predicted_class = 1 if confidence > 0.5 else 0
            
            # Проверяем правильность
            is_correct = predicted_class == expected_label
            
            results.append({
                'file': os.path.basename(file_path),
                'confidence': confidence,
                'predicted': predicted_class,
                'expected': expected_label,
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"    {status} Confidence: {confidence:.3f}, Predicted: {predicted_class}, Expected: {expected_label}")
            
        except Exception as e:
            print(f"    ❌ Ошибка предсказания: {e}")
    
    return results

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ ДАННЫХ")
    print("=" * 50)
    
    # Загружаем модель
    model = load_model()
    if model is None:
        print("❌ Не удалось загрузить модель")
        return
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    print(f"📁 Позитивные данные: {positives_dir}")
    print(f"📁 Негативные данные: {negatives_dir}")
    
    # Получаем списки файлов
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:10]
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:10]
    
    print(f"📊 Найдено позитивных файлов: {len(positive_files)}")
    print(f"📊 Найдено негативных файлов: {len(negative_files)}")
    
    # Тестируем на позитивных данных
    positive_results = test_model_on_files(model, positive_files, 1, "позитивных")
    
    # Тестируем на негативных данных
    negative_results = test_model_on_files(model, negative_files, 0, "негативных")
    
    # Анализируем результаты
    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 50)
    
    # Позитивные результаты
    if positive_results:
        correct_positives = sum(1 for r in positive_results if r['correct'])
        total_positives = len(positive_results)
        avg_confidence_positives = np.mean([r['confidence'] for r in positive_results])
        
        print(f"✅ Позитивные данные:")
        print(f"   Правильно: {correct_positives}/{total_positives} ({correct_positives/total_positives*100:.1f}%)")
        print(f"   Средняя уверенность: {avg_confidence_positives:.3f}")
    
    # Негативные результаты
    if negative_results:
        correct_negatives = sum(1 for r in negative_results if r['correct'])
        total_negatives = len(negative_results)
        avg_confidence_negatives = np.mean([r['confidence'] for r in negative_results])
        
        print(f"❌ Негативные данные:")
        print(f"   Правильно: {correct_negatives}/{total_negatives} ({correct_negatives/total_negatives*100:.1f}%)")
        print(f"   Средняя уверенность: {avg_confidence_negatives:.3f}")
    
    # Общие результаты
    all_results = positive_results + negative_results
    if all_results:
        total_correct = sum(1 for r in all_results if r['correct'])
        total_tests = len(all_results)
        overall_accuracy = total_correct / total_tests * 100
        
        print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
        
        # Анализ уверенности
        positive_confidences = [r['confidence'] for r in positive_results]
        negative_confidences = [r['confidence'] for r in negative_results]
        
        if positive_confidences:
            print(f"📈 Позитивные: мин={min(positive_confidences):.3f}, макс={max(positive_confidences):.3f}")
        if negative_confidences:
            print(f"📉 Негативные: мин={min(negative_confidences):.3f}, макс={max(negative_confidences):.3f}")

if __name__ == "__main__":
    main()