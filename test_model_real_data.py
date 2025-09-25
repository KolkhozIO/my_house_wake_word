#!/usr/bin/env python3
"""
Тестирование модели на реальных данных microWakeWord
"""

import sys
import os
sys.path.append('/home/microWakeWord')

import numpy as np
import tensorflow as tf
from pathlib import Path
import glob
import random

def load_spectrograms_from_mmap(data_dir, max_samples=100):
    """Загружает спектрограммы из RaggedMmap файлов"""
    
    print(f"📁 Загрузка данных из: {data_dir}")
    
    try:
        # Ищем директории wakeword_mmap
        mmap_dirs = list(Path(data_dir).glob("*/wakeword_mmap"))
        if not mmap_dirs:
            print(f"❌ RaggedMmap директории не найдены в {data_dir}")
            return None
        
        print(f"📊 Найдено RaggedMmap директорий: {len(mmap_dirs)}")
        
        # Загружаем первую директорию
        mmap_dir = mmap_dirs[0]
        print(f"📁 Загружаем: {mmap_dir}")
        
        # Используем RaggedMmap для загрузки
        from microwakeword.data import RaggedMmap
        
        data = RaggedMmap(mmap_dir)
        print(f"📊 Размер данных: {len(data)}")
        
        # Берем случайные образцы
        indices = random.sample(range(len(data)), min(max_samples, len(data)))
        spectrograms = []
        
        for i, idx in enumerate(indices):
            if i % 20 == 0:
                print(f"🔄 Загружено: {i}/{len(indices)}")
            
            spec = data[idx]
            if spec.shape == (147, 40):  # Правильная форма для новой модели
                spectrograms.append(spec)
        
        print(f"✅ Загружено спектрограмм: {len(spectrograms)}")
        return np.array(spectrograms)
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None

def test_model_on_data(model_path, spectrograms, label="Unknown"):
    """Тестирует модель на данных"""
    
    print(f"🔄 Тестирование модели на {label} данных...")
    
    try:
        # Загружаем TFLite модель
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Форма входа модели: {input_details[0]['shape']}")
        print(f"📊 Форма данных: {spectrograms.shape}")
        
        # Подготавливаем данные для батча
        batch_size = input_details[0]['shape'][0]
        results = []
        
        # Обрабатываем по батчам
        for i in range(0, len(spectrograms), batch_size):
            batch = spectrograms[i:i+batch_size]
            
            # Дополняем до размера батча если нужно
            if len(batch) < batch_size:
                # Дублируем последний элемент
                while len(batch) < batch_size:
                    batch = np.vstack([batch, batch[-1:]])
            
            # Запускаем инференс
            interpreter.set_tensor(input_details[0]['index'], batch.astype(np.float32))
            interpreter.invoke()
            
            # Получаем результат
            output = interpreter.get_tensor(output_details[0]['index'])
            results.extend(output[:len(spectrograms[i:i+batch_size])])
        
        results = np.array(results)
        
        # Анализируем результаты
        print(f"📊 Результаты для {label}:")
        print(f"  Среднее: {np.mean(results):.4f}")
        print(f"  Медиана: {np.median(results):.4f}")
        print(f"  Мин: {np.min(results):.4f}")
        print(f"  Макс: {np.max(results):.4f}")
        print(f"  Стандартное отклонение: {np.std(results):.4f}")
        
        # Считаем активации (вероятности > 0.5)
        activations = np.sum(results > 0.5)
        activation_rate = activations / len(results)
        print(f"  Активации (>0.5): {activations}/{len(results)} ({activation_rate:.2%})")
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return None

def main():
    """Основная функция тестирования"""
    
    print("🎯 Тестирование модели microWakeWord на реальных данных")
    print("=" * 60)
    
    model_path = "/home/microWakeWord_data/original_library_model.tflite"
    
    # Проверяем что модель существует
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return False
    
    print(f"✅ Модель найдена: {model_path}")
    
    # Тестируем на позитивных данных
    print("\n🎯 ТЕСТИРОВАНИЕ НА ПОЗИТИВНЫХ ДАННЫХ")
    print("-" * 40)
    
    positives_dir = "/home/microWakeWord_data/features_positives"
    positive_data = load_spectrograms_from_mmap(positives_dir, max_samples=50)
    
    if positive_data is not None:
        positive_results = test_model_on_data(model_path, positive_data, "Позитивные")
    else:
        print("❌ Не удалось загрузить позитивные данные")
        positive_results = None
    
    # Тестируем на негативных данных
    print("\n🎯 ТЕСТИРОВАНИЕ НА НЕГАТИВНЫХ ДАННЫХ")
    print("-" * 40)
    
    negatives_dir = "/home/microWakeWord_data/features_negatives"
    negative_data = load_spectrograms_from_mmap(negatives_dir, max_samples=50)
    
    if negative_data is not None:
        negative_results = test_model_on_data(model_path, negative_data, "Негативные")
    else:
        print("❌ Не удалось загрузить негативные данные")
        negative_results = None
    
    # Анализ результатов
    print("\n📊 ОБЩИЙ АНАЛИЗ")
    print("=" * 40)
    
    if positive_results is not None and negative_results is not None:
        # Считаем метрики
        pos_mean = np.mean(positive_results)
        neg_mean = np.mean(negative_results)
        
        pos_activations = np.sum(positive_results > 0.5) / len(positive_results)
        neg_activations = np.sum(negative_results > 0.5) / len(negative_results)
        
        print(f"📈 Позитивные данные:")
        print(f"  Средняя вероятность: {pos_mean:.4f}")
        print(f"  Активации: {pos_activations:.2%}")
        
        print(f"📉 Негативные данные:")
        print(f"  Средняя вероятность: {neg_mean:.4f}")
        print(f"  Активации: {neg_activations:.2%}")
        
        # Разделение классов
        separation = pos_mean - neg_mean
        print(f"🎯 Разделение классов: {separation:.4f}")
        
        if separation > 0.1:
            print("✅ Хорошее разделение классов!")
        elif separation > 0.05:
            print("⚠️ Умеренное разделение классов")
        else:
            print("❌ Плохое разделение классов")
        
        # Wake word метрики
        print(f"\n🎤 WAKE WORD МЕТРИКИ:")
        print(f"  FRR (False Reject Rate): {1 - pos_activations:.2%}")
        print(f"  FAR (False Accept Rate): {neg_activations:.2%}")
        
        if pos_activations > 0.8 and neg_activations < 0.1:
            print("🎉 ОТЛИЧНЫЕ РЕЗУЛЬТАТЫ!")
        elif pos_activations > 0.7 and neg_activations < 0.2:
            print("✅ ХОРОШИЕ РЕЗУЛЬТАТЫ!")
        else:
            print("⚠️ ТРЕБУЕТСЯ ДОРАБОТКА")
    
    else:
        print("❌ Не удалось получить результаты для анализа")
        return False
    
    print("\n🎉 Тестирование завершено!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)