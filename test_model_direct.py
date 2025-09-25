#!/usr/bin/env python3
"""
Прямое тестирование обученной модели microWakeWord
Загружает данные напрямую из MMAP файлов
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

# Добавляем путь к библиотеке
sys.path.append(os.path.join(os.path.dirname(__file__), 'microwakeword'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.path_manager import paths

def load_model():
    """Загружает обученную модель"""
    print("🔄 Загрузка обученной модели...")
    
    # Путь к весам модели
    weights_path = "/home/microWakeWord_data/trained_models/wakeword/best_weights.weights.h5"
    
    if not os.path.exists(weights_path):
        print(f"❌ Веса модели не найдены: {weights_path}")
        return None
    
    try:
        # Импортируем компоненты microWakeWord
        import microwakeword.mixednet as mixednet
        
        # Создаем объект flags для совместимости
        class Flags:
            def __init__(self):
                self.pointwise_filters = "48,48,48,48"
                self.repeat_in_block = "1,1,1,1"
                self.mixconv_kernel_sizes = "[5],[9],[13],[21]"
                self.residual_connection = "0,0,0,0"
                self.first_conv_filters = 32
                self.first_conv_kernel_size = 3
                self.stride = 1
                self.spatial_attention = False
                self.temporal_attention = False
                self.attention_heads = 1
                self.attention_dim = 64
                self.pooled = False
                self.max_pool = False
        
        flags = Flags()
        
        # Создаем модель
        model = mixednet.model(flags, (194, 40), 32)
        
        # Загружаем веса
        model.load_weights(weights_path)
        
        print("✅ Модель загружена успешно")
        return model
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_mmap_data():
    """Загружает данные напрямую из MMAP файлов"""
    print("🔄 Загрузка данных из MMAP файлов...")
    
    try:
        from microwakeword.data import RaggedMmap
        
        # Путь к позитивным данным
        positives_path = os.path.join(paths.FEATURES_POSITIVES, "training", "wakeword_mmap")
        
        if not os.path.exists(positives_path):
            print(f"❌ Позитивные данные не найдены: {positives_path}")
            return None
        
        # Загружаем данные
        data = RaggedMmap(positives_path)
        
        print(f"✅ Данные загружены: {len(data)} образцов")
        return data
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_on_data(model, data):
    """Тестирует модель на данных"""
    print("🔄 Тестирование модели...")
    
    try:
        predictions = []
        
        # Тестируем на первых 100 образцах
        test_size = min(100, len(data))
        
        for i in range(test_size):
            # Получаем образец
            sample = data[i]
            
            # Проверяем размер
            if sample.shape[0] != 194:
                # Обрезаем или дополняем до нужного размера
                if sample.shape[0] > 194:
                    sample = sample[:194, :]
                else:
                    # Дополняем нулями
                    padding = np.zeros((194 - sample.shape[0], 40))
                    sample = np.vstack([sample, padding])
            
            # Преобразуем в нужный формат
            if len(sample.shape) == 2:
                # Добавляем batch dimension
                sample = np.expand_dims(sample, axis=0)
            
            # Получаем предсказание
            pred = model.predict(sample, verbose=0)
            predictions.append(pred[0][0])
            
            if (i + 1) % 20 == 0:
                print(f"   Обработано {i + 1}/{test_size} образцов")
        
        predictions = np.array(predictions)
        
        print(f"📊 Обработано {len(predictions)} образцов")
        print(f"📊 Среднее предсказание: {np.mean(predictions):.4f}")
        print(f"📊 Медианное предсказание: {np.median(predictions):.4f}")
        print(f"📊 Минимальное предсказание: {np.min(predictions):.4f}")
        print(f"📊 Максимальное предсказание: {np.max(predictions):.4f}")
        
        # Проверяем сколько предсказаний выше порога 0.5
        above_threshold = np.sum(predictions > 0.5)
        print(f"📊 Предсказаний > 0.5: {above_threshold} из {len(predictions)} ({above_threshold/len(predictions)*100:.1f}%)")
        
        # Проверяем сколько предсказаний выше порога 0.9
        above_threshold_90 = np.sum(predictions > 0.9)
        print(f"📊 Предсказаний > 0.9: {above_threshold_90} из {len(predictions)} ({above_threshold_90/len(predictions)*100:.1f}%)")
        
        return {
            'predictions': predictions,
            'mean_prediction': np.mean(predictions),
            'median_prediction': np.median(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'above_threshold_50': above_threshold,
            'above_threshold_90': above_threshold_90,
            'total_samples': len(predictions)
        }
        
    except Exception as e:
        print(f"❌ Ошибка тестирования модели: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results(results):
    """Выводит результаты в красивом формате"""
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ microWakeWord")
    print("="*60)
    
    print(f"\n📈 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    print(f"   Среднее значение:     {results['mean_prediction']:.4f}")
    print(f"   Медианное значение:  {results['median_prediction']:.4f}")
    print(f"   Минимальное значение: {results['min_prediction']:.4f}")
    print(f"   Максимальное значение: {results['max_prediction']:.4f}")
    
    print(f"\n🎯 ПОРОГОВЫЕ ЗНАЧЕНИЯ:")
    print(f"   Предсказаний > 0.5:  {results['above_threshold_50']:4d} из {results['total_samples']:4d} ({results['above_threshold_50']/results['total_samples']*100:.1f}%)")
    print(f"   Предсказаний > 0.9:  {results['above_threshold_90']:4d} из {results['total_samples']:4d} ({results['above_threshold_90']/results['total_samples']*100:.1f}%)")
    
    print(f"\n✅ ОЦЕНКА РЕЗУЛЬТАТОВ:")
    
    # Оценка по wake word стандартам
    if results['mean_prediction'] > 0.8:
        print(f"   ✅ Среднее предсказание отличное (> 0.8)")
    elif results['mean_prediction'] > 0.6:
        print(f"   ⚠️  Среднее предсказание хорошее (> 0.6)")
    else:
        print(f"   ❌ Среднее предсказание плохое (< 0.6)")
    
    if results['above_threshold_50']/results['total_samples'] > 0.8:
        print(f"   ✅ Процент срабатываний отличный (> 80%)")
    elif results['above_threshold_50']/results['total_samples'] > 0.6:
        print(f"   ⚠️  Процент срабатываний хороший (> 60%)")
    else:
        print(f"   ❌ Процент срабатываний плохой (< 60%)")
    
    if results['above_threshold_90']/results['total_samples'] > 0.5:
        print(f"   ✅ Высокая уверенность отличная (> 50%)")
    elif results['above_threshold_90']/results['total_samples'] > 0.3:
        print(f"   ⚠️  Высокая уверенность хорошая (> 30%)")
    else:
        print(f"   ❌ Высокая уверенность плохая (< 30%)")
    
    print("="*60)

def save_results(results):
    """Сохраняет результаты в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/microWakeWord_data/model_test_results_{timestamp}.json"
    
    # Подготавливаем данные для JSON
    results_json = {
        'timestamp': timestamp,
        'model_path': '/home/microWakeWord_data/trained_models/wakeword/best_weights.weights.h5',
        'test_type': 'direct_mmap_data',
        'results': {
            'mean_prediction': float(results['mean_prediction']),
            'median_prediction': float(results['median_prediction']),
            'min_prediction': float(results['min_prediction']),
            'max_prediction': float(results['max_prediction']),
            'above_threshold_50': int(results['above_threshold_50']),
            'above_threshold_90': int(results['above_threshold_90']),
            'total_samples': int(results['total_samples'])
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"💾 Результаты сохранены: {results_file}")

def main():
    """Основная функция"""
    print("🎯 Прямое тестирование модели microWakeWord")
    print("="*50)
    
    # Загружаем модель
    model = load_model()
    if model is None:
        return False
    
    # Загружаем данные
    data = load_mmap_data()
    if data is None:
        return False
    
    # Тестируем модель
    results = test_model_on_data(model, data)
    if results is None:
        return False
    
    # Выводим результаты
    print_results(results)
    
    # Сохраняем результаты
    save_results(results)
    
    print("\n🎉 Тестирование завершено успешно!")
    return True

if __name__ == "__main__":
    main()