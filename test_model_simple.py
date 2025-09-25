#!/usr/bin/env python3
"""
Простое тестирование обученной модели microWakeWord только на позитивных данных
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

def load_positive_data():
    """Загружает только позитивные данные"""
    print("🔄 Загрузка позитивных данных...")
    
    try:
        import microwakeword.data as input_data
        
        # Создаем обработчик данных только для позитивных
        data_processor = input_data.FeatureHandler({
            'features': [
                {
                    'features_dir': paths.FEATURES_POSITIVES,
                    'penalty_weight': 1.0,
                    'sampling_weight': 1.0,
                    'truncation_strategy': 'random',
                    'truth': True,
                    'type': 'mmap'
                }
            ],
            'batch_size': 32,
            'training_input_shape': (194, 40),
            'stride': 1,
            'window_step_ms': 10
        })
        
        # Получаем данные
        test_data = data_processor.get_data(mode="train", batch_size=32, features_length=194)
        
        print(f"✅ Позитивные данные загружены")
        return test_data
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_on_positives(model, test_data):
    """Тестирует модель на позитивных данных"""
    print("🔄 Тестирование модели на позитивных данных...")
    
    try:
        predictions = []
        true_labels = []
        
        batch_count = 0
        for batch_x, batch_y in test_data:
            # Получаем предсказания
            batch_pred = model.predict(batch_x, verbose=0)
            predictions.extend(batch_pred.flatten())
            true_labels.extend(batch_y.flatten())
            
            batch_count += 1
            if batch_count >= 10:  # Ограничиваем количество батчей для теста
                break
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
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
            'true_labels': true_labels,
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
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛИ НА ПОЗИТИВНЫХ ДАННЫХ")
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
        'test_type': 'positive_data_only',
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
    print("🎯 Простое тестирование модели microWakeWord на позитивных данных")
    print("="*70)
    
    # Загружаем модель
    model = load_model()
    if model is None:
        return False
    
    # Загружаем позитивные данные
    test_data = load_positive_data()
    if test_data is None:
        return False
    
    # Тестируем модель
    results = test_model_on_positives(model, test_data)
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