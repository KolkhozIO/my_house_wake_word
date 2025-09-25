#!/usr/bin/env python3
"""
Тестирование обученной модели microWakeWord на текущих данных
Показывает метрики производительности
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

def load_test_data():
    """Загружает тестовые данные"""
    print("🔄 Загрузка тестовых данных...")
    
    try:
        import microwakeword.data as input_data
        
        # Создаем обработчик данных
        data_processor = input_data.FeatureHandler({
            'features': [
                {
                    'features_dir': paths.FEATURES_POSITIVES,
                    'penalty_weight': 1.0,
                    'sampling_weight': 1.0,
                    'truncation_strategy': 'random',
                    'truth': True,
                    'type': 'mmap'
                },
                {
                    'features_dir': paths.FEATURES_NEGATIVES,
                    'penalty_weight': 1.0,
                    'sampling_weight': 1.0,
                    'truncation_strategy': 'random',
                    'truth': False,
                    'type': 'mmap'
                }
            ],
            'batch_size': 32,
            'training_input_shape': (194, 40),
            'stride': 1,
            'window_step_ms': 10
        })
        
        # Получаем данные для тестирования (используем train режим)
        test_data = data_processor.get_data(mode="train", batch_size=32, features_length=194)
        
        print(f"✅ Тестовые данные загружены")
        return test_data
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model, test_data):
    """Оценивает модель на тестовых данных"""
    print("🔄 Оценка модели...")
    
    try:
        # Предсказания модели
        predictions = []
        true_labels = []
        
        for batch_x, batch_y in test_data:
            # Получаем предсказания
            batch_pred = model.predict(batch_x, verbose=0)
            predictions.extend(batch_pred.flatten())
            true_labels.extend(batch_y.flatten())
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        # Вычисляем метрики
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # Бинарные предсказания (порог 0.5)
        binary_pred = (predictions > 0.5).astype(int)
        
        # Метрики
        accuracy = accuracy_score(true_labels, binary_pred)
        precision = precision_score(true_labels, binary_pred, zero_division=0)
        recall = recall_score(true_labels, binary_pred, zero_division=0)
        f1 = f1_score(true_labels, binary_pred, zero_division=0)
        
        # AUC (если есть оба класса)
        try:
            auc = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc = 0.0
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, binary_pred)
        
        # Wake Word специфичные метрики
        tn, fp, fn, tp = cm.ravel()
        
        # False Reject Rate (FRR) - процент пропущенных wake words
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # False Accept Rate (FAR) - процент ложных срабатываний
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Accepts per Hour (FA/h) - примерная оценка
        # Предполагаем что модель работает 1 час = 3600 секунд
        # Каждые 10ms = 100 проверок в секунду = 360,000 проверок в час
        checks_per_hour = 360000
        fa_per_hour = far * checks_per_hour
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'frr': frr,
            'far': far,
            'fa_per_hour': fa_per_hour,
            'confusion_matrix': cm,
            'total_samples': len(true_labels),
            'positive_samples': int(np.sum(true_labels)),
            'negative_samples': int(len(true_labels) - np.sum(true_labels))
        }
        
    except Exception as e:
        print(f"❌ Ошибка оценки модели: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_metrics(metrics):
    """Выводит метрики в красивом формате"""
    print("\n" + "="*60)
    print("📊 МЕТРИКИ ОБУЧЕННОЙ МОДЕЛИ microWakeWord")
    print("="*60)
    
    print(f"\n📈 ОБЩИЕ МЕТРИКИ:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"   AUC:       {metrics['auc']:.4f}")
    
    print(f"\n🎯 WAKE WORD СПЕЦИФИЧНЫЕ МЕТРИКИ:")
    print(f"   FRR (False Reject Rate):     {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
    print(f"   FAR (False Accept Rate):     {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
    print(f"   FA/h (False Accepts/hour):  {metrics['fa_per_hour']:.1f}")
    
    print(f"\n📊 ДАННЫЕ:")
    print(f"   Всего образцов:     {metrics['total_samples']}")
    print(f"   Позитивных:        {metrics['positive_samples']}")
    print(f"   Негативных:        {metrics['negative_samples']}")
    print(f"   Соотношение:       {metrics['negative_samples']/metrics['positive_samples']:.1f}:1")
    
    print(f"\n🔢 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives:  {cm[0,0]:6d}")
    print(f"   False Positives: {cm[0,1]:6d}")
    print(f"   False Negatives: {cm[1,0]:6d}")
    print(f"   True Positives:  {cm[1,1]:6d}")
    
    print(f"\n✅ ОЦЕНКА РЕЗУЛЬТАТОВ:")
    
    # Оценка по wake word стандартам
    if metrics['frr'] < 0.05:
        print(f"   ✅ FRR отличный (< 5%)")
    elif metrics['frr'] < 0.1:
        print(f"   ⚠️  FRR хороший (< 10%)")
    else:
        print(f"   ❌ FRR плохой (> 10%)")
    
    if metrics['fa_per_hour'] < 1.0:
        print(f"   ✅ FA/h отличный (< 1/час)")
    elif metrics['fa_per_hour'] < 5.0:
        print(f"   ⚠️  FA/h хороший (< 5/час)")
    else:
        print(f"   ❌ FA/h плохой (> 5/час)")
    
    if metrics['auc'] > 0.9:
        print(f"   ✅ AUC отличный (> 0.9)")
    elif metrics['auc'] > 0.8:
        print(f"   ⚠️  AUC хороший (> 0.8)")
    else:
        print(f"   ❌ AUC плохой (< 0.8)")
    
    print("="*60)

def save_metrics(metrics):
    """Сохраняет метрики в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = f"/home/microWakeWord_data/model_metrics_{timestamp}.json"
    
    # Подготавливаем данные для JSON
    metrics_json = {
        'timestamp': timestamp,
        'model_path': '/home/microWakeWord_data/trained_models/wakeword/best_weights.weights.h5',
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'frr': float(metrics['frr']),
            'far': float(metrics['far']),
            'fa_per_hour': float(metrics['fa_per_hour']),
            'total_samples': int(metrics['total_samples']),
            'positive_samples': int(metrics['positive_samples']),
            'negative_samples': int(metrics['negative_samples'])
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"💾 Метрики сохранены: {metrics_file}")

def main():
    """Основная функция"""
    print("🎯 Тестирование модели microWakeWord")
    print("="*50)
    
    # Загружаем модель
    model = load_model()
    if model is None:
        return False
    
    # Загружаем тестовые данные
    test_data = load_test_data()
    if test_data is None:
        return False
    
    # Оцениваем модель
    metrics = evaluate_model(model, test_data)
    if metrics is None:
        return False
    
    # Выводим метрики
    print_metrics(metrics)
    
    # Сохраняем метрики
    save_metrics(metrics)
    
    print("\n🎉 Тестирование завершено успешно!")
    return True

if __name__ == "__main__":
    main()