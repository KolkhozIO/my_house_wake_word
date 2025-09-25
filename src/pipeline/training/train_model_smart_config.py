#!/usr/bin/env python3
"""
Умная система конфигурации для обучения microWakeWord
"""

import os
import sys
import yaml
import time
from datetime import datetime

# Добавляем путь к проекту
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths

def create_smart_training_config(force_recreate=False, custom_params=None):
    """
    Создает или обновляет конфигурацию для обучения с умной логикой
    
    Args:
        force_recreate (bool): Принудительно пересоздать конфигурацию
        custom_params (dict): Кастомные параметры для переопределения
    """
    
    print("🔧 Умное управление конфигурацией обучения...", flush=True)
    
    data_dir = "/home/microWakeWord_data"
    config_path = os.path.join(data_dir, 'training_parameters.yaml')
    
    # Проверяем наличие сгенерированных спектрограмм
    required_dirs = [
        os.path.join(paths.FEATURES_POSITIVES, "training", "wakeword_mmap"),
        os.path.join(paths.FEATURES_NEGATIVES, "training", "wakeword_mmap"),
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Директория спектрограмм не найдена: {dir_path}", flush=True)
            print("   Сначала запустите generate_spectrograms.py", flush=True)
            return False
    
    # Проверяем существующую конфигурацию
    existing_config = None
    config_needs_update = False
    
    if os.path.exists(config_path) and not force_recreate:
        try:
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f)
            print("📋 Найдена существующая конфигурация", flush=True)
            
            # Проверяем ключевые параметры
            if existing_config:
                print(f"   clip_duration_ms: {existing_config.get('clip_duration_ms', 'НЕТ')}", flush=True)
                print(f"   batch_size: {existing_config.get('batch_size', 'НЕТ')}", flush=True)
                
                # Проверяем корректность критических параметров
                critical_checks = {
                    'clip_duration_ms': 1030,  # Должно соответствовать данным (147, 40)
                    'window_step_ms': 10,
                    'batch_size': 32
                }
                
                for param, expected_value in critical_checks.items():
                    if existing_config.get(param) != expected_value:
                        print(f"⚠️ {param} некорректен: {existing_config.get(param)} != {expected_value}", flush=True)
                        config_needs_update = True
                
                if not config_needs_update:
                    print("✅ Существующая конфигурация корректна, используем её", flush=True)
                    return True
                else:
                    print("⚠️ Конфигурация требует обновления", flush=True)
                    
        except Exception as e:
            print(f"⚠️ Ошибка чтения конфигурации: {e}", flush=True)
            config_needs_update = True
    
    # Создаем/обновляем конфигурацию
    print("📝 Создание/обновление конфигурации YAML...", flush=True)
    
    # Базовые параметры (оптимизированы для данных формы 147x40)
    base_config = {
        'batch_size': 32,
        'clip_duration_ms': 1030,  # КРИТИЧНО: соответствует данным (147, 40)
        'eval_step_interval': 500,
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
        'freq_mask_count': [2],
        'freq_mask_max_size': [15],
        'learning_rates': [0.001],
        'maximization_metric': 'average_viable_recall',
        'minimization_metric': None,
        'negative_class_weight': [1],
        'positive_class_weight': [1],
        'target_minimization': 0.9,
        'time_mask_count': [2],
        'time_mask_max_size': [15],
        'train_dir': os.path.join(data_dir, 'trained_models/wakeword'),
        'training_steps': [10000],
        'window_step_ms': 10
    }
    
    # Применяем кастомные параметры
    if custom_params:
        print("🔧 Применяем кастомные параметры:", flush=True)
        for key, value in custom_params.items():
            if key in base_config:
                old_value = base_config[key]
                base_config[key] = value
                print(f"   {key}: {old_value} → {value}", flush=True)
            else:
                print(f"   ⚠️ Неизвестный параметр: {key}", flush=True)
    
    # Если есть существующая конфигурация, сохраняем её параметры где возможно
    if existing_config and not force_recreate:
        # Сохраняем кастомные параметры из существующей конфигурации
        preserve_keys = ['learning_rates', 'training_steps', 'negative_class_weight', 'positive_class_weight']
        for key in preserve_keys:
            if key in existing_config and key not in (custom_params or {}):
                base_config[key] = existing_config[key]
                print(f"   Сохранен параметр: {key} = {existing_config[key]}", flush=True)
    
    # Создаем резервную копию существующей конфигурации
    if existing_config and os.path.exists(config_path):
        backup_path = f"{config_path}.backup.{int(time.time())}"
        try:
            with open(backup_path, 'w') as f:
                yaml.dump(existing_config, f, default_flow_style=False)
            print(f"💾 Создана резервная копия: {backup_path}", flush=True)
        except Exception as e:
            print(f"⚠️ Не удалось создать резервную копию: {e}", flush=True)
    
    # Сохраняем конфигурацию
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)
    
    print(f"✅ Конфигурация сохранена: {config_path}", flush=True)
    print(f"   clip_duration_ms: {base_config['clip_duration_ms']} (соответствует данным)", flush=True)
    print(f"   batch_size: {base_config['batch_size']}", flush=True)
    print(f"   training_steps: {base_config['training_steps']}", flush=True)
    
    return True

def validate_config():
    """Проверяет корректность текущей конфигурации"""
    
    print("🔍 Проверка конфигурации...", flush=True)
    
    config_path = "/home/microWakeWord_data/training_parameters.yaml"
    
    if not os.path.exists(config_path):
        print("❌ Файл конфигурации не найден", flush=True)
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Проверяем критически важные параметры
        critical_params = {
            'clip_duration_ms': 1030,
            'window_step_ms': 10,
            'batch_size': 32
        }
        
        all_valid = True
        for param, expected_value in critical_params.items():
            actual_value = config.get(param)
            if actual_value != expected_value:
                print(f"❌ {param}: {actual_value} != {expected_value}", flush=True)
                all_valid = False
            else:
                print(f"✅ {param}: {actual_value}", flush=True)
        
        if all_valid:
            print("🎉 Конфигурация корректна!", flush=True)
        else:
            print("⚠️ Конфигурация требует исправления", flush=True)
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Ошибка проверки конфигурации: {e}", flush=True)
        return False

if __name__ == "__main__":
    # Тестируем умную конфигурацию
    print("🧪 Тестирование умной системы конфигурации")
    print("=" * 50)
    
    # Проверяем текущую конфигурацию
    validate_config()
    
    print()
    print("🔧 Создание конфигурации с кастомными параметрами...")
    
    # Создаем конфигурацию с кастомными параметрами
    custom_params = {
        'learning_rates': [0.0005],  # Более низкая скорость обучения
        'training_steps': [5000]      # Меньше шагов для быстрого тестирования
    }
    
    success = create_smart_training_config(
        force_recreate=False,
        custom_params=custom_params
    )
    
    if success:
        print("✅ Умная конфигурация создана успешно!")
        validate_config()
    else:
        print("❌ Ошибка создания конфигурации")