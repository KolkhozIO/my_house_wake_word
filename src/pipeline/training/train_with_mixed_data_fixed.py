#!/usr/bin/env python3
"""
Обучение модели на смешанных данных с шумом - ИСПРАВЛЕННАЯ ВЕРСИЯ
ТОЧНО копирует рабочий подход из train_model_only.py
"""

import os
import sys
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML
from src.utils.path_manager import paths

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Настройка для предотвращения проблем с переносами строк
sys.stdout.reconfigure(line_buffering=True)

# Варианты смешанных данных - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
MIXED_VARIANTS = {
    "conservative": {
        "name": "Консервативный",
        "positives_dir": paths.POSITIVES_RAW,  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
        "negatives_dir": paths.NEGATIVES_RAW,  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
        "hard_negatives_dir": paths.HARD_NEGATIVES_PARALLEL,  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
    }
}

def create_training_config(variant_name):
    """Создает конфигурацию для обучения - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML"""
    logger.info(f"📝 Создание конфигурации YAML для варианта: {variant_name}")
    
    # Проверяем существование данных - СТАТИЧЕСКИЕ ПУТИ ИЗ XML
    logger.info("🔍 Проверка существования данных...")
    
    if not os.path.exists(paths.FEATURES_POSITIVES):
        logger.error(f"❌ Позитивные спектрограммы не найдены: {paths.FEATURES_POSITIVES}")
        return None
    
    logger.info("✅ Позитивные спектрограммы найдены")
    logger.info("⚠️ Негативные спектрограммы будут сгенерированы автоматически")
    
    # СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    config = {
        'batch_size': 32,
        'clip_duration_ms': 1500,
        'eval_step_interval': 500,
        'features': [
            {
                'features_dir': paths.FEATURES_POSITIVES,  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
                'penalty_weight': 1.0,
                'sampling_weight': 1.0,
                'truncation_strategy': 'random',
                'truth': True,
                'type': 'mmap'
            },
            {
                'features_dir': paths.FEATURES_NEGATIVES,  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
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
        'train_dir': paths.get_model_dir(variant_name),  # СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
        'training_steps': [10000],
        'window_step_ms': 10
    }
    
    logger.info("✅ Конфигурация создана без ambient данных")
    
    # Сохраняем конфигурацию - СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
    config_path = paths.get_training_config_path(variant_name)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✅ Конфигурация сохранена: {config_path}")
    return config_path

def train_model(variant_name):
    """Запускает обучение модели - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML"""
    
    logger.info(f"🎯 Запуск обучения модели для варианта: {variant_name}")
    
    # СТАТИЧЕСКИЙ ПУТЬ К КОНФИГУРАЦИИ ИЗ XML
    config_path = paths.get_training_config_path(variant_name)
    
    if not os.path.exists(config_path):
        logger.error(f"❌ Конфигурация не найдена: {config_path}")
        return False
    
    logger.info("🎯 Запуск обучения...")
    
    # Создаем директорию для результатов - СТАТИЧЕСКИЙ ПУТЬ ИЗ XML
    train_dir = paths.get_model_dir(variant_name)
    os.makedirs(train_dir, exist_ok=True)
    
    try:
        # Используем библиотеку напрямую вместо subprocess - ТОЧНО как в рабочем коде
        logger.info("🔄 Загрузка библиотеки microwakeword...")
        
        import microwakeword.model_train_eval as mte
        import microwakeword.mixednet as mixednet
        import microwakeword.data as input_data
        
        # Создаем объект flags для совместимости - ТОЧНО как в рабочем коде
        class Flags:
            def __init__(self):
                self.training_config = config_path
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
        
        # Загружаем конфигурацию - ТОЧНО как в рабочем коде
        logger.info("📋 Загрузка конфигурации...")
        config = mte.load_config(flags, mixednet)
        
        # Создаем обработчик данных - ТОЧНО как в рабочем коде
        logger.info("📊 Создание обработчика данных...")
        data_processor = input_data.FeatureHandler(config)
        
        # Создаем модель - ТОЧНО как в рабочем коде
        logger.info("🏗️ Создание модели...")
        model = mixednet.model(flags, config["training_input_shape"], config["batch_size"])
        
        # Запускаем обучение - ТОЧНО как в рабочем коде
        logger.info("🚀 Запуск обучения...")
        print("\n🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ НА СМЕШАННЫХ ДАННЫХ\n", flush=True)
        mte.train_model(config, model, data_processor, restore_checkpoint=True)
        
        logger.info("✅ Обучение завершено успешно!")
        print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!\n", flush=True)
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка обучения: {e}")
        print(f"\n❌ ОШИБКА ОБУЧЕНИЯ: {e}", flush=True)
        print("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!\n", flush=True)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Обучение модели на смешанных данных с шумом - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    parser.add_argument("--variant", choices=list(MIXED_VARIANTS.keys()), 
                       default="conservative", help="Вариант смешанных данных")
    
    args = parser.parse_args()
    
    # Получаем информацию о варианте
    variant_info = MIXED_VARIANTS[args.variant]
    variant_name = variant_info["name"]
    
    logger.info(f"🎯 ОБУЧЕНИЕ НА СМЕШАННЫХ ДАННЫХ С ШУМОМ - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print(f"\n🎯 ОБУЧЕНИЕ НА СМЕШАННЫХ ДАННЫХ С ШУМОМ - ИСПРАВЛЕННАЯ ВЕРСИЯ", flush=True)
    logger.info(f"📊 Вариант: {variant_name}")
    print(f"📊 Вариант: {variant_name}\n", flush=True)
    
    # Создаем конфигурацию обучения
    config_path = create_training_config(variant_name.lower())
    if not config_path:
        logger.error("❌ Не удалось создать конфигурацию!")
        return 1
    
    # Обучаем модель
    success = train_model(variant_name.lower())
    if success:
        logger.info(f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info(f"📁 Конфигурация: {config_path}")
        logger.info(f"🚀 Готово для создания TFLite модели!")
        print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!", flush=True)
        return 0
    else:
        logger.error("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!")
        print("\n❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!", flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())