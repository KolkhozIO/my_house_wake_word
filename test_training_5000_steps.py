#!/usr/bin/env python3
"""
Тестовый скрипт обучения на 5000 шагов с централизованным логированием
"""

import os
import sys
import yaml
from pathlib import Path

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append('/home/microWakeWord')

# Импортируем централизованное логирование
from src.utils.centralized_logger import setup_logging, get_logger
from src.utils.print_replacer import enable_print_replacement

# Импортируем оригинальный скрипт обучения
from src.pipeline.training.train_model_only import main as train_main

def main():
    """Главная функция с централизованным логированием"""
    
    # Настройка централизованного логирования
    setup_logging()
    logger = get_logger("test_training_5000")
    enable_print_replacement("test_training_5000")
    
    logger.info("🚀 ЗАПУСК ТЕСТОВОГО ОБУЧЕНИЯ НА 5000 ШАГОВ")
    logger.info("=" * 50)
    
    # Проверяем конфигурацию
    config_path = "/home/microWakeWord/config/test_5000_steps.yaml"
    if not os.path.exists(config_path):
        logger.error(f"❌ Конфигурационный файл не найден: {config_path}")
        return False
    
    logger.info(f"📄 Используем конфигурацию: {config_path}")
    
    # Читаем конфигурацию для проверки
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_steps = config.get('model', {}).get('training_steps', [])
    logger.info(f"📊 Количество шагов обучения: {training_steps}")
    
    # Временно заменяем конфигурацию в оригинальном скрипте
    original_config_path = "/home/microWakeWord/config/unified_config.yaml"
    
    try:
        # Создаем бэкап оригинальной конфигурации
        backup_path = "/home/microWakeWord/config/unified_config.yaml.backup"
        if os.path.exists(original_config_path):
            import shutil
            shutil.copy2(original_config_path, backup_path)
            logger.info(f"💾 Создан бэкап конфигурации: {backup_path}")
        
        # Копируем тестовую конфигурацию
        import shutil
        shutil.copy2(config_path, original_config_path)
        logger.info(f"📋 Установлена тестовая конфигурация")
        
        # Запускаем обучение
        logger.info("🎯 НАЧИНАЕМ ОБУЧЕНИЕ...")
        logger.info("=" * 50)
        
        success = train_main()
        
        if success:
            logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        else:
            logger.error("❌ ОБУЧЕНИЕ ЗАВЕРШИЛОСЬ С ОШИБКАМИ!")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА ВО ВРЕМЯ ОБУЧЕНИЯ: {e}")
        import traceback
        logger.error(f"Детали ошибки: {traceback.format_exc()}")
        return False
        
    finally:
        # Восстанавливаем оригинальную конфигурацию
        try:
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, original_config_path)
                logger.info("🔄 Восстановлена оригинальная конфигурация")
                os.remove(backup_path)
                logger.info("🗑️ Удален бэкап конфигурации")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось восстановить конфигурацию: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)