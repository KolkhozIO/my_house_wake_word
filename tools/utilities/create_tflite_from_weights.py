#!/usr/bin/env python3
"""
Создание TFLite модели из обученных весов
Обходит ошибку валидации и создает готовую модель для ESPHome
"""

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf

# Добавляем путь к оригинальной библиотеке
sys.path.insert(0, '/home/microWakeWord/backups/mww_orig')

from microwakeword import mixednet
from microwakeword.layers import modes

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tflite_from_weights():
    """Создает TFLite модель из обученных весов"""
    
    # Путь к обученной модели
    model_dir = "/home/microWakeWord_data/models/historical/trained_models/limited_model_36529"
    weights_path = os.path.join(model_dir, "last_weights.weights.h5")
    
    if not os.path.exists(weights_path):
        logger.error(f"❌ Файл весов не найден: {weights_path}")
        return False
    
    logger.info(f"📁 Загружаем веса из: {weights_path}")
    
    try:
        # Создаем конфигурацию модели
        config = {
            "spectrogram_length_final_layer": 272,
            "batch_size": 16,
            "learning_rate": 0.001,
            "training_steps": 100,
            "validation_steps": 10,
            "features_dir": "/home/microWakeWord_data/limited_features",
            "features_dir_negatives": "/home/microWakeWord_data/limited_features_negatives",
            "train_dir": model_dir,
            "summaries_dir": os.path.join(model_dir, "logs/"),
        }
        
        # Создаем флаги для вычисления spectrogram_length
        class Flags:
            def __init__(self):
                self.pointwise_filters = '48, 48, 48, 48'
                self.depthwise_filters = '48, 48, 48, 48'
                self.residual_connection = '0,0,0,0'
                self.strides = '1,1,1,1'
                self.dilation_rate = '1,1,1,1'
                self.kernel_size = '3,3,3,3'
                self.pool_size = '2,2,2,2'
                self.dropout_rate = 0.1
                self.activation = 'relu'
                self.use_batch_norm = True
                self.use_separable_conv = True
                # Добавляем недостающие атрибуты ТОЧНО КАК В ОБУЧЕНИИ
                self.first_conv_filters = 32
                self.first_conv_kernel_size = 3
                self.repeat_in_block = '1, 1, 1, 1'
                self.mixconv_kernel_sizes = '[5], [9], [13], [21]'
                self.stride = 1
                self.spatial_attention = 0
                self.pooled = 0
        
        flags = Flags()
        
        # Используем ТОЧНЫЕ параметры из training_config.yaml
        config["spectrogram_length"] = 272
        config["spectrogram_length_final_layer"] = 226
        config["mode"] = modes.Modes.TRAINING
        
        # Получаем правильную форму входа ТАК ЖЕ КАК В ОБУЧЕНИИ
        input_shape = modes.get_input_data_shape(config, modes.Modes.TRAINING)
        config["training_input_shape"] = input_shape
        
        logger.info(f"📊 Spectrogram length: {config['spectrogram_length']}")
        logger.info(f"📊 Input shape: {input_shape}")
        
        
        # Создаем модель
        logger.info("🏗️ Создание модели...")
        model = mixednet.model(flags, shape=input_shape, batch_size=config['batch_size'])
        
        # Загружаем веса
        logger.info("📥 Загрузка весов...")
        model.load_weights(weights_path)
        
        # Сохраняем модель в SavedModel формате сначала
        logger.info("💾 Сохранение модели в SavedModel...")
        saved_model_path = "/home/microWakeWord_data/original_library_model_saved"
        model.export(saved_model_path)
        
        # Конвертируем из SavedModel в TFLite
        logger.info("🔄 Конвертация в TFLite из SavedModel...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Конвертируем в TFLite
        tflite_model = converter.convert()
        
        # Сохраняем TFLite модель
        tflite_path = os.path.join(model_dir, "model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Создаем манифест для ESPHome
        manifest = {
            "version": 2,
            "type": "micro",
            "model": "model.tflite",
            "input_shape": input_shape,
            "spectrogram_length": config["spectrogram_length"],
            "wake_word": "милый дом",
            "description": "Модель обучена на смешанных данных (TTS + реальные негативные)"
        }
        
        manifest_path = os.path.join(model_dir, "model.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Копируем модель в основную директорию
        main_tflite_path = "/home/microWakeWord_data/limited_model.tflite"
        main_manifest_path = "/home/microWakeWord_data/limited_model.json"
        
        import shutil
        shutil.copy2(tflite_path, main_tflite_path)
        shutil.copy2(manifest_path, main_manifest_path)
        
        logger.info(f"✅ TFLite модель создана: {tflite_path}")
        logger.info(f"✅ Манифест создан: {manifest_path}")
        logger.info(f"✅ Основная модель: {main_tflite_path}")
        logger.info(f"✅ Основной манифест: {main_manifest_path}")
        
        # Проверяем размер модели
        model_size = os.path.getsize(tflite_path) / 1024  # KB
        logger.info(f"📊 Размер модели: {model_size:.1f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания TFLite модели: {e}")
        import traceback
        logger.error(f"❌ Детали ошибки: {traceback.format_exc()}")
        return False

def main():
    """Основная функция"""
    logger.info("🎯 СОЗДАНИЕ TFLITE МОДЕЛИ ИЗ ОБУЧЕННЫХ ВЕСОВ")
    
    success = create_tflite_from_weights()
    
    if success:
        logger.info("🎉 TFLite модель создана успешно!")
        logger.info("📁 Модель готова для использования в ESPHome")
    else:
        logger.error("❌ Не удалось создать TFLite модель")
        sys.exit(1)

if __name__ == "__main__":
    main()