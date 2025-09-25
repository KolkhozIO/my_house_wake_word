#!/usr/bin/env python3
"""
Тестирование модели microWakeWord на реальных данных
"""

import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import logging
from pathlib import Path
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """Загружает обученную модель"""
    logger.info("🔄 Загрузка модели...")
    
    try:
        # Загружаем конфигурацию
        config_path = '/home/microWakeWord_data/training_parameters.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Создаем модель
        from microwakeword.mixednet import model
        
        class Flags:
            def __init__(self, config):
                self.pointwise_filters = config.get('pointwise_filters', '48, 48, 48, 48')
                self.repeat_in_block = config.get('repeat_in_block', '1, 1, 1, 1')
                self.mixconv_kernel_sizes = config.get('mixconv_kernel_sizes', '[5], [9], [13], [21]')
                self.residual_connection = config.get('residual_connection', '0,0,0,0')
                self.max_pool = config.get('max_pool', 0)
                self.first_conv_filters = config.get('first_conv_filters', 32)
                self.first_conv_kernel_size = config.get('first_conv_kernel_size', 3)
                self.spatial_attention = config.get('spatial_attention', 0)
                self.pooled = config.get('pooled', 0)
                self.stride = config.get('stride', 1)
        
        flags = Flags(config)
        
        # Добавляем spectrogram_length
        from microwakeword.mixednet import spectrogram_slices_dropped
        config['spectrogram_length_final_layer'] = config.get('spectrogram_length_final_layer', 226)
        config['spectrogram_length'] = config['spectrogram_length_final_layer'] + spectrogram_slices_dropped(flags)
        
        # Создаем модель с правильной формой
        from microwakeword.layers import modes
        config['training_input_shape'] = modes.get_input_data_shape(config, modes.Modes.TRAINING)
        
        model_instance = model(flags, shape=config['training_input_shape'], batch_size=1)
        
        # Загружаем лучшие веса
        weights_path = '/home/microWakeWord_data/original_library_model_best_weights.weights.h5'
        model_instance.load_weights(weights_path)
        
        logger.info("✅ Модель загружена успешно")
        return model_instance, config
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None

def load_audio_file(file_path):
    """Загружает аудиофайл и конвертирует в спектрограмму"""
    try:
        import librosa
        
        # Загружаем аудио
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Создаем спектрограмму
        spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=40,
            hop_length=160,
            n_fft=1024
        )
        
        # Логарифмируем
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Нормализуем
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
        
        return spectrogram
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки аудио {file_path}: {e}")
        return None

def test_model_on_files(model, config, test_files, expected_label, test_name):
    """Тестирует модель на наборе файлов"""
    logger.info(f"🧪 Тестирование {test_name} ({len(test_files)} файлов)")
    
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    
    for file_path in test_files:
        try:
            # Загружаем спектрограмму
            spectrogram = load_audio_file(file_path)
            if spectrogram is None:
                continue
            
            # Подготавливаем данные для модели
            # Обрезаем или дополняем до нужной длины
            target_length = config['training_input_shape'][0]
            if spectrogram.shape[1] > target_length:
                spectrogram = spectrogram[:, :target_length]
            elif spectrogram.shape[1] < target_length:
                # Дополняем нулями
                padding = np.zeros((spectrogram.shape[0], target_length - spectrogram.shape[1]))
                spectrogram = np.concatenate([spectrogram, padding], axis=1)
            
            # ИСПРАВЛЕНО: Транспонируем для правильной формы (272, 40) вместо (40, 272)
            spectrogram = spectrogram.T  # Транспонируем: (40, 272) -> (272, 40)
            
            # Добавляем batch dimension
            spectrogram = np.expand_dims(spectrogram, axis=0)
            
            # Предсказание
            prediction = model.predict(spectrogram, verbose=0)
            predicted_label = 1 if prediction[0][0] > 0.5 else 0
            
            predictions.append(prediction[0][0])
            
            # Проверяем правильность
            if predicted_label == expected_label:
                correct_predictions += 1
            total_predictions += 1
            
            logger.info(f"📁 {os.path.basename(file_path)}: {prediction[0][0]:.4f} -> {predicted_label} (ожидалось: {expected_label})")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки файла {file_path}: {e}")
            continue
    
    # Статистика
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = np.mean(predictions) if predictions else 0
    
    logger.info(f"📊 {test_name} результаты:")
    logger.info(f"   Правильных предсказаний: {correct_predictions}/{total_predictions}")
    logger.info(f"   Точность: {accuracy:.2%}")
    logger.info(f"   Средняя уверенность: {avg_confidence:.4f}")
    
    return accuracy, avg_confidence, predictions

def main():
    """Основная функция тестирования"""
    logger.info("🎯 ТЕСТИРОВАНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
    
    # Загружаем модель
    model, config = load_model()
    if model is None:
        logger.error("❌ Не удалось загрузить модель")
        return
    
    # Подготавливаем тестовые файлы
    positives_dir = "/home/microWakeWord_data/positives_final"
    negatives_dir = "/home/microWakeWord_data/negatives_real"
    
    # Получаем списки файлов
    positive_files = [os.path.join(positives_dir, f) for f in os.listdir(positives_dir) if f.endswith('.wav')]
    negative_files = [os.path.join(negatives_dir, f) for f in os.listdir(negatives_dir) if f.endswith('.wav')]
    
    # Выбираем случайные файлы для тестирования
    test_positive_count = min(50, len(positive_files))
    test_negative_count = min(100, len(negative_files))
    
    test_positive_files = random.sample(positive_files, test_positive_count)
    test_negative_files = random.sample(negative_files, test_negative_count)
    
    logger.info(f"📁 Позитивных файлов для тестирования: {len(test_positive_files)}")
    logger.info(f"📁 Негативных файлов для тестирования: {len(test_negative_files)}")
    
    # Тестируем на позитивных данных
    pos_accuracy, pos_confidence, pos_predictions = test_model_on_files(
        model, config, test_positive_files, 1, "ПОЗИТИВНЫЕ ДАННЫЕ"
    )
    
    # Тестируем на негативных данных
    neg_accuracy, neg_confidence, neg_predictions = test_model_on_files(
        model, config, test_negative_files, 0, "НЕГАТИВНЫЕ ДАННЫЕ"
    )
    
    # Общая статистика
    total_correct = (pos_accuracy * len(test_positive_files)) + (neg_accuracy * len(test_negative_files))
    total_files = len(test_positive_files) + len(test_negative_files)
    overall_accuracy = total_correct / total_files if total_files > 0 else 0
    
    logger.info("🎯 ОБЩИЕ РЕЗУЛЬТАТЫ:")
    logger.info(f"   Общая точность: {overall_accuracy:.2%}")
    logger.info(f"   Точность на позитивных: {pos_accuracy:.2%}")
    logger.info(f"   Точность на негативных: {neg_accuracy:.2%}")
    logger.info(f"   Средняя уверенность (позитивные): {pos_confidence:.4f}")
    logger.info(f"   Средняя уверенность (негативные): {neg_confidence:.4f}")
    
    # Анализ распределения предсказаний
    logger.info("📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ:")
    logger.info(f"   Позитивные предсказания: мин={min(pos_predictions):.4f}, макс={max(pos_predictions):.4f}, среднее={np.mean(pos_predictions):.4f}")
    logger.info(f"   Негативные предсказания: мин={min(neg_predictions):.4f}, макс={max(neg_predictions):.4f}, среднее={np.mean(neg_predictions):.4f}")
    
    # Проверка на переобучение
    if pos_confidence > 0.9 and neg_confidence < 0.1:
        logger.warning("⚠️ ВОЗМОЖНОЕ ПЕРЕОБУЧЕНИЕ: Слишком высокая уверенность")
    elif pos_confidence < 0.5 or neg_confidence > 0.5:
        logger.warning("⚠️ ПРОБЛЕМЫ С МОДЕЛЬЮ: Низкая уверенность в предсказаниях")
    else:
        logger.info("✅ МОДЕЛЬ РАБОТАЕТ КОРРЕКТНО")

if __name__ == "__main__":
    main()