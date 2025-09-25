#!/usr/bin/env python3
"""
Автоматическая обработка собранных реальных данных
"""

import os
import sys
import librosa
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_audio_files(input_dir, output_dir, target_duration=1.5):
    """Обрабатывает аудиофайлы: нормализует, обрезает, конвертирует"""
    logger.info(f"🔄 Обработка файлов из {input_dir} в {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for file_path in Path(input_dir).glob("*.wav"):
        try:
            # Загружаем аудио
            audio, sr = librosa.load(str(file_path), sr=16000)
            
            # Нормализуем громкость
            audio = librosa.util.normalize(audio)
            
            # Обрезаем до целевой длительности
            target_samples = int(target_duration * sr)
            if len(audio) > target_samples:
                # Берем середину
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            elif len(audio) < target_samples:
                # Дополняем нулями
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Сохраняем обработанный файл
            output_file = os.path.join(output_dir, file_path.name)
            librosa.output.write_wav(output_file, audio, sr)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {file_path}: {e}")
            error_count += 1
    
    logger.info(f"✅ Обработано: {processed_count}, Ошибок: {error_count}")
    return processed_count, error_count

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Разделяет датасет на train/validation/test"""
    logger.info("📊 Разделение датасета на train/validation/test")
    
    import random
    random.seed(42)
    
    # Создаем директории
    splits = ["train", "validation", "test"]
    for split in splits:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    
    # Обрабатываем позитивные данные
    pos_dir = os.path.join(data_dir, "positives_processed")
    if os.path.exists(pos_dir):
        pos_files = list(Path(pos_dir).glob("*.wav"))
        random.shuffle(pos_files)
        
        n_train = int(len(pos_files) * train_ratio)
        n_val = int(len(pos_files) * val_ratio)
        
        train_files = pos_files[:n_train]
        val_files = pos_files[n_train:n_train + n_val]
        test_files = pos_files[n_train + n_val:]
        
        # Копируем файлы
        for files, split in [(train_files, "train"), (val_files, "validation"), (test_files, "test")]:
            split_dir = os.path.join(data_dir, split, "positives")
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                import shutil
                shutil.copy2(file_path, os.path.join(split_dir, file_path.name))
    
    # Обрабатываем негативные данные
    neg_dir = os.path.join(data_dir, "negatives_processed")
    if os.path.exists(neg_dir):
        neg_files = list(Path(neg_dir).glob("*.wav"))
        random.shuffle(neg_files)
        
        n_train = int(len(neg_files) * train_ratio)
        n_val = int(len(neg_files) * val_ratio)
        
        train_files = neg_files[:n_train]
        val_files = neg_files[n_train:n_train + n_val]
        test_files = neg_files[n_train + n_val:]
        
        # Копируем файлы
        for files, split in [(train_files, "train"), (val_files, "validation"), (test_files, "test")]:
            split_dir = os.path.join(data_dir, split, "negatives")
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                import shutil
                shutil.copy2(file_path, os.path.join(split_dir, file_path.name))
    
    logger.info("✅ Датасет разделен на train/validation/test")

def main():
    """Основная функция обработки данных"""
    logger.info("🎯 ОБРАБОТКА РЕАЛЬНЫХ ДАННЫХ")
    
    real_data_dir = "/home/microWakeWord_data/real_wake_word_data"
    
    # Обрабатываем позитивные данные
    pos_raw = os.path.join(real_data_dir, "positives_raw")
    pos_processed = os.path.join(real_data_dir, "positives_processed")
    if os.path.exists(pos_raw):
        process_audio_files(pos_raw, pos_processed)
    
    # Обрабатываем негативные данные
    neg_raw = os.path.join(real_data_dir, "noise_raw")
    neg_processed = os.path.join(real_data_dir, "negatives_processed")
    if os.path.exists(neg_raw):
        process_audio_files(neg_raw, neg_processed)
    
    # Разделяем на train/validation/test
    split_dataset(real_data_dir)
    
    logger.info("🎉 Обработка данных завершена!")

if __name__ == "__main__":
    main()
