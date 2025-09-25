#!/usr/bin/env python3
"""
Генерация спектрограмм для TTS негативных данных
"""

import os
import sys
import time
from pathlib import Path

# Добавляем путь к библиотеке microWakeWord
sys.path.insert(0, './microwakeword')

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
from src.utils.path_manager import paths

def generate_negatives_spectrograms():
    """Генерирует спектрограммы для TTS негативных данных"""
    
    print("🚀 ГЕНЕРАЦИЯ СПЕКТРОГРАММ ДЛЯ TTS НЕГАТИВНЫХ ДАННЫХ")
    print("=" * 60)
    
    # Пути к данным - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    input_dir = paths.NEGATIVES_TTS
    output_dir = paths.FEATURES_NEGATIVES
    
    # Проверяем входные данные
    if not os.path.exists(input_dir):
        print(f"❌ Входная директория не найдена: {input_dir}")
        return False
    
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"📊 Найдено {len(input_files)} TTS негативных файлов")
    
    if len(input_files) == 0:
        print("❌ Нет WAV файлов для обработки")
        return False
    
    # Создаем выходную структуру
    os.makedirs(os.path.join(output_dir, "training", "wakeword_mmap"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation", "wakeword_mmap"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "wakeword_mmap"), exist_ok=True)
    
    try:
        # Импорты microWakeWord
        from microwakeword.audio.clips import Clips
        from microwakeword.audio.augmentation import Augmentation
        from microwakeword.audio.spectrograms import SpectrogramGeneration
        from mmap_ninja.ragged import RaggedMmap
        
        print("✅ Библиотека microWakeWord загружена")
        
        # Настройка Clips для негативных данных
        clips = Clips(
            input_directory=input_dir,
            file_pattern='*.wav',
            max_clip_duration_s=1.5,
            remove_silence=False,
            random_split_seed=10,
            split_count=0.1,  # 10% на валидацию/тест
        )
        
        print(f"📊 Настройка Clips: {input_dir}")
        
        # Аугментации для негативных данных
        augmentation = Augmentation(
            augmentation_duration_s=1.5,
            augmentation_probabilities={
                "SevenBandParametricEQ": 0.1,
                "TanhDistortion": 0.1,
                "PitchShift": 0.1,
                "BandStopFilter": 0.1,
                "AddColorNoise": 0.1,
                "Gain": 1.0,
            },
            min_jitter_s=0.195,
            max_jitter_s=0.205,
        )
        
        # Генерация спектрограмм
        spectrograms = SpectrogramGeneration(
            clips=clips,
            augmenter=augmentation,
            slide_frames=1,
            step_ms=4.42,
        )
        
        print("📊 Генерация спектрограмм для негативных данных...")
        
        # Генерируем данные для всех splits
        splits = ["train", "validation", "test"]
        
        for split in splits:
            print(f"📊 Генерация {split} данных...")
            
            output_path = os.path.join(output_dir, split, "wakeword_mmap")
            
            # Создаем RaggedMmap для сохранения
            RaggedMmap.from_generator(
                out_dir=output_path,
                sample_generator=spectrograms.spectrogram_generator(split=split, repeat=1),
                batch_size=1000,
                verbose=True,
            )
            
            print(f"✅ {split} данные сохранены в {output_path}")
        
        print("🎉 Генерация спектрограмм завершена!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка генерации спектрограмм: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    success = generate_negatives_spectrograms()
    
    if success:
        print("✅ Генерация завершена успешно!")
        
        # Проверяем результат - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
        output_dir = paths.FEATURES_NEGATIVES
        for split in ["training", "validation", "testing"]:
            split_dir = os.path.join(output_dir, split, "wakeword_mmap")
            if os.path.exists(split_dir):
                files = os.listdir(split_dir)
                print(f"📊 {split}: {len(files)} файлов")
            else:
                print(f"❌ {split}: директория не создана")
    else:
        print("❌ Генерация завершилась с ошибкой")
    
    return success

if __name__ == "__main__":
    main()