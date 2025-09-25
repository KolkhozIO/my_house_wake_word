#!/usr/bin/env python3
"""
Генерация спектрограмм для негативных данных
"""

import os
import sys
from pathlib import Path

# Добавляем путь к библиотеке
sys.path.append(os.path.join(os.path.dirname(__file__), 'microwakeword'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.path_manager import paths

def generate_negative_spectrograms():
    """Генерирует спектрограммы для негативных данных"""
    
    print("🔄 Генерация спектрограмм для негативных данных...")
    
    try:
        from microwakeword.audio.clips import Clips
        from microwakeword.audio.augmentation import Augmentation
        from microwakeword.audio.spectrograms import SpectrogramGeneration
        from microwakeword.data import RaggedMmap
        
        # Путь к негативным данным
        negatives_dir = paths.NEGATIVES_PROCESSED
        
        if not os.path.exists(negatives_dir):
            print(f"❌ Директория негативных данных не найдена: {negatives_dir}")
            return False
        
        # Подсчитываем файлы
        neg_files = len([f for f in os.listdir(negatives_dir) if f.endswith('.wav')])
        print(f"📊 Найдено {neg_files} негативных файлов")
        
        # Создаем директории
        os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "training"), exist_ok=True)
        os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "validation"), exist_ok=True)
        os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "testing"), exist_ok=True)
        
        # Настройка негативных данных
        negatives_clips = Clips(
            input_directory=negatives_dir,
            file_pattern='*.wav',
            max_clip_duration_s=1.5,
            remove_silence=False,
            random_split_seed=10,
            split_count=0.1,  # 10% на валидацию/тест
        )
        
        # Аугментации для негативных данных
        negatives_augmentation = Augmentation(
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
        
        # Генерация спектрограмм для негативных данных
        negatives_spectrograms = SpectrogramGeneration(
            clips=negatives_clips,
            augmenter=negatives_augmentation,
            slide_frames=1,
            step_ms=4.42,
        )
        
        print("📊 Генерация негативных спектрограмм для обучения...")
        RaggedMmap.from_generator(
            out_dir=os.path.join(paths.FEATURES_NEGATIVES, "training", "wakeword_mmap"),
            sample_generator=negatives_spectrograms.spectrogram_generator(split="train", repeat=1),
            batch_size=1000,
            verbose=True,
        )
        
        print("📊 Генерация негативных спектрограмм для валидации...")
        RaggedMmap.from_generator(
            out_dir=os.path.join(paths.FEATURES_NEGATIVES, "validation", "wakeword_mmap"),
            sample_generator=negatives_spectrograms.spectrogram_generator(split="validation", repeat=1),
            batch_size=1000,
            verbose=True,
        )
        
        print("📊 Генерация негативных спектрограмм для тестирования...")
        RaggedMmap.from_generator(
            out_dir=os.path.join(paths.FEATURES_NEGATIVES, "testing", "wakeword_mmap"),
            sample_generator=negatives_spectrograms.spectrogram_generator(split="test", repeat=1),
            batch_size=1000,
            verbose=True,
        )
        
        print("✅ Генерация негативных спектрограмм завершена успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка генерации негативных спектрограмм: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция"""
    print("🎯 Генерация спектрограмм для негативных данных")
    print("="*50)
    
    if generate_negative_spectrograms():
        print("\n🎉 Генерация завершена успешно!")
        print("📁 Негативные спектрограммы готовы для обучения:")
        print(f"  - {paths.FEATURES_NEGATIVES}/")
        return True
    else:
        print("\n❌ Генерация завершилась с ошибками")
        return False

if __name__ == "__main__":
    main()