#!/usr/bin/env python3
"""
Генерация спектрограмм для microWakeWord
Этап 1: Только генерация спектрограмм без обучения
"""

import os
import sys
from pathlib import Path

# Добавляем путь к их библиотеке
sys.path.insert(0, './microwakeword')

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
from src.utils.path_manager import paths

def create_data_in_their_format():
    """Создает данные в формате, который ожидает их библиотека"""
    
    print("📁 Создание данных в формате microWakeWord...")
    
    # Создаем структуру директорий - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "training"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "validation"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "testing"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "training_ambient"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "testing_ambient"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_POSITIVES, "validation_ambient"), exist_ok=True)
    
    # Логируем входные данные - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_PROCESSED  # Используем уже аугментированные негативные данные
    
    if os.path.exists(positives_dir):
        pos_files = len([f for f in os.listdir(positives_dir) if f.endswith('.wav')])
        print(f"📊 ВХОДНЫЕ ДАННЫЕ - Позитивные: {pos_files} файлов")
    else:
        print(f"❌ Директория позитивных данных не найдена: {positives_dir}")
        return False
        
    if os.path.exists(negatives_dir):
        neg_files = len([f for f in os.listdir(negatives_dir) if f.endswith('.wav')])
        print(f"📊 ВХОДНЫЕ ДАННЫЕ - Негативные: {neg_files} файлов")
    else:
        print(f"❌ Директория негативных данных не найдена: {negatives_dir}")
        return False
    
    # Используем их компоненты для создания данных
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from microwakeword.data import RaggedMmap
    
    print("🔧 Настройка компонентов microWakeWord...")
    
    # Настройка позитивных данных с семплированием на лету
    positives_clips = Clips(
        input_directory=positives_dir,
        file_pattern='*.wav',
        max_clip_duration_s=1.5,  # Унифицированная длительность с семплированием
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,  # 10% на валидацию/тест
    )
    
    print(f"📊 НАСТРОЙКА CLIPS - Позитивные данные из: {positives_dir}")
    print(f"📊 НАСТРОЙКА CLIPS - Split count: 0.1 (10% на валидацию/тест)")
    
    # Унифицированные аугментации для позитивных данных
    positives_augmentation = Augmentation(
        augmentation_duration_s=1.5,  # Соответствует max_clip_duration_s
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
    
    # Генерация спектрограмм для позитивных данных
    positives_spectrograms = SpectrogramGeneration(
        clips=positives_clips,
        augmenter=positives_augmentation,
        slide_frames=1,  # Сбалансированное количество примеров
        step_ms=4.42,  # Унифицированный стандарт microWakeWord
    )
    
    print("📊 Генерация спектрограмм для обучения...")
    print("📊 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   - Всего позитивных файлов: {pos_files}")
    print(f"   - Split count: 0.1 (10% на валидацию/тест)")
    print(f"   - Ожидаемое обучение: ~{int(pos_files * 0.9)} файлов")
    print(f"   - Ожидаемая валидация: ~{int(pos_files * 0.05)} файлов")
    print(f"   - Ожидаемое тестирование: ~{int(pos_files * 0.05)} файлов")
    
    # Генерируем позитивные спектрограммы - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_POSITIVES, "training", "wakeword_mmap"),
        sample_generator=positives_spectrograms.spectrogram_generator(split="train", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    print("📊 Генерация спектрограмм для валидации...")
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_POSITIVES, "validation", "wakeword_mmap"),
        sample_generator=positives_spectrograms.spectrogram_generator(split="validation", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    print("📊 Генерация спектрограмм для тестирования...")
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_POSITIVES, "testing", "wakeword_mmap"),
        sample_generator=positives_spectrograms.spectrogram_generator(split="test", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    # Настройка негативных данных
    print("📊 НАСТРОЙКА НЕГАТИВНЫХ ДАННЫХ:")
    print(f"   - Источник: {negatives_dir}")
    print(f"   - Slide frames: 1 (сбалансированное количество примеров)")
    
    negatives_clips = Clips(
        input_directory=negatives_dir,
        file_pattern='*.wav',
        max_clip_duration_s=1.5,  # Унифицированная длительность с семплированием
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    
    # Унифицированные аугментации для негативных данных
    negatives_augmentation = Augmentation(
        augmentation_duration_s=1.5,  # Соответствует max_clip_duration_s
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
        slide_frames=1,  # Сбалансированное количество примеров
        step_ms=4.42,  # Унифицированный стандарт microWakeWord
    )
    
    print("📊 Генерация негативных спектрограмм...")
    print("📊 Генерируем негативные данные...")
    print(f"📊 Обрабатываем {neg_files} файлов × 1 = {neg_files} итераций")
    
    # Создаем директории для негативных данных - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "training"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "validation"), exist_ok=True)
    os.makedirs(os.path.join(paths.FEATURES_NEGATIVES, "testing"), exist_ok=True)
    
    # Генерируем негативные спектрограммы - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_NEGATIVES, "training", "wakeword_mmap"),
        sample_generator=negatives_spectrograms.spectrogram_generator(split="train", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_NEGATIVES, "validation", "wakeword_mmap"),
        sample_generator=negatives_spectrograms.spectrogram_generator(split="validation", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    RaggedMmap.from_generator(
        out_dir=os.path.join(paths.FEATURES_NEGATIVES, "testing", "wakeword_mmap"),
        sample_generator=negatives_spectrograms.spectrogram_generator(split="test", repeat=1),
        batch_size=1000,
        verbose=True,
    )
    
    # Генерация ambient данных
    print("📊 ГЕНЕРАЦИЯ AMBIENT ДАННЫХ:")
    print("📊 Генерируем ambient данные для обучения...")
    
    # Проверяем наличие ambient данных
    ambient_dir = paths.BACKGROUND  # Используем готовые фоновые фрагменты
    if os.path.exists(ambient_dir):
        ambient_files = len([f for f in os.listdir(ambient_dir) if f.endswith('.wav')])
        print(f"📊 Генерируем ambient данные...")
        print(f"   - Найдено {ambient_files} ambient файлов")
        
        # Создаем clips для ambient данных
        ambient_clips = Clips(
            input_directory=ambient_dir,
            file_pattern='*.wav',
            max_clip_duration_s=None,
            remove_silence=False,
            random_split_seed=10,
            split_count=0.1,
        )
        
        ambient_spectrograms = SpectrogramGeneration(
            clips=ambient_clips,
            augmenter=None,  # Без аугментаций для ambient данных
            slide_frames=1,
            step_ms=10,
        )
        
        # Создаем директории для ambient данных
        os.makedirs(os.path.join(data_dir, "generated_features_ambient", "training_ambient"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "generated_features_ambient", "testing_ambient"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "generated_features_ambient", "validation_ambient"), exist_ok=True)
        
        # Генерируем ambient данные для обучения
        RaggedMmap.from_generator(
            out_dir=os.path.join(data_dir, "generated_features_ambient", "training_ambient", "wakeword_mmap"),
            sample_generator=ambient_spectrograms.spectrogram_generator(split="train", repeat=1),
            batch_size=50,
            verbose=True,
        )
        
        # Генерируем ambient данные для тестирования
        RaggedMmap.from_generator(
            out_dir=os.path.join(data_dir, "generated_features_ambient", "testing_ambient", "wakeword_mmap"),
            sample_generator=ambient_spectrograms.spectrogram_generator(split="test", repeat=1),
            batch_size=50,
            verbose=True,
        )
        
        # Генерируем ambient данные для валидации
        RaggedMmap.from_generator(
            out_dir=os.path.join(data_dir, "generated_features_ambient", "validation_ambient", "wakeword_mmap"),
            sample_generator=ambient_spectrograms.spectrogram_generator(split="validation", repeat=1),
            batch_size=50,
            verbose=True,
        )
        
        print(f"✅ Ambient данные сгенерированы успешно")
    else:
        print(f"⚠️ Директория ambient данных не найдена: {ambient_dir}")
        print("   Продолжаем без ambient данных...")
    
    print("✅ Генерация спектрограмм завершена успешно!")
    return True

def main():
    """Основная функция - только генерация спектрограмм"""
    
    print("🎯 Генерация спектрограмм для microWakeWord")
    
    # Базовая директория для данных - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    data_dir = paths.DATA_ROOT
    
    # Проверяем наличие данных - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_PROCESSED  # Используем аугментированные негативные данные
    
    print(f"\n📊 ПРОВЕРКА ВХОДНЫХ ДАННЫХ:")
    
    if os.path.exists(positives_dir):
        pos_files = len([f for f in os.listdir(positives_dir) if f.endswith('.wav')])
        print(f"   ✅ Позитивные данные: {pos_files} файлов в {positives_dir}")
    else:
        print(f"   ❌ Позитивные данные не найдены: {positives_dir}")
        return False
    
    if os.path.exists(negatives_dir):
        neg_files = len([f for f in os.listdir(negatives_dir) if f.endswith('.wav')])
        print(f"   ✅ Негативные данные: {neg_files} файлов в {negatives_dir}")
    else:
        print(f"   ❌ Негативные данные не найдены: {negatives_dir}")
        return False
    
    print(f"   📊 Соотношение негативных к позитивным: {neg_files/pos_files:.1f}:1")
    
    try:
        # Создаем данные в их формате
        if create_data_in_their_format():
            print("\n🎉 Генерация спектрограмм завершена успешно!")
            print("📁 Спектрограммы готовы для обучения:")
            print(f"  - {paths.FEATURES_POSITIVES}/")
            print(f"  - {paths.FEATURES_NEGATIVES}/")
            print(f"  - {paths.DATA_ROOT}/features_background/")
            print("\n🚀 Теперь можно запустить обучение модели!")
            return True
        else:
            print("\n❌ Генерация спектрограмм завершилась с ошибками")
            return False
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()