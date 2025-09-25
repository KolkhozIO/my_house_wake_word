#!/usr/bin/env python3
"""
Быстрая генерация тестовых данных без Piper TTS
Создает простые аудиофайлы для тестирования пайплайна
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

def generate_simple_audio(text, sample_rate=16000, duration=0.5):
    """Генерирует простое аудио с синусоидальным тоном"""
    # Создаем синусоидальный тон с частотой, зависящей от текста
    freq = 440 + hash(text) % 200  # Частота от 440 до 640 Гц
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Создаем основной тон
    audio = np.sin(2 * np.pi * freq * t)
    
    # Добавляем небольшой шум для реалистичности
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Нормализуем
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio

def generate_file(args):
    """Генерирует один аудиофайл"""
    text, output_path, sample_rate = args
    
    try:
        audio = generate_simple_audio(text, sample_rate)
        sf.write(output_path, audio, sample_rate)
        return True
    except Exception as e:
        print(f"Ошибка генерации {output_path}: {e}")
        return False

def main():
    print("🚀 Быстрая генерация тестовых данных...")
    
    # Настройки
    data_dir = "/home/microWakeWord_data"
    sample_rate = 16000
    
    # Создаем директории
    temp_positives = os.path.join(data_dir, "positives_both_temp")
    temp_negatives = os.path.join(data_dir, "negatives_both_temp")
    final_positives = os.path.join(data_dir, "positives_both")
    final_negatives = os.path.join(data_dir, "negatives_both")
    
    for dir_path in [temp_positives, temp_negatives, final_positives, final_negatives]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Позитивные фразы
    positive_phrases = [
        "милый дом", "милый дом.", "милый дом!", "милый дом?",
        "любимый дом", "любимый дом.", "любимый дом!", "любимый дом?",
        "ну милый дом", "вот милый дом", "да милый дом", "это милый дом",
        "ну любимый дом", "вот любимый дом", "да любимый дом", "это любимый дом"
    ]
    
    # Негативные фразы
    negative_phrases = [
        "милый том", "милый дон", "милый домик", "милый домофон",
        "любимый том", "любимый дон", "любимый домик", "любимый домофон",
        "мой дом", "мой дом.", "мой дом!", "мой дом?",
        "твой дом", "его дом", "её дом", "наш дом",
        "дом милый", "дом любимый", "дом красивый", "дом большой",
        "my home", "my dome", "my house", "my room",
        "hello world", "good morning", "good evening", "good night",
        "привет мир", "доброе утро", "добрый вечер", "спокойной ночи",
        "как дела", "что делаешь", "где ты", "кто ты",
        "время", "погода", "новости", "музыка"
    ]
    
    # Генерируем позитивные файлы
    print("📝 Генерация позитивных файлов...")
    positive_tasks = []
    for i in range(3200):  # 3200 позитивных файлов
        phrase = positive_phrases[i % len(positive_phrases)]
        filename = f"{i:03d}_{i % 20}.wav"
        output_path = os.path.join(temp_positives, filename)
        positive_tasks.append((phrase, output_path, sample_rate))
    
    # Генерируем негативные файлы
    print("📝 Генерация негативных файлов...")
    negative_tasks = []
    for i in range(690):  # 690 негативных файлов
        phrase = negative_phrases[i % len(negative_phrases)]
        filename = f"{i:03d}_{i % 20}.wav"
        output_path = os.path.join(temp_negatives, filename)
        negative_tasks.append((phrase, output_path, sample_rate))
    
    # Используем все ядра
    num_cores = mp.cpu_count()
    print(f"Используем {num_cores} ядер для генерации {len(positive_tasks + negative_tasks)} файлов...")
    
    # Генерируем все файлы
    all_tasks = positive_tasks + negative_tasks
    
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(
            pool.imap(generate_file, all_tasks),
            total=len(all_tasks),
            desc="Генерация аудио"
        ))
    
    success_count = sum(results)
    print(f"✅ Успешно сгенерировано {success_count}/{len(all_tasks)} файлов")
    
    # Атомарная замена директорий
    print("🔄 Атомарная замена директорий...")
    
    # Удаляем старые директории
    if os.path.exists(final_positives):
        import shutil
        shutil.rmtree(final_positives, ignore_errors=True)
    if os.path.exists(final_negatives):
        import shutil
        shutil.rmtree(final_negatives, ignore_errors=True)
    
    # Перемещаем временные директории
    os.rename(temp_positives, final_positives)
    os.rename(temp_negatives, final_negatives)
    
    # Подсчитываем финальные файлы
    pos_count = len([f for f in os.listdir(final_positives) if f.endswith('.wav')])
    neg_count = len([f for f in os.listdir(final_negatives) if f.endswith('.wav')])
    
    print(f"🎉 Генерация завершена!")
    print(f"📊 Позитивных файлов: {pos_count}")
    print(f"📊 Негативных файлов: {neg_count}")
    print(f"📊 Общий размер датасета: {pos_count + neg_count}")

if __name__ == "__main__":
    main()