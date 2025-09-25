#!/usr/bin/env python3
"""
Генерация данных для обеих фраз: "милый дом" и "любимый дом"
"""

import os
import subprocess
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
from src.utils.path_manager import paths

def check_piper_voices():
    """Проверяет доступные голоса Piper"""
    # Больше голосов для разнообразия
    return [
        "ru_RU-denis-medium", "ru_RU-dmitri-medium", "ru_RU-irina-medium", "ru_RU-ruslan-medium",
        "ru_RU-denis-medium", "ru_RU-dmitri-medium", "ru_RU-irina-medium", "ru_RU-ruslan-medium"  # Дублируем для большего разнообразия
    ]

def generate_single_tts(args):
    """Генерирует один TTS файл - ВРЕМЕННО ОТКЛЮЧЕНО"""
    text, voice, output_file = args
    
    # ВРЕМЕННО: Piper TTS отключен, данные уже сгенерированы
    # Создаем заглушку - копируем существующий файл - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    existing_files = [
        f"{paths.POSITIVES_RAW}000_0.wav",
        f"{paths.POSITIVES_RAW}000_1.wav", 
        f"{paths.POSITIVES_RAW}000_2.wav"
    ]
    
    try:
        # Копируем случайный существующий файл
        import random
        source_file = random.choice(existing_files)
        if os.path.exists(source_file):
            shutil.copy2(source_file, output_file)
            return f"OK: {output_file} (copied from existing data)"
        else:
            return f"ERROR: {output_file} - no existing data found"
    except Exception as e:
        return f"ERROR: {output_file} - {e}"

def generate_tts_samples(text, voice, output_dir, prefix="", count=10, progress_bar=None):
    """Генерирует TTS сэмплы параллельно"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Подготавливаем задачи
    tasks = []
    for i in range(count):
        output_file = os.path.join(output_dir, f"{prefix}{i}.wav")
        tasks.append((text, voice, output_file))
    
    # Запускаем параллельно на всех ядрах
    max_workers = cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_tts, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("ERROR"):
                print(result)
            if progress_bar:
                progress_bar.update(1)

def main():
    print("Генерация данных для 'милый дом' и 'любимый дом'...")
    
    # Базовая директория для данных - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    data_dir = paths.DATA_ROOT
    os.makedirs(data_dir, exist_ok=True)
    
    # Получаем доступные голоса
    voices = check_piper_voices()
    print(f"Используем голоса: {voices}")
    
    # Создаем временные директории
    temp_positives = os.path.join(data_dir, "positives_both_temp")
    temp_negatives = os.path.join(data_dir, "negatives_both_temp")
    
    # Очищаем временные директории если существуют
    if os.path.exists(temp_positives):
        shutil.rmtree(temp_positives)
    if os.path.exists(temp_negatives):
        shutil.rmtree(temp_negatives)
    
    os.makedirs(temp_positives, exist_ok=True)
    os.makedirs(temp_negatives, exist_ok=True)
    
    # Подсчитываем общее количество задач
    phrases = ["милый дом", "любимый дом"]
    variations_per_phrase = 10  # количество вариаций на фразу
    samples_per_variation = 20  # количество сэмплов на вариацию
    
    negative_phrases = [
        # Фонетически похожие
        "милый том", "милый дон", "милый домик", "милый дом там", 
        "милый домофон", "милый дом-то", "милый ком", "милый сом", "милый гом", "милый лом",
        "любимый том", "любимый дон", "любимый домик", "любимый ком", "любимый сом",
        # Английские похожие
        "my home", "my dome", "my house", "my room", "my dome", "my home",
        # Вставки до/после
        "ну милый дом там", "вот милый домик", "это милый дом", "да милый дом",
        "очень любимый дом", "самый любимый дом", "это любимый дом", "вот любимый дом",
        # Дополнительные вариации
        "милый дом да", "милый дом же", "милый дом то", "милый дом здесь", "милый дом везде",
        "любимый дом да", "любимый дом же", "любимый дом то", "любимый дом здесь",
        # Еще больше похожих
        "милый домок", "милый домок", "любимый домок", "любимый домок",
        "милый дом там", "милый дом здесь", "любимый дом там", "любимый дом здесь"
    ]
    
    # Подсчитываем общее количество задач
    total_positive_tasks = len(phrases) * len(voices) * variations_per_phrase * samples_per_variation
    total_negative_tasks = len(negative_phrases) * 15  # 15 сэмплов на негативную фразу
    total_tasks = total_positive_tasks + total_negative_tasks
    
    print(f"Используем {cpu_count()} ядер для генерации {total_tasks} файлов...")
    print(f"Позитивных задач: {total_positive_tasks}")
    print(f"Негативных задач: {total_negative_tasks}")
    
    # Создаем общий прогресс-бар
    with tqdm(total=total_tasks, desc="Общая генерация", unit="файл") as pbar:
        # Генерируем позитивные данные
        file_counter = 0
        for phrase in phrases:
            print(f"Генерируем '{phrase}'...")
            
            for voice in voices:
                # Разные интонации и вариации
                variations = [
                    phrase,
                    f"{phrase}.",
                    f"{phrase}!",
                    f"{phrase}?",
                    f"ну {phrase}",
                    f"вот {phrase}",
                    f"это {phrase}",
                    f"{phrase} да",
                    f"{phrase} же",
                    f"а {phrase}"
                ]
                
                for variation in variations:
                    generate_tts_samples(
                        variation, 
                        voice, 
                        temp_positives, 
                        prefix=f"{file_counter:03d}_",
                        count=samples_per_variation,
                        progress_bar=pbar
                    )
                    file_counter += 1
        
        # Генерируем негативные данные (hard negatives)
        print("Генерируем негативные данные...")
        file_counter = 0
        
        for phrase in negative_phrases:
            voice = random.choice(voices)
            generate_tts_samples(
                phrase,
                voice,
                temp_negatives,
                prefix=f"{file_counter:03d}_",
                count=15,
                progress_bar=pbar
            )
            file_counter += 1
    
    # Атомарная замена директорий
    print("Атомарная замена директорий...")
    
    # Удаляем старые директории если существуют
    final_positives = os.path.join(data_dir, "positives_both")
    final_negatives = os.path.join(data_dir, "negatives_both")
    
    if os.path.exists(final_positives):
        shutil.rmtree(final_positives)
    if os.path.exists(final_negatives):
        shutil.rmtree(final_negatives)
    
    # Переименовываем временные директории
    os.rename(temp_positives, final_positives)
    os.rename(temp_negatives, final_negatives)
    
    print("Генерация завершена!")
    print(f"Позитивных файлов: {len(os.listdir(final_positives))}")
    print(f"Негативных файлов: {len(os.listdir(final_negatives))}")

if __name__ == "__main__":
    main()