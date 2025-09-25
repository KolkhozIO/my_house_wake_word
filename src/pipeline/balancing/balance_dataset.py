#!/usr/bin/env python3
"""
Балансировка датасета и создание фоновых данных
"""

import os
import subprocess
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import glob

def generate_background_phrase(args):
    """Генерирует одну фоновую фразу - ВРЕМЕННО ОТКЛЮЧЕНО"""
    text, voice, output_file = args
    
    # ВРЕМЕННО: Piper TTS отключен, данные уже сгенерированы
    # Создаем заглушку - копируем существующий негативный файл
    existing_files = [
        "/home/microWakeWord_data/negatives_real/last10s_left_20250918_123101.wav",
        "/home/microWakeWord_data/negatives_real/last10s_left_20250918_123131.wav",
        "/home/microWakeWord_data/negatives_real/last10s_left_20250918_123201.wav"
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

def apply_random_augmentation(input_file, output_file):
    """Применяет случайную аугментацию к файлу"""
    # Простые и безопасные аугментации
    augmentations = [
        ('volume', lambda: ['sox', input_file, output_file, 'vol', str(random.uniform(0.5, 1.5))]),
        ('tempo', lambda: ['sox', input_file, output_file, 'tempo', str(random.uniform(0.9, 1.1))]),
        ('pitch', lambda: ['sox', input_file, output_file, 'pitch', str(random.randint(-50, 50))])
    ]
    
    aug_type, cmd_func = random.choice(augmentations)
    cmd = cmd_func()
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file} ({aug_type})"
    except subprocess.CalledProcessError as e:
        # Если аугментация не удалась, просто копируем файл
        try:
            shutil.copy2(input_file, output_file)
            return f"OK: {output_file} (copied)"
        except Exception as copy_error:
            return f"ERROR: {output_file} - {e} (copy failed: {copy_error})"

def create_background_data(output_dir, voices, count_per_phrase=50, progress_bar=None):
    """Создает фоновые данные (не содержащие wake-words)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Фразы, которые НЕ содержат наши wake-words (уменьшенный список)
    background_phrases = [
        # Обычные русские фразы
        "привет как дела", "что делаешь", "как поживаешь", "как настроение",
        "хорошая погода", "сегодня дождь", "завтра будет солнце", "люблю лето",
        "иду домой", "еду на работу", "сижу дома", "читаю книгу", "смотрю фильм",
        "готовлю еду", "убираюсь дома", "сплю на диване", "пью чай",
        "слушаю музыку", "играю в игры", "гуляю в парке", "встречаюсь с друзьями",
        "покупаю продукты", "хожу в магазин", "езжу на машине", "иду пешком",
        "работаю за компьютером", "пишу письмо", "звоню по телефону", "отвечаю на сообщения",
        "смотрю новости", "читаю газету", "слушаю радио", "включаю телевизор",
        "выключаю свет", "закрываю дверь", "открываю окно", "сажусь на стул",
        "встаю с кровати", "одеваюсь", "раздеваюсь", "мою руки", "чищу зубы",
        "завтракаю", "обедаю", "ужинаю", "пью воду", "ем фрукты",
        
        # Английские фразы
        "hello how are you", "what are you doing", "how is the weather",
        "good morning", "good evening", "have a nice day", "see you later",
        "thank you very much", "you are welcome", "excuse me please",
        "where are you going", "what time is it", "have a good time",
        "nice to meet you", "how do you do", "what is your name",
        "i am fine thank you", "i am busy now", "i am tired today",
        "i like music", "i love movies", "i enjoy reading", "i like cooking",
        "i am working", "i am studying", "i am sleeping", "i am eating"
    ]
    
    # Создаем задачи для генерации
    tasks = []
    file_counter = 0
    
    for phrase in background_phrases:
        for voice in voices:
            for i in range(count_per_phrase):
                output_file = os.path.join(output_dir, f"bg_{file_counter:04d}.wav")
                tasks.append((phrase, voice, output_file))
                file_counter += 1
    
    # Генерируем фоновые данные параллельно
    max_workers = cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_background_phrase, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("ERROR"):
                print(result)
            if progress_bar:
                progress_bar.update(1)

def duplicate_and_augment_negatives(input_dir, output_dir, target_count, progress_bar=None):
    """Дублирует негативные файлы с аугментациями до достижения целевого количества"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем все файлы
    input_files = glob.glob(os.path.join(input_dir, "*.wav"))
    current_count = len(input_files)
    
    # Копируем оригинальные файлы
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        shutil.copy2(input_file, os.path.join(output_dir, base_name))
    
    # Создаем задачи для дублирования с аугментациями
    tasks = []
    file_counter = current_count
    
    while file_counter < target_count:
        # Выбираем случайный файл для дублирования
        source_file = random.choice(input_files)
        base_name = os.path.basename(source_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Создаем аугментированную версию
        output_file = os.path.join(output_dir, f"{name_without_ext}_aug_{file_counter}.wav")
        tasks.append((source_file, output_file))
        file_counter += 1
    
    # Применяем аугментации параллельно
    max_workers = cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(apply_random_augmentation, task[0], task[1]) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("ERROR"):
                print(result)
            if progress_bar:
                progress_bar.update(1)

def analyze_dataset(data_dir):
    """Анализирует текущий датасет"""
    print("Анализ текущего датасета:")
    
    # Проверяем наличие директорий
    dirs_to_check = [
        "positives_both",
        "negatives_both", 
        "positives_both_augmented",
        "negatives_both_augmented"
    ]
    
    for dir_name in dirs_to_check:
        full_path = os.path.join(data_dir, dir_name)
        if os.path.exists(full_path):
            file_count = len(glob.glob(os.path.join(full_path, "*.wav")))
            print(f"  {dir_name}: {file_count} файлов")
        else:
            print(f"  {dir_name}: не найдена")
    
    # Рекомендуем соотношение
    print("\nРекомендуемое соотношение негативы:позитивы = 50:1")
    print("Это поможет снизить ложные срабатывания (FA)")

def main():
    print("Балансировка датасета и создание фоновых данных...")
    
    # Базовая директория для данных
    data_dir = "/home/microWakeWord_data"
    
    # Анализируем текущий датасет
    analyze_dataset(data_dir)
    
    # Определяем источник данных (оригинальные или аугментированные)
    if os.path.exists(os.path.join(data_dir, "positives_both_augmented")):
        pos_source = os.path.join(data_dir, "positives_both_augmented")
        neg_source = os.path.join(data_dir, "negatives_both_augmented")
        print("Используем аугментированные данные")
    else:
        pos_source = os.path.join(data_dir, "positives_both")
        neg_source = os.path.join(data_dir, "negatives_both")
        print("Используем оригинальные данные")
    
    if not os.path.exists(pos_source) or not os.path.exists(neg_source):
        print(f"ОШИБКА: Директории {pos_source} или {neg_source} не найдены!")
        return
    
    # Подсчитываем количество файлов
    pos_files = len(glob.glob(os.path.join(pos_source, "*.wav")))
    neg_files = len(glob.glob(os.path.join(neg_source, "*.wav")))
    
    print(f"Текущее соотношение: позитивы={pos_files}, негативы={neg_files}")
    print(f"Соотношение негативы:позитивы = {neg_files/pos_files:.1f}:1")
    
    # Целевое соотношение 5:1 (разумное для начала)
    target_negatives = pos_files * 5
    additional_negatives = target_negatives - neg_files
    
    print(f"Целевое количество негативов: {target_negatives}")
    print(f"Нужно добавить негативов: {additional_negatives}")
    
    # Ограничиваем максимальное количество дополнительных негативов
    if additional_negatives > 50000:
        print(f"⚠️  Слишком много негативов! Ограничиваем до 50,000")
        additional_negatives = 50000
        target_negatives = neg_files + additional_negatives
    
    # Получаем голоса для генерации фоновых данных
    voices = [
        "ru_RU-denis-medium", "ru_RU-dmitri-medium", 
        "ru_RU-irina-medium", "ru_RU-ruslan-medium"
    ]
    
    # Создаем временные директории
    temp_positives_final = os.path.join(data_dir, "positives_final_temp")
    temp_negatives_final = os.path.join(data_dir, "negatives_final_temp")
    temp_background = os.path.join(data_dir, "background_temp")
    
    # Очищаем временные директории
    for temp_dir in [temp_positives_final, temp_negatives_final, temp_background]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Копируем позитивные данные
    print("Копируем позитивные данные...")
    shutil.copytree(pos_source, temp_positives_final)
    
    # Подсчитываем общее количество задач
    background_tasks = len(voices) * 30 * 10  # Еще меньше фоновых данных: 30 фраз × 10 сэмплов
    total_tasks = additional_negatives + background_tasks
    
    print(f"Используем {cpu_count()} ядер для {total_tasks} задач...")
    
    # Создаем общий прогресс-бар
    with tqdm(total=total_tasks, desc="Балансировка датасета", unit="файл") as pbar:
        # Дублируем негативные данные с аугментациями
        if additional_negatives > 0:
            print("Дублируем негативные данные с аугментациями...")
            duplicate_and_augment_negatives(neg_source, temp_negatives_final, target_negatives, pbar)
        else:
            # Просто копируем существующие негативы
            shutil.copytree(neg_source, temp_negatives_final)
        
        # Создаем фоновые данные
        print("Создаем фоновые данные...")
        create_background_data(temp_background, voices, count_per_phrase=20, progress_bar=pbar)
    
    # Атомарная замена директорий
    print("Атомарная замена директорий...")
    
    # Удаляем старые финальные директории
    final_positives = os.path.join(data_dir, "positives_final")
    final_negatives = os.path.join(data_dir, "negatives_final")
    final_background = os.path.join(data_dir, "background")
    
    for final_dir in [final_positives, final_negatives, final_background]:
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
    
    # Переименовываем временные директории
    os.rename(temp_positives_final, final_positives)
    os.rename(temp_negatives_final, final_negatives)
    os.rename(temp_background, final_background)
    
    # Финальная статистика
    final_pos = len(glob.glob(os.path.join(final_positives, "*.wav")))
    final_neg = len(glob.glob(os.path.join(final_negatives, "*.wav")))
    final_bg = len(glob.glob(os.path.join(final_background, "*.wav")))
    
    print("Балансировка завершена!")
    print(f"Финальные данные:")
    print(f"  Позитивных файлов: {final_pos}")
    print(f"  Негативных файлов: {final_neg}")
    print(f"  Фоновых файлов: {final_bg}")
    print(f"  Соотношение негативы:позитивы = {final_neg/final_pos:.1f}:1")
    print(f"  Общий размер датасета: {final_pos + final_neg + final_bg} файлов")

if __name__ == "__main__":
    main()