#!/usr/bin/env python3
"""
Генерация TTS негативных данных (фонетически похожие фразы)
Переименовано из generate_hard_negatives_parallel.py для ясности
"""

import os
import subprocess
import multiprocessing
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
from src.utils.path_manager import paths

def run_command(cmd):
    """Выполнить команду и вернуть результат"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Ошибка команды: {cmd}")
            print(f"STDERR: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

def generate_phrase(args):
    """Генерировать одну фразу"""
    phrase, output_file, voices = args
    
    # Выбираем случайный голос
    voice = random.choice(voices)
    
    tts_cmd = f"""
    echo "{phrase}" | piper --model {paths.DATA_ROOT}/piper_models/{voice}.onnx --output_file {output_file}
    """
    
    if run_command(tts_cmd):
        return True
    return False

def generate_negatives_tts():
    """Генерировать TTS негативные примеры параллельно"""
    
    # Создаем директории - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    output_dir = paths.NEGATIVES_TTS
    os.makedirs(output_dir, exist_ok=True)
    
    # TTS негативные фразы (фонетически похожие на "милый дом" и "любимый дом")
    negatives_phrases = [
        # Фонетически похожие на "милый дом"
        "мой том", "мой дон", "мой домик", "мой дом там", "мой домофон",
        "мой дом-то", "мой домок", "мой домище", "мой домок там", "мой домочек",
        "милый том", "милый дон", "милый домик", "милый дом там", "милый домофон",
        "милый дом-то", "милый домок", "милый домище", "милый домок там", "милый домочек",
        
        # Фонетически похожие на "любимый дом"
        "любимый том", "любимый дон", "любимый домик", "любимый дом там", "любимый домофон",
        "любимый дом-то", "любимый домок", "любимый домище", "любимый домок там", "любимый домочек",
        
        # Английские похожие
        "my home", "my dome", "my house", "my room", "my dome", "my home",
        "my tom", "my don", "my dom", "my home there", "my home phone",
        
        # Вставки до/после
        "ну милый дом там", "вот милый домик", "это милый дом", "да милый дом",
        "очень любимый дом", "самый любимый дом", "это любимый дом", "вот любимый дом",
        
        # Дополнительные вариации
        "милый дом да", "милый дом же", "милый дом то", "милый дом здесь", "милый дом везде",
        "любимый дом да", "любимый дом же", "любимый дом то", "любимый дом здесь",
        
        # Еще больше похожих
        "милый домок", "милый домок", "любимый домок", "любимый домок",
        "милый дом там", "милый дом здесь", "любимый дом там", "любимый дом здесь",
        
        # Дополнительные вариации для разнообразия
        "мой милый дом", "мой любимый дом", "твой милый дом", "твой любимый дом",
        "его милый дом", "её милый дом", "наш милый дом", "наш любимый дом",
        "дом милый", "дом любимый", "дом красивый", "дом большой",
        "дом там", "дом здесь", "дом везде", "дом всегда",
        
        # Еще больше вариаций
        "милый домок", "любимый домок", "милый домик", "любимый домик",
        "милый домочек", "любимый домочек", "милый домище", "любимый домище",
        "милый домофон", "любимый домофон", "милый дом-то", "любимый дом-то",
        
        # Контекстные фразы
        "вот милый дом", "вот любимый дом", "это милый дом", "это любимый дом",
        "да милый дом", "да любимый дом", "ну милый дом", "ну любимый дом",
        "а милый дом", "а любимый дом", "но милый дом", "но любимый дом",
        
        # Вопросительные
        "милый дом?", "любимый дом?", "где милый дом?", "где любимый дом?",
        "что милый дом?", "что любимый дом?", "какой милый дом?", "какой любимый дом?",
        
        # Восклицательные
        "милый дом!", "любимый дом!", "какой милый дом!", "какой любимый дом!",
        "вот милый дом!", "вот любимый дом!", "это милый дом!", "это любимый дом!",
        
        # Дополнительные для полного покрытия
        "милый дом да", "любимый дом да", "милый дом же", "любимый дом же",
        "милый дом то", "любимый дом то", "милый дом здесь", "любимый дом здесь",
        "милый дом везде", "любимый дом везде", "милый дом всегда", "любимый дом всегда",
        
        # Еще больше для разнообразия
        "милый домок", "любимый домок", "милый домик", "любимый домик",
        "милый домочек", "любимый домочек", "милый домище", "любимый домище",
        "милый домофон", "любимый домофон", "милый дом-то", "любимый дом-то",
        
        # Финальные вариации
        "милый дом там", "любимый дом там", "милый дом здесь", "любимый дом здесь",
        "милый дом везде", "любимый дом везде", "милый дом всегда", "любимый дом всегда",
        "милый дом да", "любимый дом да", "милый дом же", "любимый дом же",
        "милый дом то", "любимый дом то", "милый дом здесь", "любимый дом здесь"
    ]
    
    # Доступные голоса Piper
    voices = [
        "ru_RU-denis-medium", "ru_RU-dmitri-medium", 
        "ru_RU-irina-medium", "ru_RU-ruslan-medium"
    ]
    
    print(f"🚀 Генерация TTS негативных данных...")
    print(f"📊 Фраз для генерации: {len(negatives_phrases)}")
    print(f"🎤 Голоса: {voices}")
    print(f"📁 Выходная директория: {output_dir}")
    
    # Подготавливаем задачи
    tasks = []
    file_counter = 0
    
    for phrase in negatives_phrases:
        # Генерируем несколько вариантов каждой фразы
        for i in range(20):  # 20 вариантов на фразу
            output_file = os.path.join(output_dir, f"neg_{file_counter:04d}.wav")
            tasks.append((phrase, output_file, voices))
            file_counter += 1
    
    print(f"📊 Всего файлов для генерации: {len(tasks)}")
    
    # Используем все CPU ядра
    num_cores = multiprocessing.cpu_count()
    print(f"💻 Используем {num_cores} ядер для генерации")
    
    # Генерируем файлы параллельно
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Отправляем все задачи
        future_to_task = {executor.submit(generate_phrase, task): task for task in tasks}
        
        # Обрабатываем результаты с прогресс-баром
        with tqdm(total=len(tasks), desc="Генерация TTS негативных данных") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                except Exception as e:
                    print(f"❌ Ошибка генерации {task[1]}: {e}")
                pbar.update(1)
    
    print(f"✅ Генерация завершена!")
    print(f"📊 Успешно сгенерировано: {success_count}/{len(tasks)} файлов")
    print(f"📁 Результат сохранен в: {output_dir}")
    
    return output_dir

def main():
    """Основная функция"""
    print("🚀 ГЕНЕРАЦИЯ TTS НЕГАТИВНЫХ ДАННЫХ")
    print("=" * 60)
    
    # Проверяем наличие Piper моделей - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
    piper_models_dir = f"{paths.DATA_ROOT}/piper_models"
    if not os.path.exists(piper_models_dir):
        print(f"❌ Директория с Piper моделями не найдена: {piper_models_dir}")
        print("💡 Убедитесь что Piper TTS установлен и модели загружены")
        return False
    
    # Проверяем наличие моделей
    model_files = [f for f in os.listdir(piper_models_dir) if f.endswith('.onnx')]
    if not model_files:
        print(f"❌ Piper модели не найдены в {piper_models_dir}")
        print("💡 Загрузите русские модели Piper TTS")
        return False
    
    print(f"✅ Найдено {len(model_files)} Piper моделей")
    
    # Генерируем данные
    output_dir = generate_negatives_tts()
    
    # Проверяем результат
    if os.path.exists(output_dir):
        files_count = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
        print(f"🎉 Генерация завершена успешно!")
        print(f"📊 Создано файлов: {files_count}")
        print(f"📁 Директория: {output_dir}")
        return True
    else:
        print(f"❌ Ошибка: директория {output_dir} не создана")
        return False

if __name__ == "__main__":
    main()