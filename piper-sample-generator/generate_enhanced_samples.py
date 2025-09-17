#!/usr/bin/env python3
"""
Улучшенный генератор сэмплов для wake-word "мой дом"
Генерирует больше сэмплов с разными голосами и параметрами
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    print(f"Выполняем: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        return False
    print(f"Успешно: {result.stdout}")
    return True

def generate_positive_samples():
    """Генерирует положительные сэмплы с разными голосами"""
    
    voices = [
        "voices/ru_RU-denis-medium.onnx",
        "voices/ru_RU-dmitri-medium.onnx"
    ]
    
    # Основная фраза
    target_phrase = "мой дом"
    
    # Создаем папку для положительных сэмплов
    output_dir = "positives_moy_dom_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = 0
    
    for voice in voices:
        if not os.path.exists(voice):
            print(f"Пропускаем {voice} - файл не найден")
            continue
            
        print(f"\nГенерируем сэмплы с голосом: {voice}")
        
        # Генерируем сэмплы с разными параметрами
        cmd = f"""python3 generate_samples.py "{target_phrase}" \
            --model {voice} \
            --length-scales 0.8 0.9 1.0 1.1 1.2 \
            --noise-scales 0.0 0.1 0.2 \
            --noise-scale-ws 0.0 0.1 0.2 \
            --max-samples 200 \
            --output-dir {output_dir}"""
        
        if run_command(cmd):
            # Подсчитываем количество сгенерированных файлов
            files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            total_samples += len(files)
            print(f"Сгенерировано {len(files)} сэмплов с голосом {voice}")
    
    print(f"\nВсего положительных сэмплов: {total_samples}")
    return output_dir

def generate_negative_samples():
    """Генерирует негативные сэмплы с похожими фразами"""
    
    voices = [
        "voices/ru_RU-denis-medium.onnx",
        "voices/ru_RU-dmitri-medium.onnx"
    ]
    
    # Похожие фразы, которые могут вызвать ложные срабатывания
    negative_phrases = [
        "мой том",
        "мой домик", 
        "мой домовой",
        "мой дом там",
        "мой дом здесь",
        "мой дом большой",
        "мой дом маленький",
        "мой дом новый",
        "мой дом старый",
        "мой дом красивый",
        "мой дом уютный",
        "мой дом теплый",
        "мой дом холодный",
        "мой дом светлый",
        "мой дом темный"
    ]
    
    # Создаем папку для негативных сэмплов
    output_dir = "negatives_moy_dom_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = 0
    
    for voice in voices:
        if not os.path.exists(voice):
            print(f"Пропускаем {voice} - файл не найден")
            continue
            
        print(f"\nГенерируем негативные сэмплы с голосом: {voice}")
        
        for phrase in negative_phrases:
            print(f"  Генерируем: {phrase}")
            
            cmd = f"""python3 generate_samples.py "{phrase}" \
                --model {voice} \
                --length-scales 0.8 0.9 1.0 1.1 1.2 \
                --max-samples 10 \
                --output-dir {output_dir}"""
            
            if run_command(cmd):
                # Подсчитываем количество сгенерированных файлов
                files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
                total_samples = len(files)
    
    print(f"\nВсего негативных сэмплов: {total_samples}")
    return output_dir

def augment_samples(input_dir, output_dir):
    """Применяет аугментацию к сэмплам"""
    
    print(f"\nПрименяем аугментацию: {input_dir} -> {output_dir}")
    
    cmd = f"python3 augment.py --sample-rate 16000 {input_dir}/ {output_dir}/"
    
    if run_command(cmd):
        files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        print(f"Аугментированных сэмплов: {len(files)}")
        return True
    return False

def main():
    """Основная функция"""
    
    print("=== Улучшенная генерация сэмплов для wake-word 'мой дом' ===\n")
    
    # Проверяем наличие генератора
    if not os.path.exists("generate_samples.py"):
        print("Ошибка: generate_samples.py не найден!")
        return
    
    # Генерируем положительные сэмплы
    print("1. Генерация положительных сэмплов...")
    pos_dir = generate_positive_samples()
    
    # Генерируем негативные сэмплы
    print("\n2. Генерация негативных сэмплов...")
    neg_dir = generate_negative_samples()
    
    # Применяем аугментацию к положительным сэмплам
    print("\n3. Аугментация положительных сэмплов...")
    pos_aug_dir = pos_dir + "_aug"
    augment_samples(pos_dir, pos_aug_dir)
    
    # Применяем аугментацию к негативным сэмплам
    print("\n4. Аугментация негативных сэмплов...")
    neg_aug_dir = neg_dir + "_aug"
    augment_samples(neg_dir, neg_aug_dir)
    
    # Итоговая статистика
    print("\n" + "="*50)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("="*50)
    
    pos_files = len([f for f in os.listdir(pos_aug_dir) if f.endswith('.wav')]) if os.path.exists(pos_aug_dir) else 0
    neg_files = len([f for f in os.listdir(neg_aug_dir) if f.endswith('.wav')]) if os.path.exists(neg_aug_dir) else 0
    
    print(f"Положительные сэмплы (аугментированные): {pos_files}")
    print(f"Негативные сэмплы (аугментированные): {neg_files}")
    print(f"Общее количество сэмплов: {pos_files + neg_files}")
    
    print(f"\nПапки с данными:")
    print(f"  Положительные: {pos_aug_dir}")
    print(f"  Негативные: {neg_aug_dir}")
    
    print("\nГотово! Теперь можно обучать модель с улучшенным датасетом.")

if __name__ == "__main__":
    main()