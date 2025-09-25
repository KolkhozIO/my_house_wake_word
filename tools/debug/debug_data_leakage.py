#!/usr/bin/env python3
"""
Проверка на утечку данных между train/test наборами
"""

import os
import sys
import numpy as np
import librosa
import glob
import hashlib
from collections import defaultdict

# Добавляем путь к библиотеке microWakeWord
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths

def get_file_hash(file_path):
    """Получает хеш файла"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def get_audio_hash(file_path):
    """Получает хеш аудио данных"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        # Обрезаем до 1.5 секунд для консистентности
        target_samples = int(1.5 * sr)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Создаем хеш от аудио данных
        return hashlib.md5(audio.tobytes()).hexdigest()
    except:
        return None

def check_data_leakage():
    """Проверяет утечку данных между train/test"""
    print("🔍 ПРОВЕРКА НА УТЕЧКУ ДАННЫХ:")
    print("=" * 50)
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    # Получаем файлы
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:200]
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:200]
    
    print(f"📁 Проверяем {len(positive_files)} позитивных и {len(negative_files)} негативных файлов")
    
    # Проверяем дубликаты файлов
    print(f"\n🔍 ПРОВЕРКА ДУБЛИКАТОВ ФАЙЛОВ:")
    
    file_hashes = defaultdict(list)
    audio_hashes = defaultdict(list)
    
    all_files = positive_files + negative_files
    
    for file_path in all_files:
        file_hash = get_file_hash(file_path)
        audio_hash = get_audio_hash(file_path)
        
        if file_hash:
            file_hashes[file_hash].append(file_path)
        if audio_hash:
            audio_hashes[audio_hash].append(file_path)
    
    # Проверяем дубликаты файлов
    file_duplicates = {h: files for h, files in file_hashes.items() if len(files) > 1}
    audio_duplicates = {h: files for h, files in audio_hashes.items() if len(files) > 1}
    
    print(f"📊 Дубликаты файлов: {len(file_duplicates)}")
    print(f"📊 Дубликаты аудио: {len(audio_duplicates)}")
    
    if file_duplicates:
        print(f"\n🚨 НАЙДЕНЫ ДУБЛИКАТЫ ФАЙЛОВ:")
        for i, (hash_val, files) in enumerate(list(file_duplicates.items())[:5]):
            print(f"  {i+1}. Хеш {hash_val[:8]}...:")
            for file_path in files:
                print(f"     - {os.path.basename(file_path)}")
    
    if audio_duplicates:
        print(f"\n🚨 НАЙДЕНЫ ДУБЛИКАТЫ АУДИО:")
        for i, (hash_val, files) in enumerate(list(audio_duplicates.items())[:5]):
            print(f"  {i+1}. Аудио хеш {hash_val[:8]}...:")
            for file_path in files:
                print(f"     - {os.path.basename(file_path)}")
    
    # Проверяем статистики данных
    print(f"\n📈 АНАЛИЗ СТАТИСТИК ДАННЫХ:")
    
    pos_data = []
    neg_data = []
    
    for file_path in positive_files[:50]:  # Берем только 50 для скорости
        try:
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            target_samples = int(1.5 * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            pos_data.append(audio)
        except:
            pass
    
    for file_path in negative_files[:50]:  # Берем только 50 для скорости
        try:
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            target_samples = int(1.5 * sr)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            neg_data.append(audio)
        except:
            pass
    
    if pos_data and neg_data:
        pos_data = np.array(pos_data)
        neg_data = np.array(neg_data)
        
        print(f"   Позитивные: mean={np.mean(pos_data):.3f}, std={np.std(pos_data):.3f}")
        print(f"   Негативные: mean={np.mean(neg_data):.3f}, std={np.std(neg_data):.3f}")
        print(f"   Разница: {abs(np.mean(pos_data) - np.mean(neg_data)):.3f}")
        
        # Проверяем перекрытие распределений
        pos_min, pos_max = np.min(pos_data), np.max(pos_data)
        neg_min, neg_max = np.min(neg_data), np.max(neg_data)
        
        overlap = max(0, min(pos_max, neg_max) - max(pos_min, neg_min))
        pos_range = pos_max - pos_min
        neg_range = neg_max - neg_min
        
        print(f"   Перекрытие диапазонов: {overlap:.3f}")
        print(f"   Позитивный диапазон: {pos_range:.3f}")
        print(f"   Негативный диапазон: {neg_range:.3f}")
        
        if overlap / max(pos_range, neg_range) > 0.8:
            print("⚠️ ВНИМАНИЕ: Сильное перекрытие распределений!")
        else:
            print("✅ Распределения достаточно разделены")
    
    # Проверяем имена файлов на паттерны
    print(f"\n🔍 АНАЛИЗ ИМЕН ФАЙЛОВ:")
    
    pos_names = [os.path.basename(f) for f in positive_files[:20]]
    neg_names = [os.path.basename(f) for f in negative_files[:20]]
    
    print(f"   Позитивные файлы (первые 5): {pos_names[:5]}")
    print(f"   Негативные файлы (первые 5): {neg_names[:5]}")
    
    # Проверяем на общие паттерны
    pos_prefixes = set([name.split('_')[0] for name in pos_names if '_' in name])
    neg_prefixes = set([name.split('_')[0] for name in neg_names if '_' in name])
    
    common_prefixes = pos_prefixes.intersection(neg_prefixes)
    if common_prefixes:
        print(f"⚠️ ВНИМАНИЕ: Общие префиксы в именах: {common_prefixes}")
    else:
        print("✅ Нет общих префиксов в именах файлов")
    
    return {
        'file_duplicates': len(file_duplicates),
        'audio_duplicates': len(audio_duplicates),
        'pos_data_mean': np.mean(pos_data) if len(pos_data) > 0 else 0,
        'neg_data_mean': np.mean(neg_data) if len(neg_data) > 0 else 0,
        'data_difference': abs(np.mean(pos_data) - np.mean(neg_data)) if len(pos_data) > 0 and len(neg_data) > 0 else 0
    }

def main():
    """Основная функция"""
    print("🚀 ПРОВЕРКА НА УТЕЧКУ ДАННЫХ")
    print("=" * 60)
    
    results = check_data_leakage()
    
    print(f"\n🎉 ПРОВЕРКА ЗАВЕРШЕНА!")
    
    # Анализ результатов
    if results['file_duplicates'] > 0 or results['audio_duplicates'] > 0:
        print("🚨 ОБНАРУЖЕНА УТЕЧКА ДАННЫХ!")
        print("   - Есть дубликаты файлов или аудио данных")
        print("   - Это может объяснить 100% точность")
    elif results['data_difference'] < 5:
        print("⚠️ ПОДОЗРИТЕЛЬНО МАЛАЯ РАЗНИЦА В ДАННЫХ!")
        print("   - Позитивные и негативные данные слишком похожи")
        print("   - Это может объяснить 100% точность")
    else:
        print("✅ Данные выглядят корректно")
        print("   - Нет явной утечки данных")
        print("   - Проблема может быть в архитектуре модели")

if __name__ == "__main__":
    main()