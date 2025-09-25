#!/usr/bin/env python3
"""
Смешивание TTS данных с реальным шумом для предотвращения переобучения
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
from src.utils.path_manager import paths

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Параметры смешивания для разных вариантов
MIXING_VARIANTS = {
    "conservative": {
        "noise_ratio": 0.05,      # 5% шума, 95% TTS
        "snr_range": (20, 30),    # высокий SNR
        "name": "Консервативный"
    },
    "moderate": {
        "noise_ratio": 0.1,       # 10% шума, 90% TTS
        "snr_range": (15, 25),    # средний SNR
        "name": "Умеренный"
    },
    "aggressive": {
        "noise_ratio": 0.2,       # 20% шума, 80% TTS
        "snr_range": (10, 20),    # низкий SNR
        "name": "Агрессивный"
    },
    "extreme": {
        "noise_ratio": 0.3,       # 30% шума, 70% TTS
        "snr_range": (5, 15),     # очень низкий SNR
        "name": "Экстремальный"
    }
}

# Источники TTS данных - будут обновлены в main() с учетом варианта
TTS_SOURCES_TEMPLATE = {
    "positives": {
        "path": paths.POSITIVES_RAW,
        "output_template": f"{paths.DATA_ROOT}/positives_with_noise_{{variant}}/",
        "description": "TTS позитивные данные"
    },
    "negatives": {
        "path": paths.NEGATIVES_TTS,
        "output_template": f"{paths.DATA_ROOT}/negatives_with_noise_{{variant}}/",
        "description": "TTS негативные данные"
    },
    "hard_negatives": {
        "path": "/home/microWakeWord_data/hard_negatives/",
        "output_template": f"{paths.DATA_ROOT}/hard_negatives_with_noise_{{variant}}/",
        "description": "Hard negatives данные"
    },
    "negatives_tts": {
        "path": paths.NEGATIVES_TTS,
        "output_template": f"{paths.DATA_ROOT}/negatives_tts_with_noise_{{variant}}/",
        "description": "TTS негативные данные"
    }
}

# Источник реального шума - СТРОГО СТАТИЧЕСКИЕ ПУТИ ИЗ XML БЕЗ ХАРДКОДА
NOISE_SOURCE = paths.NEGATIVES_RAW

def load_audio(file_path, sr=16000):
    """Загружает аудио файл"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {file_path}: {e}")
        return None

def save_audio(audio, file_path, sr=16000):
    """Сохраняет аудио файл"""
    try:
        sf.write(file_path, audio, sr)
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения {file_path}: {e}")
        return False

def calculate_snr(signal, noise):
    """Вычисляет Signal-to-Noise Ratio"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def mix_audio_with_noise(tts_audio, noise_audio, variant_params):
    """Смешивает TTS аудио с шумом"""
    try:
        # Нормализуем аудио
        tts_audio = librosa.util.normalize(tts_audio)
        noise_audio = librosa.util.normalize(noise_audio)
        
        # Обрезаем шум до длины TTS аудио
        if len(noise_audio) > len(tts_audio):
            start_idx = random.randint(0, len(noise_audio) - len(tts_audio))
            noise_audio = noise_audio[start_idx:start_idx + len(tts_audio)]
        elif len(noise_audio) < len(tts_audio):
            # Повторяем шум если он короче
            repeats = (len(tts_audio) // len(noise_audio)) + 1
            noise_audio = np.tile(noise_audio, repeats)[:len(tts_audio)]
        
        # Вычисляем параметры смешивания
        noise_ratio = variant_params["noise_ratio"]
        snr_min, snr_max = variant_params["snr_range"]
        target_snr = random.uniform(snr_min, snr_max)
        
        # Вычисляем мощность шума для достижения целевого SNR
        signal_power = np.mean(tts_audio ** 2)
        target_noise_power = signal_power / (10 ** (target_snr / 10))
        current_noise_power = np.mean(noise_audio ** 2)
        
        if current_noise_power > 0:
            noise_scale = np.sqrt(target_noise_power / current_noise_power)
            noise_audio = noise_audio * noise_scale
        
        # Смешиваем аудио
        mixed_audio = (1 - noise_ratio) * tts_audio + noise_ratio * noise_audio
        
        # Нормализуем результат
        mixed_audio = librosa.util.normalize(mixed_audio)
        
        return mixed_audio
        
    except Exception as e:
        logger.error(f"❌ Ошибка смешивания аудио: {e}")
        return None

def process_single_file(args):
    """Обрабатывает один файл"""
    tts_file, noise_files, variant_params, output_dir = args
    
    try:
        # Загружаем TTS аудио
        tts_audio = load_audio(tts_file)
        if tts_audio is None:
            return f"❌ Не удалось загрузить TTS файл: {tts_file}"
        
        # Выбираем случайный файл шума
        noise_file = random.choice(noise_files)
        noise_audio = load_audio(noise_file)
        if noise_audio is None:
            return f"❌ Не удалось загрузить файл шума: {noise_file}"
        
        # Смешиваем аудио
        mixed_audio = mix_audio_with_noise(tts_audio, noise_audio, variant_params)
        if mixed_audio is None:
            return f"❌ Не удалось смешать аудио для: {tts_file}"
        
        # Создаем выходной путь
        tts_path = Path(tts_file)
        output_file = Path(output_dir) / tts_path.name
        
        # Сохраняем результат
        if save_audio(mixed_audio, output_file):
            return f"✅ Обработан: {tts_path.name}"
        else:
            return f"❌ Не удалось сохранить: {tts_path.name}"
            
    except Exception as e:
        return f"❌ Ошибка обработки {tts_file}: {e}"

def get_noise_files():
    """Получает список файлов шума"""
    noise_files = []
    if os.path.exists(NOISE_SOURCE):
        for file_path in Path(NOISE_SOURCE).glob("*.wav"):
            noise_files.append(str(file_path))
    
    if not noise_files:
        logger.error(f"❌ Не найдены файлы шума в {NOISE_SOURCE}")
        return None
    
    logger.info(f"📁 Найдено {len(noise_files)} файлов шума")
    return noise_files

def process_tts_source(source_name, source_info, variant_params, noise_files):
    """Обрабатывает один источник TTS данных"""
    logger.info(f"🔄 Обработка {source_info['description']}...")
    
    # Проверяем существование исходной директории
    if not os.path.exists(source_info["path"]):
        logger.warning(f"⚠️ Директория не найдена: {source_info['path']}")
        return False
    
    # Создаем выходную директорию
    os.makedirs(source_info["output"], exist_ok=True)
    
    # Получаем список TTS файлов
    tts_files = list(Path(source_info["path"]).glob("*.wav"))
    if not tts_files:
        logger.warning(f"⚠️ Не найдены WAV файлы в {source_info['path']}")
        return False
    
    logger.info(f"📊 Обрабатываем {len(tts_files)} файлов...")
    
    # Подготавливаем задачи для параллельной обработки
    tasks = []
    for tts_file in tts_files:
        tasks.append((str(tts_file), noise_files, variant_params, source_info["output"]))
    
    # Обрабатываем файлы параллельно
    max_workers = min(cpu_count(), 8)  # Ограничиваем количество процессов
    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Отправляем задачи
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        # Обрабатываем результаты с прогресс-баром
        with tqdm(total=len(tasks), desc=f"Обработка {source_name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result.startswith("✅"):
                    processed_count += 1
                else:
                    error_count += 1
                    logger.error(result)
                pbar.update(1)
    
    logger.info(f"✅ {source_name}: {processed_count} успешно, {error_count} ошибок")
    return processed_count > 0

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Смешивание TTS данных с реальным шумом")
    parser.add_argument("--variant", choices=list(MIXING_VARIANTS.keys()), 
                       default="moderate", help="Вариант параметров смешивания")
    parser.add_argument("--source", choices=list(TTS_SOURCES.keys()), 
                       help="Обработать только указанный источник")
    parser.add_argument("--test", action="store_true", 
                       help="Тестовый режим (обработать только 10 файлов)")
    
    args = parser.parse_args()
    
    # Получаем параметры варианта
    variant_params = MIXING_VARIANTS[args.variant]
    variant_name = variant_params["name"]
    
    logger.info(f"🎯 СМЕШИВАНИЕ TTS ДАННЫХ С РЕАЛЬНЫМ ШУМОМ")
    logger.info(f"📊 Вариант: {variant_name}")
    logger.info(f"📊 Параметры: {variant_params}")
    
    # Создаем TTS_SOURCES с правильными именами папок для варианта
    TTS_SOURCES = {}
    for source_name, source_info in TTS_SOURCES_TEMPLATE.items():
        TTS_SOURCES[source_name] = {
            "path": source_info["path"],
            "output": source_info["output_template"].format(variant=args.variant),
            "description": source_info["description"]
        }
    
    # Получаем файлы шума
    noise_files = get_noise_files()
    if not noise_files:
        return 1
    
    # Определяем источники для обработки
    if args.source:
        sources_to_process = {args.source: TTS_SOURCES[args.source]}
    else:
        sources_to_process = TTS_SOURCES
    
    # Обрабатываем каждый источник
    total_processed = 0
    total_errors = 0
    
    for source_name, source_info in sources_to_process.items():
        logger.info(f"\n🔄 ОБРАБОТКА ИСТОЧНИКА: {source_name}")
        logger.info(f"📁 Входная директория: {source_info['path']}")
        logger.info(f"📁 Выходная директория: {source_info['output']}")
        
        # Тестовый режим - ограничиваем количество файлов
        if args.test:
            logger.info("🧪 ТЕСТОВЫЙ РЕЖИМ - ограничиваем количество файлов")
            # Временно изменяем список файлов для тестирования
            original_path = source_info["path"]
            test_files = list(Path(original_path).glob("*.wav"))[:10]
            if test_files:
                # Создаем временную директорию с ограниченным набором файлов
                test_dir = f"/tmp/test_{source_name}"
                os.makedirs(test_dir, exist_ok=True)
                for test_file in test_files:
                    import shutil
                    shutil.copy2(test_file, test_dir)
                source_info["path"] = test_dir
        
        success = process_tts_source(source_name, source_info, variant_params, noise_files)
        
        if success:
            total_processed += 1
        else:
            total_errors += 1
    
    # Итоговый отчет
    logger.info(f"\n🎉 СМЕШИВАНИЕ ЗАВЕРШЕНО!")
    logger.info(f"📊 Вариант: {variant_name}")
    logger.info(f"✅ Успешно обработано источников: {total_processed}")
    logger.info(f"❌ Ошибок: {total_errors}")
    
    if total_processed > 0:
        logger.info(f"📁 Результаты сохранены в директориях с суффиксом '_with_noise'")
        logger.info(f"🚀 Готово для следующего этапа: аугментации и обучение")
        return 0
    else:
        logger.error(f"❌ Не удалось обработать ни одного источника")
        return 1

if __name__ == "__main__":
    sys.exit(main())