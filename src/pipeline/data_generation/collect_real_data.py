#!/usr/bin/env python3
"""
Сбор реальных wake word данных для исправления переобучения
"""

import os
import sys
import logging
import subprocess
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_real_data_structure():
    """Создает структуру для сбора реальных данных"""
    logger.info("📁 Создание структуры для реальных данных...")
    
    try:
        # Создаем директории
        real_data_dir = "/home/microWakeWord_data/real_wake_word_data"
        os.makedirs(real_data_dir, exist_ok=True)
        
        # Поддиректории
        subdirs = [
            "positives_raw",      # Сырые записи wake word
            "positives_processed", # Обработанные записи
            "noise_raw",      # Сырые записи фона
            "negatives_processed", # Обработанные записи фона
            "validation_set",     # Валидационный набор
            "test_set"           # Тестовый набор
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(real_data_dir, subdir), exist_ok=True)
        
        logger.info("✅ Структура создана")
        return real_data_dir
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания структуры: {e}")
        return None

def generate_real_wake_word_instructions():
    """Создает инструкции для сбора реальных данных"""
    logger.info("📝 Создание инструкций для сбора данных...")
    
    instructions = """
# ИНСТРУКЦИИ ПО СБОРУ РЕАЛЬНЫХ WAKE WORD ДАННЫХ

## 🎯 ЦЕЛЬ
Собрать реальные записи фразы "милый дом" / "любимый дом" для исправления переобучения модели.

## 📋 ЧТО НУЖНО ЗАПИСАТЬ

### ✅ ПОЗИТИВНЫЕ ДАННЫЕ (Wake Word):
- Фраза: "милый дом" (разные интонации)
- Фраза: "любимый дом" (разные интонации)
- Разные голоса (мужские, женские, детские)
- Разные расстояния (близко, далеко)
- Разные комнаты (тихая, шумная)
- Разное время суток

### ❌ НЕГАТИВНЫЕ ДАННЫЕ (Фон):
- Обычная речь (без wake word)
- Музыка, телевизор, радио
- Шумы дома (стиральная машина, пылесос)
- Уличные шумы
- Разговоры других людей

## 📊 ТРЕБУЕМОЕ КОЛИЧЕСТВО

### Минимальные требования:
- **Позитивные**: 1000 записей (по 500 каждой фразы)
- **Негативные**: 10000 записей (разные типы фона)
- **Длительность**: 1-3 секунды на запись
- **Качество**: 16kHz, моно, WAV

### Рекомендуемые параметры:
- **Позитивные**: 2000+ записей
- **Негативные**: 20000+ записей
- **Разнообразие**: 10+ разных голосов
- **Условия**: 5+ разных комнат/условий

## 🎙️ ПАРАМЕТРЫ ЗАПИСИ

### Технические требования:
- **Sample Rate**: 16000 Hz
- **Channels**: Моно (1 канал)
- **Format**: WAV, 16-bit
- **Длительность**: 1-5 секунд
- **Громкость**: Нормальная речь (не крик, не шепот)

### Условия записи:
- **Расстояние**: 1-3 метра от микрофона
- **Окружение**: Разные комнаты дома
- **Время**: Разное время суток
- **Фон**: Естественный домашний шум

## 📁 СТРУКТУРА ФАЙЛОВ

### Позитивные данные:
```
/home/microWakeWord_data/real_wake_word_data/positives_raw/
├── miliy_dom_male_001.wav
├── miliy_dom_female_001.wav
├── lyubimiy_dom_male_001.wav
├── lyubimiy_dom_female_001.wav
└── ...
```

### Негативные данные:
```
/home/microWakeWord_data/real_wake_word_data/noise_raw/
├── speech_001.wav
├── music_001.wav
├── tv_001.wav
├── noise_001.wav
└── ...
```

## 🔧 ИНСТРУМЕНТЫ ДЛЯ ЗАПИСИ

### Рекомендуемые программы:
- **Audacity**: Бесплатная, кроссплатформенная
- **QuickTime** (Mac): Встроенная запись
- **Voice Recorder** (Windows): Встроенная запись
- **Мобильные приложения**: Voice Recorder, Audio Recorder

### Настройки Audacity:
1. File → New
2. Transport → Record
3. File → Export → Export Audio
4. Format: WAV, Sample Rate: 16000 Hz, Channels: Mono

## 📝 ПЛАН СБОРА

### Неделя 1: Подготовка
- [ ] Установить Audacity
- [ ] Настроить микрофон
- [ ] Создать шаблоны файлов
- [ ] Подготовить список фраз

### Неделя 2: Позитивные данные
- [ ] Записать 200 фраз "милый дом"
- [ ] Записать 200 фраз "любимый дом"
- [ ] Разные голоса и интонации
- [ ] Проверить качество записей

### Неделя 3: Негативные данные
- [ ] Записать 2000 фоновых записей
- [ ] Разные типы шумов
- [ ] Разные комнаты
- [ ] Проверить разнообразие

### Неделя 4: Обработка
- [ ] Обрезать записи до 1-3 секунд
- [ ] Нормализовать громкость
- [ ] Проверить качество
- [ ] Разделить на train/validation/test

## ⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ

### Качество данных:
- **Четкость**: Фразы должны быть четко произнесены
- **Естественность**: Не читать по бумажке, говорить естественно
- **Разнообразие**: Разные интонации, темпы, эмоции
- **Реалистичность**: Условия как в реальном использовании

### Избегать:
- **Искусственность**: Не говорить слишком четко/медленно
- **Повторения**: Не записывать одинаковые интонации
- **Шумы**: Избегать сильных фоновых шумов
- **Искажения**: Проверять качество записи

## 🎯 КРИТЕРИИ УСПЕХА

### После сбора данных:
- [ ] 1000+ позитивных записей
- [ ] 10000+ негативных записей
- [ ] 10+ разных голосов
- [ ] 5+ разных условий записи
- [ ] Качество 16kHz, моно, WAV
- [ ] Длительность 1-3 секунды

### После обработки:
- [ ] Все файлы в правильном формате
- [ ] Нормализованная громкость
- [ ] Разделение на train/validation/test
- [ ] Метаданные о записях
- [ ] Проверка качества

## 📞 ПОДДЕРЖКА

При возникновении проблем:
1. Проверить настройки микрофона
2. Проверить формат файлов
3. Проверить качество записей
4. Обратиться за помощью к разработчикам

---
**ПОМНИТЕ: Качество данных = Качество модели!**
"""
    
    try:
        instructions_path = "/home/microWakeWord_data/real_data_collection_instructions.md"
        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(instructions)
        
        logger.info(f"✅ Инструкции сохранены: {instructions_path}")
        return instructions_path
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания инструкций: {e}")
        return None

def create_data_collection_script():
    """Создает скрипт для автоматической обработки собранных данных"""
    logger.info("🔧 Создание скрипта обработки данных...")
    
    script_content = '''#!/usr/bin/env python3
"""
Автоматическая обработка собранных реальных данных
"""

import os
import sys
import librosa
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_audio_files(input_dir, output_dir, target_duration=1.5):
    """Обрабатывает аудиофайлы: нормализует, обрезает, конвертирует"""
    logger.info(f"🔄 Обработка файлов из {input_dir} в {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    for file_path in Path(input_dir).glob("*.wav"):
        try:
            # Загружаем аудио
            audio, sr = librosa.load(str(file_path), sr=16000)
            
            # Нормализуем громкость
            audio = librosa.util.normalize(audio)
            
            # Обрезаем до целевой длительности
            target_samples = int(target_duration * sr)
            if len(audio) > target_samples:
                # Берем середину
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            elif len(audio) < target_samples:
                # Дополняем нулями
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Сохраняем обработанный файл
            output_file = os.path.join(output_dir, file_path.name)
            librosa.output.write_wav(output_file, audio, sr)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {file_path}: {e}")
            error_count += 1
    
    logger.info(f"✅ Обработано: {processed_count}, Ошибок: {error_count}")
    return processed_count, error_count

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Разделяет датасет на train/validation/test"""
    logger.info("📊 Разделение датасета на train/validation/test")
    
    import random
    random.seed(42)
    
    # Создаем директории
    splits = ["train", "validation", "test"]
    for split in splits:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    
    # Обрабатываем позитивные данные
    pos_dir = os.path.join(data_dir, "positives_processed")
    if os.path.exists(pos_dir):
        pos_files = list(Path(pos_dir).glob("*.wav"))
        random.shuffle(pos_files)
        
        n_train = int(len(pos_files) * train_ratio)
        n_val = int(len(pos_files) * val_ratio)
        
        train_files = pos_files[:n_train]
        val_files = pos_files[n_train:n_train + n_val]
        test_files = pos_files[n_train + n_val:]
        
        # Копируем файлы
        for files, split in [(train_files, "train"), (val_files, "validation"), (test_files, "test")]:
            split_dir = os.path.join(data_dir, split, "positives")
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                import shutil
                shutil.copy2(file_path, os.path.join(split_dir, file_path.name))
    
    # Обрабатываем негативные данные
    neg_dir = os.path.join(data_dir, "negatives_processed")
    if os.path.exists(neg_dir):
        neg_files = list(Path(neg_dir).glob("*.wav"))
        random.shuffle(neg_files)
        
        n_train = int(len(neg_files) * train_ratio)
        n_val = int(len(neg_files) * val_ratio)
        
        train_files = neg_files[:n_train]
        val_files = neg_files[n_train:n_train + n_val]
        test_files = neg_files[n_train + n_val:]
        
        # Копируем файлы
        for files, split in [(train_files, "train"), (val_files, "validation"), (test_files, "test")]:
            split_dir = os.path.join(data_dir, split, "negatives")
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                import shutil
                shutil.copy2(file_path, os.path.join(split_dir, file_path.name))
    
    logger.info("✅ Датасет разделен на train/validation/test")

def main():
    """Основная функция обработки данных"""
    logger.info("🎯 ОБРАБОТКА РЕАЛЬНЫХ ДАННЫХ")
    
    real_data_dir = "/home/microWakeWord_data/real_wake_word_data"
    
    # Обрабатываем позитивные данные
    pos_raw = os.path.join(real_data_dir, "positives_raw")
    pos_processed = os.path.join(real_data_dir, "positives_processed")
    if os.path.exists(pos_raw):
        process_audio_files(pos_raw, pos_processed)
    
    # Обрабатываем негативные данные
    neg_raw = os.path.join(real_data_dir, "noise_raw")
    neg_processed = os.path.join(real_data_dir, "negatives_processed")
    if os.path.exists(neg_raw):
        process_audio_files(neg_raw, neg_processed)
    
    # Разделяем на train/validation/test
    split_dataset(real_data_dir)
    
    logger.info("🎉 Обработка данных завершена!")

if __name__ == "__main__":
    main()
'''
    
    try:
        script_path = "/home/microWakeWord/process_real_data.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # Делаем скрипт исполняемым
        os.chmod(script_path, 0o755)
        
        logger.info(f"✅ Скрипт обработки создан: {script_path}")
        return script_path
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания скрипта: {e}")
        return None

def main():
    """Основная функция"""
    logger.info("🎯 ПОДГОТОВКА К СБОРУ РЕАЛЬНЫХ ДАННЫХ")
    
    # Этап 1: Создание структуры
    real_data_dir = create_real_data_structure()
    if not real_data_dir:
        logger.error("❌ Не удалось создать структуру данных")
        return False
    
    # Этап 2: Создание инструкций
    instructions_path = generate_real_wake_word_instructions()
    if not instructions_path:
        logger.error("❌ Не удалось создать инструкции")
        return False
    
    # Этап 3: Создание скрипта обработки
    script_path = create_data_collection_script()
    if not script_path:
        logger.error("❌ Не удалось создать скрипт обработки")
        return False
    
    logger.info("🎉 ПОДГОТОВКА ЗАВЕРШЕНА!")
    logger.info(f"📁 Структура данных: {real_data_dir}")
    logger.info(f"📝 Инструкции: {instructions_path}")
    logger.info(f"🔧 Скрипт обработки: {script_path}")
    
    logger.info("\\n📋 СЛЕДУЮЩИЕ ШАГИ:")
    logger.info("1. Прочитайте инструкции в real_data_collection_instructions.md")
    logger.info("2. Соберите реальные записи wake word")
    logger.info("3. Запустите: python process_real_data.py")
    logger.info("4. Переобучите модель на смешанном датасете")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Подготовка завершена успешно")
    else:
        logger.error("❌ Подготовка завершена с ошибками")
        sys.exit(1)