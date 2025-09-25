#!/usr/bin/env python3
"""
Генерация уникальных имен моделей с комментариями об изменениях
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def generate_model_name(comment=""):
    """Генерирует уникальное имя модели с комментарием"""
    
    # Получаем текущую дату и время
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    
    # Подсчитываем количество семплов
    negatives_count = len(list(Path("/home/microWakeWord_data/negatives_real_sampled").glob("*.wav")))
    background_count = len(list(Path("/home/microWakeWord_data/background_data_sampled").glob("*.wav")))
    positives_count = len(list(Path("/home/microWakeWord_data/positives_final").glob("*.wav")))
    
    total_samples = negatives_count + background_count + positives_count
    
    # Создаем описание на основе данных
    description_parts = []
    
    if negatives_count > 0:
        description_parts.append(f"neg_{negatives_count//1000}k")
    
    if background_count > 0:
        description_parts.append(f"bg_{background_count//1000}k")
    
    if positives_count > 0:
        description_parts.append(f"pos_{positives_count//1000}k")
    
    # Добавляем общее количество семплов
    description_parts.append(f"total_{total_samples//1000}k")
    
    # Добавляем комментарий если есть
    if comment:
        # Очищаем комментарий от специальных символов
        clean_comment = "".join(c for c in comment if c.isalnum() or c in "_-").replace(" ", "_")
        description_parts.append(clean_comment)
    
    # Создаем финальное имя
    description = "_".join(description_parts)
    model_name = f"model_{timestamp}_{description}.tflite"
    
    return model_name, {
        "timestamp": timestamp,
        "negatives_count": negatives_count,
        "background_count": background_count,
        "positives_count": positives_count,
        "total_samples": total_samples,
        "comment": comment,
        "description": description
    }

def update_training_config(model_name):
    """Обновляет конфигурацию обучения для нового имени модели"""
    
    config_file = "/home/microWakeWord/training_parameters.yaml"
    
    # Читаем текущую конфигурацию
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Обновляем путь к модели
    new_content = content.replace(
        "train_dir: trained_models/wakeword",
        f"train_dir: trained_models/{model_name.replace('.tflite', '')}"
    )
    
    # Сохраняем обновленную конфигурацию
    with open(config_file, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Конфигурация обновлена для модели: {model_name}")

def create_model_manifest(model_name, model_info):
    """Создает JSON манифест для модели"""
    
    manifest_content = {
        "version": 2,
        "type": "micro",
        "model_path": f"trained_models/{model_name.replace('.tflite', '')}/{model_name}",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "timestamp": model_info["timestamp"],
            "negatives_samples": model_info["negatives_count"],
            "background_samples": model_info["background_count"],
            "positives_samples": model_info["positives_count"],
            "total_samples": model_info["total_samples"],
            "comment": model_info["comment"],
            "description": model_info["description"]
        },
        "parameters": {
            "probability_cutoff": 0.95,
            "sliding_window_size": 5,
            "tensor_arena_size": 1000000
        }
    }
    
    import json
    manifest_file = f"/home/microWakeWord_data/{model_name.replace('.tflite', '.json')}"
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest_content, f, indent=2)
    
    print(f"✅ Манифест создан: {manifest_file}")
    return manifest_file

def main():
    """Основная функция"""
    
    if len(sys.argv) < 2:
        print("❌ Использование: python generate_unique_model_name.py 'комментарий об изменениях'")
        print("Пример: python generate_unique_model_name.py 'max_coverage_sampling_840k_samples'")
        sys.exit(1)
    
    comment = sys.argv[1]
    
    print("=== ГЕНЕРАЦИЯ УНИКАЛЬНОГО ИМЕНИ МОДЕЛИ ===")
    print(f"Комментарий: {comment}")
    
    # Генерируем имя модели
    model_name, model_info = generate_model_name(comment)
    
    print(f"\n📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
    print(f"   Имя: {model_name}")
    print(f"   Временная метка: {model_info['timestamp']}")
    print(f"   Негативные семплы: {model_info['negatives_count']:,}")
    print(f"   Фоновые семплы: {model_info['background_count']:,}")
    print(f"   Позитивные семплы: {model_info['positives_count']:,}")
    print(f"   Всего семплов: {model_info['total_samples']:,}")
    print(f"   Комментарий: {model_info['comment']}")
    
    # Обновляем конфигурацию
    update_training_config(model_name)
    
    # Создаем манифест
    manifest_file = create_model_manifest(model_name, model_info)
    
    print(f"\n🎯 ГОТОВО!")
    print(f"   Модель: {model_name}")
    print(f"   Манифест: {manifest_file}")
    print(f"   Конфигурация обновлена")
    
    return model_name

if __name__ == "__main__":
    model_name = main()
    print(f"\n🚀 Для запуска обучения используйте:")
    print(f"   ./manage_tasks.sh start train_model")
    print(f"   Результат будет сохранен как: {model_name}")