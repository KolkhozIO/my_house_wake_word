#!/usr/bin/env python3
"""
Неблокирующий запуск пайплайна microWakeWord
Просто запускает все задачи и сразу возвращает управление
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Выполнение команды в фоне"""
    try:
        subprocess.Popen(cmd, shell=True)
        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    print("🚀 Запуск полного пайплайна microWakeWord (неблокирующий)")
    print("=" * 60)
    
    # Проверяем, что task_manager.py доступен
    task_manager_path = Path("src/management/task_manager.py")
    if not task_manager_path.exists():
        print("❌ task_manager.py не найден!")
        return
    
    # Этап 1: Генерация данных
    print("📝 Этап 1: Запуск генерации TTS данных...")
    success = run_command(
        "source .venv/bin/activate && python src/management/task_manager.py start generate_data 'source .venv/bin/activate && python src/pipeline/data_generation/generate_both_phrases.py'"
    )
    if success:
        print("✅ Генерация данных запущена в фоне")
    else:
        print("❌ Ошибка запуска генерации данных")
        return
    
    # Этап 2: Аугментации (запустится автоматически после генерации)
    print("🎨 Этап 2: Запуск аугментаций...")
    success = run_command(
        "source .venv/bin/activate && python src/management/task_manager.py start augmentations 'source .venv/bin/activate && python src/pipeline/augmentation/apply_augmentations.py'"
    )
    if success:
        print("✅ Аугментации запущены в фоне")
    else:
        print("❌ Ошибка запуска аугментаций")
    
    # Этап 3: Балансировка датасета
    print("⚖️ Этап 3: Запуск балансировки датасета...")
    success = run_command(
        "source .venv/bin/activate && python src/management/task_manager.py start balance_dataset 'source .venv/bin/activate && python src/pipeline/balancing/balance_dataset.py'"
    )
    if success:
        print("✅ Балансировка запущена в фоне")
    else:
        print("❌ Ошибка запуска балансировки")
    
    # Этап 4: Обучение с оригинальной библиотекой
    print("🧠 Этап 4: Запуск обучения модели...")
    success = run_command(
        "source .venv/bin/activate && python src/management/task_manager.py start train_model 'source .venv/bin/activate && python src/pipeline/training/use_original_library_correctly_fixed.py'"
    )
    if success:
        print("✅ Обучение запущено в фоне")
    else:
        print("❌ Ошибка запуска обучения")
    
    
    print("\n🎉 Все задачи запущены в фоне!")
    print("📋 Для мониторинга используйте:")
    print("   ./manage_tasks.sh status")
    print("   ./manage_tasks.sh logs <имя_задачи>")
    print("   ./manage_tasks.sh stop <имя_задачи>")
    print("\n💡 Все задачи выполняются параллельно и независимо!")

if __name__ == "__main__":
    main()