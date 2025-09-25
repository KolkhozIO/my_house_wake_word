#!/usr/bin/env python3
"""
Запуск полного пайплайна microWakeWord как неблокирующих задач
"""

import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd):
    """Выполнение команды и возврат результата"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🚀 Запуск полного пайплайна microWakeWord")
    print("=" * 50)
    
    # Проверяем, что task_manager.py доступен
    if not Path("task_manager.py").exists():
        print("❌ task_manager.py не найден!")
        return
    
    # Этап 1: Генерация данных
    print("📝 Этап 1: Генерация TTS данных...")
    success, stdout, stderr = run_command(
        "source .venv/bin/activate && python task_manager.py start generate_data 'source .venv/bin/activate && python generate_both_phrases.py'"
    )
    
    if not success:
        print(f"❌ Ошибка запуска генерации данных: {stderr}")
        return
    
    print("✅ Генерация данных запущена в фоне")
    
    # Ждем завершения генерации
    print("⏳ Ожидание завершения генерации данных...")
    while True:
        success, stdout, stderr = run_command("source .venv/bin/activate && python task_manager.py status generate_data")
        if "🔴" in stdout or "finished" in stdout:
            print("✅ Генерация данных завершена")
            break
        elif "❌" in stdout:
            print("❌ Ошибка в генерации данных")
            return
        time.sleep(10)
    
    # Этап 2: Аугментации
    print("🎨 Этап 2: Применение аугментаций...")
    success, stdout, stderr = run_command(
        "source .venv/bin/activate && python task_manager.py start augmentations 'source .venv/bin/activate && python apply_augmentations.py'"
    )
    
    if not success:
        print(f"❌ Ошибка запуска аугментаций: {stderr}")
        return
    
    print("✅ Аугментации запущены в фоне")
    
    # Ждем завершения аугментаций
    print("⏳ Ожидание завершения аугментаций...")
    while True:
        success, stdout, stderr = run_command("source .venv/bin/activate && python task_manager.py status augmentations")
        if "🔴" in stdout or "finished" in stdout:
            print("✅ Аугментации завершены")
            break
        elif "❌" in stdout:
            print("❌ Ошибка в аугментациях")
            return
        time.sleep(15)
    
    # Этап 3: Балансировка датасета
    print("⚖️ Этап 3: Балансировка датасета...")
    success, stdout, stderr = run_command(
        "source .venv/bin/activate && python task_manager.py start balance_dataset 'source .venv/bin/activate && python balance_dataset.py'"
    )
    
    if not success:
        print(f"❌ Ошибка запуска балансировки: {stderr}")
        return
    
    print("✅ Балансировка запущена в фоне")
    
    # Ждем завершения балансировки
    print("⏳ Ожидание завершения балансировки...")
    while True:
        success, stdout, stderr = run_command("source .venv/bin/activate && python task_manager.py status balance_dataset")
        if "🔴" in stdout or "finished" in stdout:
            print("✅ Балансировка завершена")
            break
        elif "❌" in stdout:
            print("❌ Ошибка в балансировке")
            return
        time.sleep(20)
    
    # Этап 4: Обучение с оригинальной библиотекой
    print("🧠 Этап 4: Обучение модели...")
    success, stdout, stderr = run_command(
        "source .venv/bin/activate && python task_manager.py start train_model 'source .venv/bin/activate && python use_original_library_correctly.py'"
    )
    
    if not success:
        print(f"❌ Ошибка запуска обучения: {stderr}")
        return
    
    print("✅ Обучение запущено в фоне")
    
    # Ждем завершения обучения
    print("⏳ Ожидание завершения обучения...")
    while True:
        success, stdout, stderr = run_command("source .venv/bin/activate && python task_manager.py status train_model")
        if "🔴" in stdout or "finished" in stdout:
            print("✅ Обучение завершено")
            break
        elif "❌" in stdout:
            print("❌ Ошибка в обучении")
            return
        time.sleep(30)
    
    # Этап 5: Обучение большей модели
    print("🚀 Этап 5: Обучение большей модели...")
    success, stdout, stderr = run_command(
        "source .venv/bin/activate && python task_manager.py start train_larger 'source .venv/bin/activate && python train_larger_model.py'"
    )
    
    if not success:
        print(f"❌ Ошибка запуска обучения большей модели: {stderr}")
        return
    
    print("✅ Обучение большей модели запущено в фоне")
    
    # Ждем завершения обучения большей модели
    print("⏳ Ожидание завершения обучения большей модели...")
    while True:
        success, stdout, stderr = run_command("source .venv/bin/activate && python task_manager.py status train_larger")
        if "🔴" in stdout or "finished" in stdout:
            print("✅ Обучение большей модели завершено")
            break
        elif "❌" in stdout:
            print("❌ Ошибка в обучении большей модели")
            return
        time.sleep(30)
    
    print("\n🎉 Весь пайплайн завершен!")
    print("📋 Для мониторинга используйте:")
    print("   ./manage_tasks.sh status")
    print("   ./manage_tasks.sh logs <имя_задачи>")
    print("   ./manage_tasks.sh stop <имя_задачи>")

if __name__ == "__main__":
    main()