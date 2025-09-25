#!/usr/bin/env python3
"""
Обучение модели microWakeWord с расширенными данными и исправленными параметрами
"""

import os
import sys
import subprocess
import psutil
import time
from pathlib import Path

def monitor_resources(process, duration=60):
    """Мониторинг ресурсов процесса в реальном времени"""
    print(f"\n📊 Мониторинг ресурсов процесса (PID: {process.pid}) на {duration} секунд...")
    
    start_time = time.time()
    max_cpu = 0
    max_ram = 0
    
    while time.time() - start_time < duration:
        try:
            # Получаем информацию о процессе
            proc_info = psutil.Process(process.pid)
            cpu_percent = proc_info.cpu_percent()
            ram_mb = proc_info.memory_info().rss / 1024 / 1024
            
            # Обновляем максимумы
            max_cpu = max(max_cpu, cpu_percent)
            max_ram = max(max_ram, ram_mb)
            
            # Выводим текущие метрики
            print(f"   CPU: {cpu_percent:.1f}%, RAM: {ram_mb:.1f} MB")
            
            time.sleep(5)  # Проверяем каждые 5 секунд
            
        except psutil.NoSuchProcess:
            print(f"   Процесс завершился")
            break
        except Exception as e:
            print(f"   Ошибка мониторинга: {e}")
            break
    
    print(f"\n📈 Максимальные ресурсы:")
    print(f"   CPU: {max_cpu:.1f}%")
    print(f"   RAM: {max_ram:.1f} MB")
    
    return max_cpu, max_ram

def main():
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ microWakeWord С РАСШИРЕННЫМИ ДАННЫМИ")
    print("=" * 60)
    
    # Проверяем наличие конфигурационного файла
    config_file = "/home/microWakeWord_data/training_parameters.yaml"
    if not os.path.exists(config_file):
        print(f"❌ Конфигурационный файл не найден: {config_file}")
        return False
    
    print(f"✅ Конфигурационный файл найден: {config_file}")
    
    # Удаляем старую модель перед началом обучения
    model_dir = "/home/microWakeWord_data/trained_models/wakeword"
    if os.path.exists(model_dir):
        print(f"🗑️ Удаление старой модели из {model_dir}")
        try:
            import shutil
            shutil.rmtree(model_dir)
            os.makedirs(model_dir, exist_ok=True)
            print(f"✅ Старая модель удалена, директория пересоздана")
        except Exception as e:
            print(f"❌ Ошибка удаления старой модели: {e}")
            return False
    else:
        print(f"📁 Создание директории для модели: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
    
    # Проверяем наличие сгенерированных данных (только те, что указаны в конфигурации)
    data_dirs = [
        "/home/microWakeWord_data/generated_features_negatives_final",
        "/home/microWakeWord_data/generated_features_positives_final",
        "/home/microWakeWord_data/generated_features_background",
        "/home/microWakeWord_data/generated_features_hard_negatives_parallel",
        "/home/microWakeWord_data/generated_features_negatives_both",
        "/home/microWakeWord_data/generated_features_positives_both",
        "/home/microWakeWord_data/generated_features_positives_enhanced",
        "/home/microWakeWord_data/generated_features_hard_negatives",
    ]
    
    print("\n📊 Проверка сгенерированных данных:")
    missing_dirs = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"   ✅ {data_dir}")
        else:
            print(f"   ❌ {data_dir}")
            missing_dirs.append(data_dir)
    
    if missing_dirs:
        print(f"\n❌ Отсутствуют директории: {len(missing_dirs)}")
        return False
    
    print(f"\n✅ Все {len(data_dirs)} директорий с данными найдены!")
    
    # Проверяем наличие библиотеки microwakeword
    microwakeword_path = "./microwakeword"
    if not os.path.exists(microwakeword_path):
        print(f"❌ Библиотека microwakeword не найдена: {microwakeword_path}")
        return False
    
    print(f"✅ Библиотека microwakeword найдена: {microwakeword_path}")
    
    # Команда для обучения модели (из mww_orig) с использованием venv
    train_command = [
        "/home/microWakeWord/.venv/bin/python", "-m", "microwakeword.model_train_eval",
        "--training_config", config_file,
        "--train", "1",
        "--restore_checkpoint", "0",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "0",
        "inception"  # Используем модель inception
    ]
    
    print(f"\n🚀 Запуск обучения модели...")
    print(f"Команда: {' '.join(train_command)}")
    
    try:
        # Запускаем обучение с логированием всех выводов
        print(f"\n🚀 Запуск обучения модели...")
        print(f"Команда: {' '.join(train_command)}")
        
        # Запускаем процесс с потоковым логированием
        process = subprocess.Popen(
            train_command,
            cwd="/home/microWakeWord",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Объединяем stderr в stdout
            text=True,
            bufsize=1,  # Небуферизованный вывод
            universal_newlines=True
        )
        
        # Потоковое логирование в реальном времени
        stdout_lines = []
        print(f"\n📄 Логи обучения в реальном времени:")
        print("=" * 50)
        
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                # Убираем лишние пробелы и добавляем перенос строки если нужно
                clean_line = line.rstrip()
                if clean_line:
                    print(clean_line)
                    stdout_lines.append(clean_line)
                
                # Проверяем завершение процесса
                if process.poll() is not None:
                    break
                    
        except Exception as e:
            print(f"❌ Ошибка чтения вывода: {e}")
        
        # Ждем завершения процесса
        returncode = process.wait(timeout=3600)  # 1 час таймаут
        
        result = type('Result', (), {
            'returncode': returncode,
            'stdout': '\n'.join(stdout_lines),
            'stderr': ''
        })()
        
        print(f"\n📊 Результат обучения:")
        print(f"Код выхода: {result.returncode}")
        
        # Логируем stdout если есть
        if result.stdout:
            print(f"\n📄 STDOUT:")
            print(result.stdout)
        
        # Логируем stderr если есть
        if result.stderr:
            print(f"\n⚠️ STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n🎉 ОБУЧЕНИЕ МОДЕЛИ ЗАВЕРШЕНО УСПЕШНО!")
            
            # Проверяем наличие обученной модели
            model_dir = "/home/microWakeWord_data/trained_models/wakeword"
            if os.path.exists(model_dir):
                print(f"✅ Обученная модель сохранена в: {model_dir}")
                
                # Показываем содержимое директории модели
                model_files = list(Path(model_dir).glob("*"))
                if model_files:
                    print(f"\n📁 Файлы модели:")
                    for model_file in model_files:
                        print(f"   📄 {model_file.name}")
                else:
                    print(f"⚠️ Директория модели пуста")
            else:
                print(f"⚠️ Директория модели не найдена: {model_dir}")
            
            return True
        else:
            print(f"\n❌ ОБУЧЕНИЕ МОДЕЛИ ЗАВЕРШИЛОСЬ С ОШИБКОЙ!")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ ОБУЧЕНИЕ ПРЕРВАНО ПО ТАЙМАУТУ (1 час)")
        return False
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ЗАПУСКЕ ОБУЧЕНИЯ: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)