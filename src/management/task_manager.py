#!/usr/bin/env python3
"""
Планировщик задач для microWakeWord проекта
Неблокирующее выполнение с мониторингом статуса
Централизованное логирование с обязательными переносами строк
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import psutil

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.path_manager import paths
from src.utils.centralized_logger import get_logger, setup_logging
from src.utils.print_replacer import enable_print_replacement, log_print

class TaskManager:
    def __init__(self):
        # Настройка централизованного логирования
        self.logger = get_logger("task_manager")
        enable_print_replacement("task_manager")
        
        self.tasks_dir = Path(f"{paths.DATA_ROOT}/tasks")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.active_tasks = {}
        
        # СТРОГО СТАТИЧЕСКИЕ КОМАНДЫ ДЛЯ ЗАДАЧ ИЗ XML
        # ЕДИНЫЙ ИСТОЧНИК ИСТИНЫ ДЛЯ ВСЕХ ЗАДАЧ
        self.task_commands = {
            "generate_data": "python src/pipeline/data_generation/generate_both_phrases.py",
            "generate_spectrograms": "python src/pipeline/data_generation/generate_spectrograms.py",
            "augmentations": "python src/pipeline/augmentation/apply_augmentations.py",
            "balance_dataset": "python src/pipeline/balancing/balance_dataset.py",
            "train_model": "python src/pipeline/training/train_model_only.py",
            "train_larger": "python src/pipeline/training/train_model_expanded.py",
            "generate_hard_negatives": "python backups/generate_hard_negatives.py",
            "generate_enhanced_positives": "python backups/generate_enhanced_positives.py",
            "generate_background": "python backups/generate_background_data.py",
            "generate_negatives_tts": "python src/pipeline/data_generation/generate_negatives_tts.py",
            "fix_sample_rate_to_16000": "python backups/fix_sample_rate_to_16000.py",
            "generate_negatives_real": "python src/pipeline/data_generation/collect_real_data.py",
            "generate_background_data": "python src/pipeline/data_generation/process_real_data.py",
            "generate_negatives_spectrograms": "python src/pipeline/data_generation/generate_negatives_spectrograms.py",
            "train_mixed_conservative": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant conservative",
            "train_mixed_moderate": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant moderate",
            "train_mixed_aggressive": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant aggressive",
            "train_mixed_extreme": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant extreme",
            "train_mixed_all": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant conservative",
            "train_mixed_fixed": "python src/pipeline/training/train_with_mixed_data_fixed.py --variant conservative",
            "train_esphome_model": "python train_esphome_model.py",
            "test_training_5000": "python test_training_5000_steps.py"
        }
        
        # ОПИСАНИЯ ЗАДАЧ ДЛЯ СПРАВКИ
        self.task_descriptions = {
            "generate_data": "Генерация TTS данных",
            "generate_spectrograms": "Генерация спектрограмм",
            "augmentations": "Аудио аугментации",
            "balance_dataset": "Балансировка датасета",
            "train_model": "Обучение стандартной модели (50KB)",
            "train_larger": "Обучение большой модели (141KB)",
            "generate_hard_negatives": "Генерация hard negatives",
            "generate_enhanced_positives": "Улучшенные позитивные данные",
            "generate_background": "Генерация фоновых данных",
            "generate_negatives_tts": "TTS негативные данные",
            "fix_sample_rate_to_16000": "Конвертация sample rate",
            "generate_negatives_real": "Генерация реальных негативных данных",
            "generate_background_data": "Генерация ambient данных",
            "generate_negatives_spectrograms": "Генерация спектрограмм для TTS негативов",
            "train_mixed_conservative": "Обучение смешанной модели (консервативный)",
            "train_mixed_moderate": "Обучение смешанной модели (умеренный)",
            "train_mixed_aggressive": "Обучение смешанной модели (агрессивный)",
            "train_mixed_extreme": "Обучение смешанной модели (экстремальный)",
            "train_mixed_all": "Обучение смешанной модели (все варианты)",
            "train_mixed_fixed": "Обучение смешанной модели (исправленная)",
            "train_esphome_model": "Обучение ESPHome модели",
            "test_training_5000": "Тестовое обучение на 5000 шагов с логированием"
        }
    
    def get_task_command(self, task_name: str) -> str:
        """Получает команду для задачи по имени"""
        if task_name in self.task_commands:
            return self.task_commands[task_name]
        else:
            raise ValueError(f"Неизвестная задача: {task_name}")
    
    def is_valid_task(self, task_name: str) -> bool:
        """Проверяет, существует ли задача"""
        return task_name in self.task_commands
    
    def get_available_tasks(self) -> List[str]:
        """Возвращает список доступных задач"""
        return list(self.task_commands.keys())
    
    def get_task_description(self, task_name: str) -> str:
        """Возвращает описание задачи"""
        return self.task_descriptions.get(task_name, "Описание недоступно")
    
    def list_tasks(self):
        """Выводит список всех доступных задач"""
        self.logger.info("📋 Доступные задачи microWakeWord")
        self.logger.info("================================")
        self.logger.info("")
        
        self.logger.info("🟢 Основные задачи:")
        main_tasks = ["generate_data", "augmentations", "balance_dataset", "train_model", "train_larger"]
        for task in main_tasks:
            if task in self.task_commands:
                self.logger.info(f"  {task:<25} - {self.task_descriptions[task]}")
        
        self.logger.info("")
        self.logger.info("🟡 Дополнительные задачи:")
        extra_tasks = ["generate_hard_negatives", "generate_enhanced_positives", "generate_background", 
                      "generate_negatives_tts", "fix_sample_rate_to_16000"]
        for task in extra_tasks:
            if task in self.task_commands:
                self.logger.info(f"  {task:<25} - {self.task_descriptions[task]}")
        
        self.logger.info("")
        self.logger.info("🔵 Новые задачи:")
        new_tasks = ["generate_negatives_real", "generate_background_data", "train_esphome_model"]
        for task in new_tasks:
            if task in self.task_commands:
                self.logger.info(f"  {task:<25} - {self.task_descriptions[task]}")
        
        self.logger.info("")
        self.logger.info("🟠 Смешанные модели:")
        mixed_tasks = ["train_mixed_conservative", "train_mixed_moderate", "train_mixed_aggressive", 
                      "train_mixed_extreme", "train_mixed_all", "train_mixed_fixed"]
        for task in mixed_tasks:
            if task in self.task_commands:
                self.logger.info(f"  {task:<25} - {self.task_descriptions[task]}")
        
        self.logger.info("")
        self.logger.info("💡 Используйте './manage_tasks.sh start <task>' для запуска задачи")
        
    def start_task(self, task_name: str, command: str = None, working_dir: str = "/home/microWakeWord") -> bool:
        """Запуск задачи в фоне"""
        import time
        
        # Если команда не передана, получаем из словаря
        if command is None:
            try:
                command = self.get_task_command(task_name)
            except ValueError as e:
                self.logger.error(f"❌ {e}")
                return False
        
        task_id = f"{task_name}_{int(time.time())}"
        task_file = self.tasks_dir / f"{task_id}.json"
        
        # Проверяем, не запущена ли уже такая задача
        if self.is_task_running(task_name):
            self.logger.error(f"❌ Задача '{task_name}' уже запущена!")
            return False
            
        # Создаем PID файл
        pid_file = self.tasks_dir / f"{task_id}.pid"
        
        # Создаем лог файл с timestamp
        timestamp = int(time.time())
        log_file = f"/home/microWakeWord/logs/{task_name}_{timestamp}.log"
        
        # Запускаем процесс с логированием
        try:
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=working_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,  # Объединяем stderr с stdout
                    preexec_fn=os.setsid  # Создаем новую группу процессов
                )
            
            # Ждем немного, чтобы Python процесс запустился
            time.sleep(3)
            
            # Находим реальный Python процесс
            real_pid = None
            try:
                # Извлекаем имя скрипта из команды
                # Команда: "python generate_spectrograms_expanded.py > generate_spectrograms_expanded.log 2>&1"
                parts = command.split()
                script_name = None
                for i, part in enumerate(parts):
                    if part == 'python' and i + 1 < len(parts):
                        script_name = parts[i + 1]
                        break
                
                if script_name and script_name.endswith('.py'):
                    script_name = script_name.split('/')[-1]
                
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['name'] == 'python' and script_name in ' '.join(proc.info['cmdline']):
                        real_pid = proc.info['pid']
                        self.logger.info(f"🔍 Найден Python процесс: PID {real_pid} для {script_name}")
                        break
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка поиска Python процесса: {e}")
            
            # Используем реальный PID если найден
            final_pid = real_pid if real_pid else process.pid
            self.logger.info(f"📊 Используем PID: {final_pid} (реальный: {real_pid}, shell: {process.pid})")
            
            # Сохраняем информацию о задаче
            task_info = {
                "task_id": task_id,
                "task_name": task_name,
                "command": command,
                "working_dir": working_dir,
                "pid": final_pid,
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "log_file": log_file
            }
            
            with open(task_file, 'w') as f:
                json.dump(task_info, f, indent=2)
                
            with open(pid_file, 'w') as f:
                f.write(str(final_pid))
                
            self.active_tasks[task_id] = process
            self.logger.info(f"✅ Задача '{task_name}' запущена (PID: {final_pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка запуска задачи '{task_name}': {e}")
            return False
    
    def stop_task(self, task_name: str) -> bool:
        """Остановка задачи"""
        task_files = list(self.tasks_dir.glob(f"{task_name}_*.json"))
        
        if not task_files:
            self.logger.error(f"❌ Задача '{task_name}' не найдена")
            return False
            
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_info = json.load(f)
                
                pid = task_info.get('pid')
                if pid and psutil.pid_exists(pid):
                    # Убиваем всю группу процессов
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    time.sleep(2)
                    
                    # Если не помогло, убиваем принудительно
                    if psutil.pid_exists(pid):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                
                # Обновляем статус
                task_info['status'] = 'stopped'
                task_info['stop_time'] = datetime.now().isoformat()
                
                with open(task_file, 'w') as f:
                    json.dump(task_info, f, indent=2)
                
                self.logger.info(f"✅ Задача '{task_name}' остановлена")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка остановки задачи '{task_name}': {e}")
                return False
    
    def get_task_status(self, task_name: str = None) -> Dict:
        """Получение статуса задач"""
        if task_name:
            task_files = list(self.tasks_dir.glob(f"{task_name}_*.json"))
        else:
            task_files = list(self.tasks_dir.glob("*.json"))
        
        tasks_status = {}
        
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_info = json.load(f)
                
                task_id = task_info['task_id']
                pid = task_info.get('pid')
                
                # Проверяем, жив ли процесс
                if pid and psutil.pid_exists(pid):
                    try:
                        process = psutil.Process(pid)
                        
                        # Дополнительная проверка - ищем процесс по имени скрипта
                        script_name = None
                        try:
                            cmdline = process.cmdline()
                            if cmdline and len(cmdline) > 1:
                                script_name = cmdline[1]  # Второй аргумент - имя скрипта
                        except:
                            pass
                        # Проверяем, что процесс действительно работает
                        # Расширяем список активных статусов процесса
                        active_statuses = ['running', 'sleeping', 'disk-sleep', 'interruptible-sleep', 'uninterruptible-sleep']
                        process_status = process.status()
                        
                        # Проверяем, что процесс действительно работает
                        # Если процесс существует и не завершился - он работает
                        if process_status in active_statuses or process_status in ['zombie', 'stopped']:
                            task_info['status'] = 'running'
                            # Исправляем проблему с cpu_percent - даем время для накопления статистики
                            cpu_percent = process.cpu_percent(interval=0.1)
                            task_info['cpu_percent'] = cpu_percent if cpu_percent > 0 else process.cpu_percent()
                            task_info['memory_mb'] = process.memory_info().rss / 1024 / 1024
                            task_info['memory_percent'] = (process.memory_info().rss / 1024 / 1024) / (psutil.virtual_memory().total / 1024 / 1024) * 100
                        else:
                            # Дополнительная проверка - если процесс существует, но статус неактивный
                            # Проверяем код выхода процесса
                            try:
                                return_code = process.returncode
                                if return_code is not None and return_code != 0:
                                    task_info['status'] = 'failed'
                                    task_info['exit_code'] = return_code
                                else:
                                    # Проверяем наличие дочерних процессов
                                    children = process.children(recursive=True)
                                    if children:
                                        task_info['status'] = 'running'
                                        cpu_percent = process.cpu_percent(interval=0.1)
                                        task_info['cpu_percent'] = cpu_percent if cpu_percent > 0 else process.cpu_percent()
                                        task_info['memory_mb'] = process.memory_info().rss / 1024 / 1024
                                        task_info['memory_percent'] = (process.memory_info().rss / 1024 / 1024) / (psutil.virtual_memory().total / 1024 / 1024) * 100
                                    else:
                                        task_info['status'] = 'finished'
                            except:
                                task_info['status'] = 'finished'
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        task_info['status'] = 'finished'
                else:
                    # Если PID не существует, проверяем не запущен ли процесс с тем же именем
                    try:
                        # Ищем процесс по имени команды
                        found_process = False
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            try:
                                cmdline = proc.info['cmdline']
                                if cmdline and len(cmdline) > 1:
                                    # Проверяем, содержит ли команда нужный скрипт
                                    if any(script_name in ' '.join(cmdline) for script_name in ['microwakeword', 'model_train_eval', 'python']):
                                        if any(task_info['command'].split()[-1] in cmd for cmd in cmdline):
                                            task_info['status'] = 'running'
                                            task_info['pid'] = proc.info['pid']
                                            cpu_percent = proc.cpu_percent(interval=0.1)
                                            task_info['cpu_percent'] = cpu_percent if cpu_percent > 0 else proc.cpu_percent()
                                            task_info['memory_mb'] = proc.memory_info().rss / 1024 / 1024
                                            task_info['memory_percent'] = (proc.memory_info().rss / 1024 / 1024) / (psutil.virtual_memory().total / 1024 / 1024) * 100
                                            found_process = True
                                            break
                            except:
                                continue
                        
                        if not found_process:
                            task_info['status'] = 'finished'
                    except Exception:
                        task_info['status'] = 'finished'
                
                tasks_status[task_id] = task_info
                
            except Exception as e:
                print(f"❌ Ошибка чтения файла {task_file}: {e}")
        
        return tasks_status
    
    def is_task_running(self, task_name: str) -> bool:
        """Проверка, запущена ли задача"""
        status = self.get_task_status(task_name)
        return any(task['status'] == 'running' for task in status.values())
    
    def cleanup_finished_tasks(self):
        """Очищает завершенные задачи"""
        task_files = list(self.tasks_dir.glob("*.json"))
        
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_info = json.load(f)
                
                # Удаляем файлы завершенных задач
                if task_info.get('status') in ['finished', 'failed', 'stopped']:
                    task_file.unlink()
                    # Удаляем соответствующий PID файл
                    pid_file = self.tasks_dir / f"{task_info['task_id']}.pid"
                    if pid_file.exists():
                        pid_file.unlink()
                        
            except Exception as e:
                self.logger.error(f"❌ Ошибка очистки {task_file}: {e}")
    
    def get_task_logs(self, task_name: str, lines: int = 50) -> str:
        """Получение логов задачи"""
        task_files = list(self.tasks_dir.glob(f"{task_name}_*.json"))
        
        if not task_files:
            return f"❌ Задача '{task_name}' не найдена"
        
        # Берем последний файл задачи
        task_file = max(task_files, key=os.path.getctime)
        
        try:
            with open(task_file, 'r') as f:
                task_info = json.load(f)
            
            pid = task_info.get('pid')
            if not pid or not psutil.pid_exists(pid):
                return f"❌ Процесс задачи '{task_name}' не найден"
            
            # Используем сохраненный путь к лог файлу
            log_file = task_info.get('log_file')
            if log_file and os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        log_lines = f.readlines()
                    # Возвращаем последние N строк
                    return ''.join(log_lines[-lines:])
                except Exception as e:
                    return f"❌ Ошибка чтения лог файла: {e}"
            
            # Если нет лог файла, ищем по паттерну
            logs_dir = Path("/home/microWakeWord/logs")
            log_files = list(logs_dir.glob(f"{task_name}_*.log"))
            if log_files:
                # Берем последний лог файл
                log_file = max(log_files, key=os.path.getctime)
                try:
                    with open(log_file, 'r') as f:
                        log_lines = f.readlines()
                    return ''.join(log_lines[-lines:])
                except Exception as e:
                    return f"❌ Ошибка чтения лог файла: {e}"
            
            return f"❌ Лог файл для задачи '{task_name}' не найден"
                
        except Exception as e:
            return f"❌ Ошибка получения логов: {e}"
    
    def cleanup_finished_tasks(self):
        """Очистка завершенных задач"""
        task_files = list(self.tasks_dir.glob("*.json"))
        
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_info = json.load(f)
                
                pid = task_info.get('pid')
                if not pid or not psutil.pid_exists(pid):
                    # Удаляем файлы завершенных задач
                    pid_file = self.tasks_dir / f"{task_info['task_id']}.pid"
                    if pid_file.exists():
                        pid_file.unlink()
                    task_file.unlink()
                    
            except Exception as e:
                self.logger.error(f"❌ Ошибка очистки {task_file}: {e}")

def main():
    # Настройка централизованного логирования
    setup_logging()
    logger = get_logger("main")
    
    manager = TaskManager()
    
    if len(sys.argv) < 2:
        logger.info("Использование: python task_manager.py <команда> [параметры]")
        logger.info("Команды:")
        logger.info("  start <имя> <команда> [рабочая_директория] - запуск задачи")
        logger.info("  stop <имя> - остановка задачи")
        logger.info("  status [имя] - статус задач")
        logger.info("  logs <имя> [строк] - логи задачи")
        logger.info("  list - список доступных задач")
        logger.info("  validate <имя> - проверка существования задачи")
        logger.info("  cleanup - очистка завершенных задач")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        if len(sys.argv) < 3:
            logger.error("❌ Использование: start <имя> [команда] [рабочая_директория]")
            return
        
        task_name = sys.argv[2]
        cmd = sys.argv[3] if len(sys.argv) > 3 else None
        working_dir = sys.argv[4] if len(sys.argv) > 4 else "/home/microWakeWord"
        
        manager.start_task(task_name, cmd, working_dir)
        
    elif command == "stop":
        if len(sys.argv) < 3:
            logger.error("❌ Использование: stop <имя>")
            return
        
        task_name = sys.argv[2]
        manager.stop_task(task_name)
        
    elif command == "status":
        task_name = sys.argv[2] if len(sys.argv) > 2 else None
        status = manager.get_task_status(task_name)
        
        if not status:
            logger.info("📋 Нет активных задач")
            return
        
        logger.info("📋 Статус задач:")
        for task_id, task_info in status.items():
            if task_info['status'] == 'running':
                status_icon = "🟢"
            elif task_info['status'] == 'failed':
                status_icon = "🔴"
            else:
                status_icon = "🔴"
            
            logger.info(f"{status_icon} {task_info['task_name']} ({task_id})")
            logger.info(f"   Команда: {task_info['command']}")
            logger.info(f"   PID: {task_info.get('pid', 'N/A')}")
            logger.info(f"   Статус: {task_info['status']}")
            if task_info.get('exit_code'):
                logger.info(f"   Код выхода: {task_info['exit_code']}")
            logger.info(f"   Запуск: {task_info['start_time']}")
            
            if task_info['status'] == 'running':
                logger.info(f"   CPU: {task_info.get('cpu_percent', 0):.1f}%")
                logger.info(f"   RAM: {task_info.get('memory_mb', 0):.1f} MB ({task_info.get('memory_percent', 0):.1f}%)")
            logger.info("")
            
    elif command == "logs":
        if len(sys.argv) < 3:
            logger.error("❌ Использование: logs <имя> [строк]")
            return
        
        task_name = sys.argv[2]
        lines = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        
        logs = manager.get_task_logs(task_name, lines)
        logger.info(logs)
        
    elif command == "cleanup":
        manager.cleanup_finished_tasks()
        logger.info("✅ Завершенные задачи очищены")
        
    elif command == "stop-all":
        # Останавливаем все активные задачи
        status = manager.get_task_status()
        stopped_count = 0
        for task_id, task_info in status.items():
            if task_info['status'] == 'running':
                manager.stop_task(task_info['task_name'])
                stopped_count += 1
        logger.info(f"✅ Остановлено {stopped_count} задач")
        
    elif command == "list":
        manager.list_tasks()
        
    elif command == "validate":
        if len(sys.argv) < 3:
            logger.error("❌ Использование: validate <имя>")
            return
        
        task_name = sys.argv[2]
        if manager.is_valid_task(task_name):
            logger.info(f"✅ Задача '{task_name}' найдена")
            logger.info(f"   Описание: {manager.get_task_description(task_name)}")
            logger.info(f"   Команда: {manager.get_task_command(task_name)}")
        else:
            logger.error(f"❌ Неизвестная задача: {task_name}")
            logger.info("💡 Используйте 'list' для просмотра доступных задач")
        
    else:
        logger.error(f"❌ Неизвестная команда: {command}")

if __name__ == "__main__":
    main()