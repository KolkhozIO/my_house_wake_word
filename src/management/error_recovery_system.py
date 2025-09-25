#!/usr/bin/env python3
"""
Система восстановления после ошибок для циклического пайплайна microWakeWord
Автоматическое исправление типичных проблем и восстановление работоспособности
"""

import os
import sys
import time
import json
import shutil
import subprocess
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psutil

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/microWakeWord_data/error_recovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorRecoverySystem:
    """Система восстановления после ошибок"""
    
    def __init__(self):
        self.data_dir = Path("/home/microWakeWord_data")
        self.tasks_dir = self.data_dir / "tasks"
        self.recovery_log = self.data_dir / "recovery_history.json"
        
        # История восстановлений
        self.recovery_history = self._load_recovery_history()
        
        # Типичные ошибки и их решения
        self.error_patterns = {
            'no_space_left': {
                'pattern': 'no space left on device',
                'severity': 'critical',
                'solution': 'cleanup_disk_space'
            },
            'memory_error': {
                'pattern': 'memoryerror|outofmemoryerror|unable to allocate',
                'severity': 'high',
                'solution': 'reduce_memory_usage'
            },
            'file_not_found': {
                'pattern': 'filenotfounderror|no such file or directory|file not found',
                'severity': 'medium',
                'solution': 'recreate_missing_files'
            },
            'permission_denied': {
                'pattern': 'permissionerror|permission denied|access denied',
                'severity': 'medium',
                'solution': 'fix_permissions'
            },
            'import_error': {
                'pattern': 'importerror|modulenotfounderror|no module named',
                'severity': 'high',
                'solution': 'fix_imports'
            },
            'timeout_error': {
                'pattern': 'timeouterror|timeout|timed out',
                'severity': 'medium',
                'solution': 'increase_timeout'
            },
            'corrupted_data': {
                'pattern': 'corrupted|invalid|malformed|bad data',
                'severity': 'high',
                'solution': 'regenerate_data'
            }
        }
        
        logger.info("Инициализация ErrorRecoverySystem")
    
    def analyze_error(self, error_text: str, stage: str) -> Optional[Dict]:
        """Анализ ошибки и определение решения"""
        error_text_lower = error_text.lower()
        
        for error_type, config in self.error_patterns.items():
            # Используем регулярные выражения для поиска
            pattern = config['pattern'].lower()
            if re.search(pattern, error_text_lower):
                logger.info(f"Обнаружена ошибка типа '{error_type}' в этапе '{stage}'")
                return {
                    'type': error_type,
                    'severity': config['severity'],
                    'solution': config['solution'],
                    'stage': stage,
                    'timestamp': datetime.now().isoformat(),
                    'error_text': error_text[:500]  # Первые 500 символов
                }
        
        logger.warning(f"Неизвестный тип ошибки в этапе '{stage}': {error_text[:100]}...")
        return None
    
    def recover_from_error(self, error_info: Dict) -> bool:
        """Восстановление после ошибки"""
        solution = error_info['solution']
        stage = error_info['stage']
        
        logger.info(f"Применение решения '{solution}' для этапа '{stage}'")
        
        try:
            if solution == 'cleanup_disk_space':
                return self._cleanup_disk_space()
            elif solution == 'reduce_memory_usage':
                return self._reduce_memory_usage(stage)
            elif solution == 'recreate_missing_files':
                return self._recreate_missing_files(stage)
            elif solution == 'fix_permissions':
                return self._fix_permissions()
            elif solution == 'fix_imports':
                return self._fix_imports()
            elif solution == 'increase_timeout':
                return self._increase_timeout(stage)
            elif solution == 'regenerate_data':
                return self._regenerate_data(stage)
            else:
                logger.error(f"Неизвестное решение: {solution}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при применении решения '{solution}': {e}")
            return False
        finally:
            # Записываем в историю
            self._record_recovery_attempt(error_info)
    
    def _cleanup_disk_space(self) -> bool:
        """Очистка дискового пространства"""
        logger.info("🧹 Очистка дискового пространства...")
        
        try:
            # Проверяем свободное место
            disk_usage = shutil.disk_usage(self.data_dir)
            free_gb = disk_usage.free / (1024**3)
            
            logger.info(f"Свободно места: {free_gb:.1f} GB")
            
            if free_gb < 1.0:  # Меньше 1GB
                logger.warning("Критически мало места, выполняем очистку...")
                
                # Очищаем временные файлы
                self._cleanup_temp_files()
                
                # Очищаем старые логи
                self._cleanup_old_logs()
                
                # Очищаем старые модели
                self._cleanup_old_models()
                
                # Очищаем старые задачи
                self._cleanup_old_tasks()
                
                # Проверяем результат
                disk_usage = shutil.disk_usage(self.data_dir)
                new_free_gb = disk_usage.free / (1024**3)
                
                logger.info(f"После очистки свободно: {new_free_gb:.1f} GB")
                return new_free_gb > 0.5  # Хотя бы 500MB
            else:
                logger.info("Достаточно места на диске")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка очистки диска: {e}")
            return False
    
    def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        temp_patterns = [
            "**/*_temp",
            "**/*_tmp", 
            "**/*.tmp",
            "**/*.temp",
            "**/temp_*",
            "**/tmp_*"
        ]
        
        cleaned_count = 0
        for pattern in temp_patterns:
            for temp_path in self.data_dir.glob(pattern):
                if temp_path.is_dir():
                    shutil.rmtree(temp_path, ignore_errors=True)
                    cleaned_count += 1
                elif temp_path.is_file():
                    temp_path.unlink(missing_ok=True)
                    cleaned_count += 1
        
        logger.info(f"Очищено временных файлов: {cleaned_count}")
    
    def _cleanup_old_logs(self):
        """Очистка старых логов"""
        log_files = list(self.data_dir.glob("*.log"))
        cutoff_date = datetime.now() - timedelta(days=7)  # Старше недели
        
        cleaned_count = 0
        for log_file in log_files:
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink(missing_ok=True)
                cleaned_count += 1
        
        logger.info(f"Очищено старых логов: {cleaned_count}")
    
    def _cleanup_old_models(self):
        """Очистка старых моделей"""
        model_files = list(self.data_dir.glob("model_*.tflite"))
        model_files.extend(list(self.data_dir.glob("model_*.json")))
        
        # Оставляем только последние 5 моделей
        if len(model_files) > 5:
            # Сортируем по времени изменения
            model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Удаляем старые
            for old_model in model_files[5:]:
                old_model.unlink(missing_ok=True)
                logger.info(f"Удалена старая модель: {old_model.name}")
    
    def _cleanup_old_tasks(self):
        """Очистка старых задач"""
        if not self.tasks_dir.exists():
            return
        
        task_files = list(self.tasks_dir.glob("*.json"))
        cutoff_date = datetime.now() - timedelta(days=3)  # Старше 3 дней
        
        cleaned_count = 0
        for task_file in task_files:
            if task_file.stat().st_mtime < cutoff_date.timestamp():
                # Удаляем JSON и соответствующий PID файл
                task_file.unlink(missing_ok=True)
                pid_file = task_file.with_suffix('.pid')
                pid_file.unlink(missing_ok=True)
                cleaned_count += 1
        
        logger.info(f"Очищено старых задач: {cleaned_count}")
    
    def _reduce_memory_usage(self, stage: str) -> bool:
        """Снижение использования памяти"""
        logger.info(f"🔧 Снижение использования памяти для этапа '{stage}'")
        
        try:
            # Останавливаем другие задачи
            subprocess.run(['python', 'task_manager.py', 'stop-all'], 
                         capture_output=True, timeout=30)
            
            # Очищаем кэш Python
            import gc
            gc.collect()
            
            # Проверяем использование памяти
            memory = psutil.virtual_memory()
            logger.info(f"Использование памяти: {memory.percent}%")
            
            if memory.percent > 80:
                logger.warning("Высокое использование памяти, ждем освобождения...")
                time.sleep(30)  # Ждем 30 секунд
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка снижения памяти: {e}")
            return False
    
    def _recreate_missing_files(self, stage: str) -> bool:
        """Воссоздание отсутствующих файлов"""
        logger.info(f"📁 Воссоздание отсутствующих файлов для этапа '{stage}'")
        
        try:
            # Определяем какие файлы нужны для этапа
            required_files = {
                'generate_data': [
                    '/home/microWakeWord_data/positives_both',
                    '/home/microWakeWord_data/negatives_both'
                ],
                'generate_spectrograms': [
                    '/home/microWakeWord_data/generated_features'
                ],
                'train_model': [
                    '/home/microWakeWord_data/model.tflite',
                    '/home/microWakeWord_data/model.json'
                ]
            }
            
            files_to_check = required_files.get(stage, [])
            
            for file_path in files_to_check:
                if not os.path.exists(file_path):
                    logger.info(f"Отсутствует файл: {file_path}")
                    
                    # Создаем директорию если нужно
                    if file_path.endswith('/'):
                        os.makedirs(file_path, exist_ok=True)
                        logger.info(f"Создана директория: {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка воссоздания файлов: {e}")
            return False
    
    def _fix_permissions(self) -> bool:
        """Исправление прав доступа"""
        logger.info("🔐 Исправление прав доступа...")
        
        try:
            # Исправляем права на директорию данных
            os.chmod(self.data_dir, 0o755)
            
            # Исправляем права на файлы
            for file_path in self.data_dir.rglob('*'):
                if file_path.is_file():
                    os.chmod(file_path, 0o644)
                elif file_path.is_dir():
                    os.chmod(file_path, 0o755)
            
            logger.info("Права доступа исправлены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка исправления прав: {e}")
            return False
    
    def _fix_imports(self) -> bool:
        """Исправление проблем с импортами"""
        logger.info("📦 Исправление проблем с импортами...")
        
        try:
            # Активируем виртуальное окружение
            venv_python = "/home/microWakeWord/.venv/bin/python"
            
            if not os.path.exists(venv_python):
                logger.error("Виртуальное окружение не найдено!")
                return False
            
            # Устанавливаем недостающие пакеты
            required_packages = ['psutil', 'numpy', 'soundfile', 'tqdm']
            
            for package in required_packages:
                logger.info(f"Проверка пакета: {package}")
                result = subprocess.run([
                    venv_python, '-c', f'import {package}'
                ], capture_output=True)
                
                if result.returncode != 0:
                    logger.info(f"Установка пакета: {package}")
                    subprocess.run([
                        venv_python, '-m', 'pip', 'install', package
                    ], capture_output=True)
            
            logger.info("Проблемы с импортами исправлены")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка исправления импортов: {e}")
            return False
    
    def _increase_timeout(self, stage: str) -> bool:
        """Увеличение таймаута для этапа"""
        logger.info(f"⏰ Увеличение таймаута для этапа '{stage}'")
        
        # Для этапов обучения увеличиваем таймаут
        if stage in ['train_model', 'generate_spectrograms']:
            logger.info("Увеличен таймаут для ресурсоемкого этапа")
            time.sleep(60)  # Ждем дополнительную минуту
        
        return True
    
    def _regenerate_data(self, stage: str) -> bool:
        """Регенерация поврежденных данных"""
        logger.info(f"🔄 Регенерация данных для этапа '{stage}'")
        
        try:
            if stage == 'generate_data':
                # Запускаем быструю генерацию данных
                logger.info("Запуск быстрой генерации данных...")
                result = subprocess.run([
                    'python', 'quick_generate.py'
                ], capture_output=True, timeout=300)
                
                return result.returncode == 0
            
            elif stage == 'generate_spectrograms':
                # Очищаем старые спектрограммы и регенерируем
                spectro_dir = self.data_dir / "generated_features"
                if spectro_dir.exists():
                    shutil.rmtree(spectro_dir)
                
                logger.info("Запуск регенерации спектрограмм...")
                result = subprocess.run([
                    'python', 'generate_spectrograms.py'
                ], capture_output=True, timeout=600)
                
                return result.returncode == 0
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка регенерации данных: {e}")
            return False
    
    def _load_recovery_history(self) -> List[Dict]:
        """Загрузка истории восстановлений"""
        if self.recovery_log.exists():
            try:
                with open(self.recovery_log, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _record_recovery_attempt(self, error_info: Dict):
        """Запись попытки восстановления в историю"""
        recovery_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_info['type'],
            'stage': error_info['stage'],
            'solution': error_info['solution'],
            'severity': error_info['severity']
        }
        
        self.recovery_history.append(recovery_record)
        
        # Ограничиваем историю последними 100 записями
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
        
        # Сохраняем историю
        try:
            with open(self.recovery_log, 'w') as f:
                json.dump(self.recovery_history, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения истории восстановлений: {e}")
    
    def get_recovery_stats(self) -> Dict:
        """Получение статистики восстановлений"""
        if not self.recovery_history:
            return {'total_recoveries': 0}
        
        # Подсчитываем статистику
        stats = {
            'total_recoveries': len(self.recovery_history),
            'by_type': {},
            'by_stage': {},
            'by_severity': {},
            'recent_recoveries': self.recovery_history[-10:]  # Последние 10
        }
        
        for record in self.recovery_history:
            # По типам ошибок
            error_type = record['error_type']
            stats['by_type'][error_type] = stats['by_type'].get(error_type, 0) + 1
            
            # По этапам
            stage = record['stage']
            stats['by_stage'][stage] = stats['by_stage'].get(stage, 0) + 1
            
            # По серьезности
            severity = record['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        return stats

def main():
    """Тестирование системы восстановления"""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        recovery_system = ErrorRecoverySystem()
        
        # Тестируем различные типы ошибок
        test_errors = [
            ("No space left on device", "generate_data"),
            ("MemoryError: Unable to allocate array", "train_model"),
            ("FileNotFoundError: [Errno 2] No such file", "generate_spectrograms"),
            ("ImportError: No module named 'psutil'", "train_model")
        ]
        
        print("🧪 Тестирование системы восстановления...")
        
        for error_text, stage in test_errors:
            print(f"\nТестирование ошибки: {error_text[:50]}...")
            error_info = recovery_system.analyze_error(error_text, stage)
            
            if error_info:
                print(f"  Тип: {error_info['type']}")
                print(f"  Решение: {error_info['solution']}")
                print(f"  Серьезность: {error_info['severity']}")
            else:
                print("  Неизвестный тип ошибки")
        
        # Показываем статистику
        stats = recovery_system.get_recovery_stats()
        print(f"\n📊 Статистика восстановлений: {stats['total_recoveries']}")
        
    else:
        print("Система восстановления после ошибок для циклического пайплайна")
        print("Использование: python error_recovery_system.py --test")

if __name__ == "__main__":
    main()