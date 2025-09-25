#!/usr/bin/env python3
"""
Параллельный пайплайн microWakeWord - все этапы работают параллельно по кругу
Каждый этап работает в своем потоке с независимым циклом выполнения
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
import logging
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from .error_recovery_system import ErrorRecoverySystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/microWakeWord_data/parallel_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StageWorker:
    """Воркер для отдельного этапа пайплайна"""
    
    def __init__(self, stage_name: str, config: Dict, shared_state: Dict, error_recovery: ErrorRecoverySystem):
        self.stage_name = stage_name
        self.config = config
        self.shared_state = shared_state
        self.error_recovery = error_recovery
        self.running = False
        self.thread = None
        
        # Статистика этапа
        self.stats = {
            'cycles_completed': 0,
            'success_count': 0,
            'error_count': 0,
            'last_success': None,
            'last_error': None,
            'avg_duration': 0,
            'current_status': 'stopped'
        }
        
        # Очередь команд для этапа
        self.command_queue = queue.Queue()
        
        logger.info(f"Инициализация воркера для этапа: {stage_name}")
    
    def start(self):
        """Запуск воркера"""
        if self.running:
            logger.warning(f"Воркер {self.stage_name} уже запущен")
            return
        
        self.running = True
        self.stats['current_status'] = 'starting'
        
        # Запускаем поток
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"🚀 Воркер {self.stage_name} запущен")
    
    def stop(self):
        """Остановка воркера"""
        if not self.running:
            return
        
        logger.info(f"🛑 Остановка воркера {self.stage_name}")
        self.running = False
        self.stats['current_status'] = 'stopping'
        
        # Ждем завершения потока с коротким таймаутом
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)  # Уменьшили таймаут до 5 секунд
            
            # Если поток не остановился, принудительно завершаем
            if self.thread.is_alive():
                logger.warning(f"⚠️ Воркер {self.stage_name} не остановился за 5 секунд, принудительное завершение")
        
        self.stats['current_status'] = 'stopped'
        logger.info(f"✅ Воркер {self.stage_name} остановлен")
    
    def _worker_loop(self):
        """Основной цикл воркера"""
        logger.info(f"🔄 Начало цикла воркера {self.stage_name}")
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                logger.info(f"🔄 {self.stage_name} - Цикл #{cycle_count}")
                
                # Проверяем можно ли выполнить этап
                if not self._can_execute_stage():
                    logger.info(f"⏳ {self.stage_name} - Ожидание условий для выполнения")
                    
                    # Разбиваем ожидание на короткие интервалы для быстрой остановки
                    check_interval = self.config['check_interval']
                    for _ in range(check_interval):
                        if not self.running:
                            break
                        time.sleep(1)
                    continue
                
                # Выполняем этап
                start_time = time.time()
                success = self._execute_stage()
                duration = time.time() - start_time
                
                # Обновляем статистику
                self.stats['cycles_completed'] += 1
                self.stats['avg_duration'] = (
                    (self.stats['avg_duration'] * (self.stats['cycles_completed'] - 1) + duration) 
                    / self.stats['cycles_completed']
                )
                
                if success:
                    self.stats['success_count'] += 1
                    self.stats['last_success'] = datetime.now()
                    self.stats['error_count'] = 0  # Сбрасываем счетчик ошибок
                    logger.info(f"✅ {self.stage_name} - Цикл #{cycle_count} выполнен успешно ({duration:.1f}с)")
                else:
                    self.stats['error_count'] += 1
                    self.stats['last_error'] = datetime.now()
                    logger.error(f"❌ {self.stage_name} - Цикл #{cycle_count} завершился с ошибкой")
                    
                    # Пытаемся восстановиться
                    self._attempt_recovery()
                
                # Ждем перед следующим циклом с проверкой остановки
                sleep_time = self.config['cycle_interval']
                logger.info(f"⏳ {self.stage_name} - Ожидание {sleep_time}с до следующего цикла")
                
                # Разбиваем ожидание на короткие интервалы для быстрой остановки
                for _ in range(sleep_time):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {self.stage_name} - Ошибка в цикле воркера: {e}")
                self.stats['error_count'] += 1
                self.stats['last_error'] = datetime.now()
                time.sleep(30)  # Ждем 30 секунд при критической ошибке
    
    def _can_execute_stage(self) -> bool:
        """Проверить можно ли выполнить этап"""
        # Проверяем зависимости
        dependencies = self.config.get('dependencies', [])
        for dep in dependencies:
            if not self._check_dependency(dep):
                return False
        
        # Проверяем системные ресурсы
        if not self._check_resources():
            return False
        
        # Проверяем не запущен ли уже этот этап
        if self._is_stage_running():
            return False
        
        return True
    
    def _check_dependency(self, dependency: str) -> bool:
        """Проверить зависимость"""
        if dependency == 'fresh_data':
            # Проверяем есть ли свежие данные
            return self._has_fresh_data()
        elif dependency == 'spectrograms_ready':
            # Проверяем готовы ли спектрограммы
            return self._has_spectrograms()
        elif dependency == 'model_ready':
            # Проверяем готова ли модель
            return self._has_model()
        
        return True
    
    def _check_resources(self) -> bool:
        """Проверить системные ресурсы"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Проверяем что система не перегружена
        max_cpu = self.config.get('max_cpu_percent', 90)
        max_memory = self.config.get('max_memory_percent', 85)
        
        if cpu_percent > max_cpu:
            logger.warning(f"{self.stage_name} - Высокая загрузка CPU: {cpu_percent}%")
            return False
        
        if memory.percent > max_memory:
            logger.warning(f"{self.stage_name} - Высокая загрузка памяти: {memory.percent}%")
            return False
        
        return True
    
    def _is_stage_running(self) -> bool:
        """Проверить запущен ли этап"""
        try:
            result = subprocess.run([
                'python', 'task_manager.py', 'status', self.stage_name
            ], capture_output=True, text=True, timeout=10)
            
            return 'running' in result.stdout.lower()
        except:
            return False
    
    def _execute_stage(self) -> bool:
        """Выполнить этап"""
        logger.info(f"🎯 {self.stage_name} - Выполнение этапа")
        
        try:
            # Формируем команду
            command = self._build_command()
            
            # Запускаем через task_manager
            result = subprocess.run([
                'python', 'task_manager.py', 'start', self.stage_name, command
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"{self.stage_name} - Ошибка запуска: {result.stderr}")
                return False
            
            # Ждем завершения
            success = self._wait_for_completion()
            
            if success:
                logger.info(f"✅ {self.stage_name} - Этап выполнен успешно")
            else:
                logger.error(f"❌ {self.stage_name} - Этап завершился с ошибкой")
            
            return success
            
        except Exception as e:
            logger.error(f"{self.stage_name} - Ошибка выполнения этапа: {e}")
            return False
    
    def _build_command(self) -> str:
        """Построить команду для этапа"""
        base_commands = {
            'generate_data': 'python quick_generate.py',
            'generate_spectrograms': 'python generate_spectrograms.py',
            'augmentations': 'python apply_augmentations.py',
            'balance_dataset': 'python balance_dataset.py',
            'train_model': 'python use_original_library_correctly.py'
        }
        
        base_cmd = base_commands.get(self.stage_name, f'python {self.stage_name}.py')
        
        # Добавляем переменные окружения для ограничения ресурсов
        cpu_cores = self.config.get('cpu_cores', 2)
        env_vars = [
            f'OMP_NUM_THREADS={cpu_cores}',
            f'MKL_NUM_THREADS={cpu_cores}',
            f'OPENBLAS_NUM_THREADS={cpu_cores}'
        ]
        
        cmd = f'source .venv/bin/activate && {" ".join(env_vars)} {base_cmd}'
        
        return cmd
    
    def _wait_for_completion(self) -> bool:
        """Ждать завершения этапа"""
        timeout = self.config.get('timeout_minutes', 30) * 60  # в секундах
        check_interval = 10  # проверяем каждые 10 секунд
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if not self._is_stage_running():
                # Этап завершился, проверяем успешность
                return self._check_stage_success()
            
            time.sleep(check_interval)
        
        # Таймаут
        logger.warning(f"{self.stage_name} - Таймаут выполнения ({timeout}с)")
        return False
    
    def _check_stage_success(self) -> bool:
        """Проверить успешность выполнения этапа"""
        success_indicators = {
            'generate_data': [
                '/home/microWakeWord_data/positives_both',
                '/home/microWakeWord_data/negatives_both'
            ],
            'generate_spectrograms': [
                '/home/microWakeWord_data/generated_features'
            ],
            'augmentations': [
                '/home/microWakeWord_data/generated_augmented_features'
            ],
            'balance_dataset': [
                '/home/microWakeWord_data/balanced_dataset'
            ],
            'train_model': [
                '/home/microWakeWord_data/model.tflite'
            ]
        }
        
        indicators = success_indicators.get(self.stage_name, [])
        for indicator in indicators:
            if not os.path.exists(indicator):
                return False
        
        return True
    
    def _attempt_recovery(self):
        """Попытка восстановления после ошибки"""
        logger.info(f"🔧 {self.stage_name} - Попытка восстановления")
        
        try:
            # Получаем логи последней задачи
            error_text = self._get_last_task_error()
            
            if error_text:
                # Анализируем ошибку
                error_info = self.error_recovery.analyze_error(error_text, self.stage_name)
                
                if error_info:
                    logger.info(f"{self.stage_name} - Обнаружена ошибка типа '{error_info['type']}', применяем решение '{error_info['solution']}'")
                    
                    # Пытаемся восстановиться
                    recovery_success = self.error_recovery.recover_from_error(error_info)
                    
                    if recovery_success:
                        logger.info(f"✅ {self.stage_name} - Восстановление успешно")
                        self.stats['error_count'] = 0
                    else:
                        logger.error(f"❌ {self.stage_name} - Восстановление не удалось")
                else:
                    logger.warning(f"{self.stage_name} - Неизвестный тип ошибки")
            
        except Exception as e:
            logger.error(f"{self.stage_name} - Ошибка при восстановлении: {e}")
    
    def _get_last_task_error(self) -> Optional[str]:
        """Получить текст последней ошибки"""
        try:
            tasks_dir = Path("/home/microWakeWord_data/tasks")
            log_files = list(tasks_dir.glob(f"{self.stage_name}_*.log"))
            
            if not log_files:
                return None
            
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_log, 'r') as f:
                lines = f.readlines()
            
            error_lines = []
            for line in lines[-50:]:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                    error_lines.append(line.strip())
            
            return '\n'.join(error_lines) if error_lines else None
            
        except Exception as e:
            logger.error(f"{self.stage_name} - Ошибка получения лога: {e}")
            return None
    
    def _has_fresh_data(self) -> bool:
        """Проверить есть ли свежие данные"""
        data_dirs = [
            '/home/microWakeWord_data/positives_both',
            '/home/microWakeWord_data/negatives_both'
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = list(Path(data_dir).glob('*.wav'))
                if files:
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    file_age = time.time() - latest_file.stat().st_mtime
                    if file_age < 3600:  # Менее часа
                        return True
        
        return False
    
    def _has_spectrograms(self) -> bool:
        """Проверить есть ли спектрограммы"""
        spectro_dir = Path('/home/microWakeWord_data/generated_features')
        return spectro_dir.exists() and any(spectro_dir.iterdir())
    
    def _has_model(self) -> bool:
        """Проверить есть ли модель"""
        model_file = Path('/home/microWakeWord_data/model.tflite')
        return model_file.exists()
    
    def get_stats(self) -> Dict:
        """Получить статистику воркера"""
        return {
            'stage_name': self.stage_name,
            'current_status': self.stats['current_status'],
            'cycles_completed': self.stats['cycles_completed'],
            'success_count': self.stats['success_count'],
            'error_count': self.stats['error_count'],
            'last_success': self.stats['last_success'],
            'last_error': self.stats['last_error'],
            'avg_duration': self.stats['avg_duration']
        }

class ParallelPipelineManager:
    """Менеджер параллельного пайплайна"""
    
    def __init__(self):
        self.running = False
        self.workers = {}
        self.error_recovery = ErrorRecoverySystem()
        
        # Конфигурация этапов
        self.stage_configs = {
            'generate_data': {
                'cpu_cores': 8,
                'memory_gb': 2,
                'cycle_interval': 300,  # 5 минут
                'check_interval': 30,   # 30 секунд
                'timeout_minutes': 10,
                'max_cpu_percent': 80,
                'max_memory_percent': 80,
                'dependencies': []
            },
            'generate_spectrograms': {
                'cpu_cores': 16,
                'memory_gb': 4,
                'cycle_interval': 600,  # 10 минут
                'check_interval': 60,   # 1 минута
                'timeout_minutes': 30,
                'max_cpu_percent': 85,
                'max_memory_percent': 85,
                'dependencies': ['fresh_data']
            },
            'augmentations': {
                'cpu_cores': 4,
                'memory_gb': 1,
                'cycle_interval': 900,  # 15 минут
                'check_interval': 60,   # 1 минута
                'timeout_minutes': 20,
                'max_cpu_percent': 70,
                'max_memory_percent': 70,
                'dependencies': ['spectrograms_ready']
            },
            'balance_dataset': {
                'cpu_cores': 4,
                'memory_gb': 1,
                'cycle_interval': 1200, # 20 минут
                'check_interval': 60,   # 1 минута
                'timeout_minutes': 15,
                'max_cpu_percent': 70,
                'max_memory_percent': 70,
                'dependencies': ['spectrograms_ready']
            },
            'train_model': {
                'cpu_cores': 8,
                'memory_gb': 3,
                'cycle_interval': 1800, # 30 минут
                'check_interval': 120,   # 2 минуты
                'timeout_minutes': 60,
                'max_cpu_percent': 90,
                'max_memory_percent': 90,
                'dependencies': ['spectrograms_ready']
            }
        }
        
        logger.info("Инициализация ParallelPipelineManager")
    
    def start_parallel_pipeline(self):
        """Запуск параллельного пайплайна"""
        logger.info("🚀 Запуск параллельного пайплайна microWakeWord")
        self.running = True
        
        # Обработчик сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Создаем и запускаем воркеры
            for stage_name, config in self.stage_configs.items():
                worker = StageWorker(stage_name, config, {}, self.error_recovery)
                worker.start()
                self.workers[stage_name] = worker
            
            logger.info(f"✅ Запущено {len(self.workers)} воркеров")
            
            # Мониторинг
            self._monitor_workers()
            
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
        finally:
            self._cleanup()
    
    def _monitor_workers(self):
        """Мониторинг воркеров"""
        logger.info("📊 Начало мониторинга воркеров")
        
        while self.running:
            try:
                # Логируем статистику каждые 5 минут
                self._log_workers_stats()
                
                # Проверяем состояние воркеров
                self._check_workers_health()
                
                # Ждем 5 минут
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(60)
    
    def _log_workers_stats(self):
        """Логирование статистики воркеров"""
        logger.info("📊 Статистика воркеров:")
        
        for stage_name, worker in self.workers.items():
            stats = worker.get_stats()
            logger.info(f"  {stage_name}: статус={stats['current_status']}, "
                       f"циклов={stats['cycles_completed']}, "
                       f"успехов={stats['success_count']}, "
                       f"ошибок={stats['error_count']}")
    
    def _check_workers_health(self):
        """Проверка здоровья воркеров"""
        for stage_name, worker in self.workers.items():
            if not worker.thread.is_alive():
                logger.error(f"❌ Воркер {stage_name} остановился неожиданно!")
                
                # Перезапускаем воркер
                logger.info(f"🔄 Перезапуск воркера {stage_name}")
                worker.start()
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов"""
        logger.info(f"Получен сигнал {signum}, останавливаем пайплайн...")
        self.running = False
    
    def _cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Очистка ресурсов...")
        
        # Останавливаем всех воркеров
        for stage_name, worker in self.workers.items():
            logger.info(f"Остановка воркера {stage_name}")
            worker.stop()
        
        # Останавливаем все задачи
        try:
            subprocess.run(['python', 'task_manager.py', 'stop-all'], 
                         capture_output=True, timeout=30)
        except:
            pass
        
        logger.info("✅ Параллельный пайплайн остановлен")
    
    def get_overall_stats(self) -> Dict:
        """Получить общую статистику"""
        total_cycles = sum(worker.stats['cycles_completed'] for worker in self.workers.values())
        total_successes = sum(worker.stats['success_count'] for worker in self.workers.values())
        total_errors = sum(worker.stats['error_count'] for worker in self.workers.values())
        
        return {
            'total_cycles': total_cycles,
            'total_successes': total_successes,
            'total_errors': total_errors,
            'workers_count': len(self.workers),
            'running_workers': sum(1 for worker in self.workers.values() if worker.running)
        }

def main():
    """Главная функция"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
Параллельный пайплайн microWakeWord

Использование:
    python parallel_pipeline_manager.py [--help]

Особенности:
    - Все этапы работают параллельно и непрерывно
    - Каждый этап работает в своем потоке
    - Независимые циклы выполнения для каждого этапа
    - Автоматическое восстановление после ошибок
    - Мониторинг системных ресурсов
    - Graceful shutdown по Ctrl+C

Этапы пайплайна (все работают параллельно):
    1. generate_data - Генерация данных (каждые 5 мин)
    2. generate_spectrograms - Генерация спектрограмм (каждые 10 мин)
    3. augmentations - Аугментации (каждые 15 мин)
    4. balance_dataset - Балансировка (каждые 20 мин)
    5. train_model - Обучение модели (каждые 30 мин)

Логи: /home/microWakeWord_data/parallel_pipeline.log
        """)
        return
    
    # Проверяем что мы в правильной директории
    if not os.path.exists('task_manager.py'):
        logger.error("task_manager.py не найден! Запустите из директории microWakeWord")
        return
    
    # Создаем и запускаем менеджер
    pipeline_manager = ParallelPipelineManager()
    pipeline_manager.start_parallel_pipeline()

if __name__ == "__main__":
    main()