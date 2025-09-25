#!/usr/bin/env python3
"""
Циклический пайплайн microWakeWord с фиксированными ресурсами
Непрерывное выполнение этапов с автоматическим восстановлением после ошибок
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
from .error_recovery_system import ErrorRecoverySystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/microWakeWord_data/cyclic_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResourceManager:
    """Управление ресурсами для каждого этапа пайплайна"""
    
    def __init__(self):
        self.total_cores = mp.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Фиксированные ресурсы для каждого этапа
        self.stage_resources = {
            'generate_data': {
                'cpu_cores': min(8, self.total_cores // 4),  # 25% CPU
                'memory_gb': min(2, self.total_memory_gb // 4),  # 25% RAM
                'priority': 'high',
                'timeout_minutes': 10
            },
            'generate_spectrograms': {
                'cpu_cores': min(16, self.total_cores // 2),  # 50% CPU
                'memory_gb': min(4, self.total_memory_gb // 2),  # 50% RAM
                'priority': 'high',
                'timeout_minutes': 30
            },
            'augmentations': {
                'cpu_cores': min(4, self.total_cores // 8),  # 12.5% CPU
                'memory_gb': min(1, self.total_memory_gb // 8),  # 12.5% RAM
                'priority': 'medium',
                'timeout_minutes': 20
            },
            'balance_dataset': {
                'cpu_cores': min(4, self.total_cores // 8),  # 12.5% CPU
                'memory_gb': min(1, self.total_memory_gb // 8),  # 12.5% RAM
                'priority': 'medium',
                'timeout_minutes': 15
            },
            'train_model': {
                'cpu_cores': min(8, self.total_cores // 4),  # 25% CPU
                'memory_gb': min(3, self.total_memory_gb // 3),  # 33% RAM
                'priority': 'high',
                'timeout_minutes': 60
            }
        }
        
        logger.info(f"Инициализация ResourceManager: {self.total_cores} ядер, {self.total_memory_gb:.1f}GB RAM")
        for stage, resources in self.stage_resources.items():
            logger.info(f"  {stage}: {resources['cpu_cores']} ядер, {resources['memory_gb']}GB RAM")
    
    def get_resources(self, stage: str) -> Dict:
        """Получить ресурсы для этапа"""
        return self.stage_resources.get(stage, {
            'cpu_cores': 2,
            'memory_gb': 1,
            'priority': 'low',
            'timeout_minutes': 10
        })
    
    def check_system_resources(self) -> bool:
        """Проверить доступность системных ресурсов"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Проверяем что система не перегружена
        if cpu_percent > 90:
            logger.warning(f"Высокая загрузка CPU: {cpu_percent}%")
            return False
        
        if memory.percent > 85:
            logger.warning(f"Высокая загрузка памяти: {memory.percent}%")
            return False
        
        return True

class CyclicPipelineManager:
    """Менеджер циклического пайплайна"""
    
    def __init__(self):
        self.tasks_dir = Path("/home/microWakeWord_data/tasks")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        self.resource_manager = ResourceManager()
        self.error_recovery = ErrorRecoverySystem()
        self.running = False
        self.stages = [
            'generate_data',
            'generate_spectrograms', 
            'augmentations',
            'balance_dataset',
            'train_model'
        ]
        
        # Статистика выполнения
        self.stage_stats = {stage: {
            'success_count': 0,
            'error_count': 0,
            'last_success': None,
            'last_error': None,
            'avg_duration': 0
        } for stage in self.stages}
        
        # Настройки цикла
        self.cycle_delay = 30  # секунд между циклами
        self.max_concurrent_stages = 2  # максимум параллельных этапов
        self.error_threshold = 3  # максимум ошибок подряд для этапа
        
        logger.info("Инициализация CyclicPipelineManager")
    
    def start_cyclic_pipeline(self):
        """Запуск циклического пайплайна"""
        logger.info("🚀 Запуск циклического пайплайна microWakeWord")
        self.running = True
        
        # Обработчик сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                logger.info(f"🔄 Цикл #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Проверяем системные ресурсы
                if not self.resource_manager.check_system_resources():
                    logger.warning("Система перегружена, ждем 60 секунд...")
                    time.sleep(60)
                    continue
                
                # Выполняем этапы пайплайна
                self._execute_pipeline_cycle()
                
                # Статистика
                self._log_cycle_stats()
                
                # Ждем перед следующим циклом
                logger.info(f"⏳ Ожидание {self.cycle_delay} секунд до следующего цикла...")
                time.sleep(self.cycle_delay)
                
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки")
        finally:
            self._cleanup()
    
    def _execute_pipeline_cycle(self):
        """Выполнение одного цикла пайплайна"""
        active_stages = []
        
        for stage in self.stages:
            # Проверяем можно ли запустить этап
            if not self._can_start_stage(stage):
                continue
            
            # Проверяем лимит параллельных этапов
            if len(active_stages) >= self.max_concurrent_stages:
                logger.info(f"Достигнут лимит параллельных этапов ({self.max_concurrent_stages})")
                break
            
            # Запускаем этап
            if self._start_stage(stage):
                active_stages.append(stage)
                logger.info(f"✅ Запущен этап: {stage}")
            else:
                logger.error(f"❌ Ошибка запуска этапа: {stage}")
        
        # Мониторим активные этапы
        if active_stages:
            self._monitor_active_stages(active_stages)
    
    def _can_start_stage(self, stage: str) -> bool:
        """Проверить можно ли запустить этап"""
        # Проверяем не запущен ли уже
        if self._is_stage_running(stage):
            return False
        
        # Проверяем не слишком ли много ошибок подряд
        stats = self.stage_stats[stage]
        if stats['error_count'] >= self.error_threshold:
            logger.warning(f"Этап {stage} имеет {stats['error_count']} ошибок подряд, пропускаем")
            return False
        
        # Проверяем зависимости (упрощенная логика)
        if stage == 'generate_spectrograms' and not self._has_recent_data():
            logger.info("Нет свежих данных для генерации спектрограмм")
            return False
        
        return True
    
    def _start_stage(self, stage: str) -> bool:
        """Запуск этапа"""
        resources = self.resource_manager.get_resources(stage)
        
        # Формируем команду с учетом ресурсов
        command = self._build_stage_command(stage, resources)
        
        # Запускаем через task_manager
        try:
            result = subprocess.run([
                'python', 'task_manager.py', 'start', stage, command
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Этап {stage} запущен успешно")
                return True
            else:
                logger.error(f"Ошибка запуска этапа {stage}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Таймаут запуска этапа {stage}")
            return False
        except Exception as e:
            logger.error(f"Исключение при запуске этапа {stage}: {e}")
            return False
    
    def _build_stage_command(self, stage: str, resources: Dict) -> str:
        """Построить команду для этапа с учетом ресурсов"""
        base_commands = {
            'generate_data': 'python quick_generate.py',
            'generate_spectrograms': 'python generate_spectrograms.py',
            'augmentations': 'python apply_augmentations.py',
            'balance_dataset': 'python balance_dataset.py',
            'train_model': 'python use_original_library_correctly.py'
        }
        
        base_cmd = base_commands.get(stage, f'python {stage}.py')
        
        # Добавляем переменные окружения для ограничения ресурсов
        env_vars = [
            f'OMP_NUM_THREADS={resources["cpu_cores"]}',
            f'MKL_NUM_THREADS={resources["cpu_cores"]}',
            f'OPENBLAS_NUM_THREADS={resources["cpu_cores"]}'
        ]
        
        cmd = f'source .venv/bin/activate && {" ".join(env_vars)} {base_cmd}'
        
        return cmd
    
    def _is_stage_running(self, stage: str) -> bool:
        """Проверить запущен ли этап"""
        try:
            result = subprocess.run([
                'python', 'task_manager.py', 'status', stage
            ], capture_output=True, text=True, timeout=10)
            
            return 'running' in result.stdout.lower()
        except:
            return False
    
    def _has_recent_data(self) -> bool:
        """Проверить есть ли свежие данные"""
        data_dirs = [
            '/home/microWakeWord_data/positives_both',
            '/home/microWakeWord_data/negatives_both'
        ]
        
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                # Проверяем что директория не пустая и не слишком старая
                files = list(Path(data_dir).glob('*.wav'))
                if files:
                    # Проверяем время последнего изменения
                    latest_file = max(files, key=lambda f: f.stat().st_mtime)
                    file_age = time.time() - latest_file.stat().st_mtime
                    
                    # Если файлы свежие (менее 1 часа)
                    if file_age < 3600:
                        return True
        
        return False
    
    def _monitor_active_stages(self, active_stages: List[str]):
        """Мониторинг активных этапов"""
        logger.info(f"Мониторинг активных этапов: {', '.join(active_stages)}")
        
        # Ждем завершения этапов или таймаут
        start_time = time.time()
        timeout = 300  # 5 минут максимум на мониторинг
        
        while active_stages and (time.time() - start_time) < timeout:
            completed_stages = []
            
            for stage in active_stages:
                if not self._is_stage_running(stage):
                    completed_stages.append(stage)
                    self._handle_stage_completion(stage)
            
            # Убираем завершенные этапы
            for stage in completed_stages:
                active_stages.remove(stage)
            
            if active_stages:
                time.sleep(10)  # Проверяем каждые 10 секунд
    
    def _handle_stage_completion(self, stage: str):
        """Обработка завершения этапа"""
        logger.info(f"Этап {stage} завершен")
        
        # Обновляем статистику
        stats = self.stage_stats[stage]
        stats['last_success'] = datetime.now()
        stats['success_count'] += 1
        
        # Проверяем успешность выполнения
        if self._check_stage_success(stage):
            logger.info(f"✅ Этап {stage} выполнен успешно")
            # Сбрасываем счетчик ошибок при успехе
            stats['error_count'] = 0
        else:
            logger.error(f"❌ Этап {stage} завершился с ошибкой")
            stats['error_count'] += 1
            stats['last_error'] = datetime.now()
            
            # Пытаемся восстановиться от ошибки
            self._attempt_error_recovery(stage)
    
    def _check_stage_success(self, stage: str) -> bool:
        """Проверить успешность выполнения этапа"""
        # Проверяем наличие ожидаемых файлов/результатов
        success_indicators = {
            'generate_data': ['/home/microWakeWord_data/positives_both', '/home/microWakeWord_data/negatives_both'],
            'generate_spectrograms': ['/home/microWakeWord_data/generated_features'],
            'augmentations': ['/home/microWakeWord_data/generated_augmented_features'],
            'balance_dataset': ['/home/microWakeWord_data/balanced_dataset'],
            'train_model': ['/home/microWakeWord_data/model.tflite']
        }
        
        indicators = success_indicators.get(stage, [])
        for indicator in indicators:
            if not os.path.exists(indicator):
                return False
        
        return True
    
    def _attempt_error_recovery(self, stage: str):
        """Попытка восстановления после ошибки"""
        logger.info(f"🔧 Попытка восстановления для этапа {stage}")
        
        try:
            # Получаем логи последней задачи
            error_text = self._get_last_task_error(stage)
            
            if error_text:
                # Анализируем ошибку
                error_info = self.error_recovery.analyze_error(error_text, stage)
                
                if error_info:
                    logger.info(f"Обнаружена ошибка типа '{error_info['type']}', применяем решение '{error_info['solution']}'")
                    
                    # Пытаемся восстановиться
                    recovery_success = self.error_recovery.recover_from_error(error_info)
                    
                    if recovery_success:
                        logger.info(f"✅ Восстановление для этапа {stage} успешно")
                        # Сбрасываем счетчик ошибок
                        self.stage_stats[stage]['error_count'] = 0
                    else:
                        logger.error(f"❌ Восстановление для этапа {stage} не удалось")
                else:
                    logger.warning(f"Неизвестный тип ошибки для этапа {stage}")
            else:
                logger.warning(f"Не удалось получить текст ошибки для этапа {stage}")
                
        except Exception as e:
            logger.error(f"Ошибка при попытке восстановления для этапа {stage}: {e}")
    
    def _get_last_task_error(self, stage: str) -> Optional[str]:
        """Получить текст последней ошибки для этапа"""
        try:
            # Ищем последний лог файл для этапа
            log_files = list(self.tasks_dir.glob(f"{stage}_*.log"))
            if not log_files:
                return None
            
            # Берем самый новый
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            
            # Читаем последние строки лога
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                
            # Ищем строки с ошибками
            error_lines = []
            for line in lines[-50:]:  # Последние 50 строк
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                    error_lines.append(line.strip())
            
            return '\n'.join(error_lines) if error_lines else None
            
        except Exception as e:
            logger.error(f"Ошибка получения лога для этапа {stage}: {e}")
            return None
    
    def _log_cycle_stats(self):
        """Логирование статистики цикла"""
        logger.info("📊 Статистика этапов:")
        for stage, stats in self.stage_stats.items():
            logger.info(f"  {stage}: успехов={stats['success_count']}, ошибок={stats['error_count']}")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"Получен сигнал {signum}, останавливаем пайплайн...")
        self.running = False
    
    def _cleanup(self):
        """Очистка ресурсов"""
        logger.info("🧹 Очистка ресурсов...")
        
        # Останавливаем все активные задачи
        try:
            subprocess.run(['python', 'task_manager.py', 'stop-all'], 
                         capture_output=True, timeout=30)
        except:
            pass
        
        logger.info("✅ Циклический пайплайн остановлен")

def main():
    """Главная функция"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("""
Циклический пайплайн microWakeWord

Использование:
    python cyclic_pipeline_manager.py [--help]

Особенности:
    - Непрерывное выполнение этапов пайплайна
    - Фиксированные ресурсы для каждого этапа
    - Автоматическое восстановление после ошибок
    - Мониторинг системных ресурсов
    - Graceful shutdown по Ctrl+C

Этапы пайплайна:
    1. generate_data - Генерация данных (25% CPU, 25% RAM)
    2. generate_spectrograms - Генерация спектрограмм (50% CPU, 50% RAM)
    3. augmentations - Аугментации (12.5% CPU, 12.5% RAM)
    4. balance_dataset - Балансировка (12.5% CPU, 12.5% RAM)
    5. train_model - Обучение модели (25% CPU, 33% RAM)

Логи: /home/microWakeWord_data/cyclic_pipeline.log
        """)
        return
    
    # Проверяем что мы в правильной директории
    if not os.path.exists('task_manager.py'):
        logger.error("task_manager.py не найден! Запустите из директории microWakeWord")
        return
    
    # Создаем и запускаем менеджер
    pipeline_manager = CyclicPipelineManager()
    pipeline_manager.start_cyclic_pipeline()

if __name__ == "__main__":
    main()