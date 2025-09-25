#!/usr/bin/env python3
"""
Централизованная система логирования для microWakeWord пайплайна
С обязательными переносами строк и структурированным выводом
"""

import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
import threading
from typing import Optional, Dict, Any

class FlushStreamHandler(logging.StreamHandler):
    """StreamHandler с принудительным flush и переносами строк"""
    def emit(self, record):
        super().emit(record)
        # Принудительный перенос строки
        self.stream.write('\n')
        self.flush()

class NewlineFormatter(logging.Formatter):
    """Форматтер с обязательными переносами строк"""
    
    def format(self, record):
        # Получаем отформатированное сообщение
        formatted = super().format(record)
        
        # Убираем существующие переносы строк в конце
        formatted = formatted.rstrip('\n\r')
        
        # Добавляем обязательный перенос строки
        return formatted + '\n'

class CentralizedLogger:
    """Централизованный логгер для всего пайплайна с обязательными переносами строк"""
    
    def __init__(self, name="microWakeWord", log_dir="/home/microWakeWord/logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Очищаем существующие обработчики
        self.logger.handlers.clear()
        
        # Настраиваем форматирование с переносами строк
        self.formatter = NewlineFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Настраиваем обработчики
        self._setup_handlers()
        
        # Блокировка для потокобезопасности
        self._lock = threading.Lock()
        
    def _setup_handlers(self):
        """Настройка обработчиков логов с обязательными переносами строк"""
        
        # 1. Консольный вывод (INFO и выше) с переносами строк
        console_handler = FlushStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
        # 2. Основной файл логов (DEBUG и выше) с переносами строк
        main_log_file = self.log_dir / f"{self.name}_main.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # 3. Файл ошибок (ERROR и выше) с переносами строк
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
        # 4. JSON лог для структурированных данных
        json_log_file = self.log_dir / f"{self.name}_structured.json"
        self.json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        self.json_handler.setLevel(logging.INFO)
        
        # JSON форматтер с переносами строк
        json_formatter = JsonFormatter()
        self.json_handler.setFormatter(json_formatter)
        self.logger.addHandler(self.json_handler)
        
    def get_logger(self, component_name=None):
        """Получить логгер для конкретного компонента"""
        if component_name:
            return self.logger.getChild(component_name)
        return self.logger
    
    def info(self, message: str, **kwargs):
        """Логирование INFO с переносом строки"""
        with self._lock:
            self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Логирование DEBUG с переносом строки"""
        with self._lock:
            self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Логирование WARNING с переносом строки"""
        with self._lock:
            self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Логирование ERROR с переносом строки"""
        with self._lock:
            self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Логирование CRITICAL с переносом строки"""
        with self._lock:
            self.logger.critical(message, **kwargs)
    
    def log_task_start(self, task_name: str, command: str, pid: Optional[int] = None):
        """Логирование начала задачи с переносами строк"""
        self.info(f"🚀 ЗАДАЧА ЗАПУЩЕНА: {task_name}")
        self.info(f"   Команда: {command}")
        if pid:
            self.info(f"   PID: {pid}")
        
        # Структурированное логирование
        self._log_structured({
            "event": "task_start",
            "task_name": task_name,
            "command": command,
            "pid": pid,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_task_end(self, task_name: str, return_code: int, duration: Optional[float] = None):
        """Логирование завершения задачи с переносами строк"""
        status = "✅ УСПЕШНО" if return_code == 0 else "❌ ОШИБКА"
        self.info(f"🏁 ЗАДАЧА ЗАВЕРШЕНА: {task_name} - {status}")
        self.info(f"   Код выхода: {return_code}")
        if duration:
            self.info(f"   Время выполнения: {duration:.2f} секунд")
        
        # Структурированное логирование
        self._log_structured({
            "event": "task_end",
            "task_name": task_name,
            "return_code": return_code,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_training_progress(self, epoch: int, batch: int, accuracy: float, loss: float, 
                            recall: Optional[float] = None, precision: Optional[float] = None):
        """Логирование прогресса обучения с переносами строк"""
        metrics = f"Epoch: {epoch}, Batch: {batch}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
        if recall:
            metrics += f", Recall: {recall:.4f}"
        if precision:
            metrics += f", Precision: {precision:.4f}"
        
        self.info(f"📊 ОБУЧЕНИЕ: {metrics}")
        
        # Структурированное логирование
        self._log_structured({
            "event": "training_progress",
            "epoch": epoch,
            "batch": batch,
            "accuracy": accuracy,
            "loss": loss,
            "recall": recall,
            "precision": precision,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_resource_usage(self, cpu_percent: float, ram_mb: float, process_name: Optional[str] = None):
        """Логирование использования ресурсов с переносами строк"""
        self.info(f"💻 РЕСУРСЫ: CPU: {cpu_percent:.1f}%, RAM: {ram_mb:.1f} MB")
        if process_name:
            self.info(f"   Процесс: {process_name}")
        
        # Структурированное логирование
        self._log_structured({
            "event": "resource_usage",
            "cpu_percent": cpu_percent,
            "ram_mb": ram_mb,
            "process_name": process_name,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_data_generation(self, source_name: str, files_count: int, duration: Optional[float] = None):
        """Логирование генерации данных с переносами строк"""
        self.info(f"📁 ДАННЫЕ: {source_name} - {files_count} файлов")
        if duration:
            self.info(f"   Время генерации: {duration:.2f} секунд")
        
        # Структурированное логирование
        self._log_structured({
            "event": "data_generation",
            "source_name": source_name,
            "files_count": files_count,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_error(self, error_msg: str, component: Optional[str] = None, exception: Optional[Exception] = None):
        """Логирование ошибок с переносами строк"""
        if component:
            self.error(f"❌ ОШИБКА в {component}: {error_msg}")
        else:
            self.error(f"❌ ОШИБКА: {error_msg}")
        
        if exception:
            self.logger.exception("Детали ошибки:")
        
        # Структурированное логирование
        self._log_structured({
            "event": "error",
            "component": component,
            "error_msg": error_msg,
            "exception": str(exception) if exception else None,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_subprocess_output(self, process_name: str, output: str, is_error: bool = False):
        """Логирование вывода subprocess с переносами строк"""
        level = self.error if is_error else self.info
        prefix = "❌ ОШИБКА" if is_error else "📄 ВЫВОД"
        
        # Разбиваем вывод на строки и логируем каждую
        lines = output.strip().split('\n')
        for line in lines:
            if line.strip():  # Только непустые строки
                level(f"{prefix} [{process_name}]: {line.strip()}")
    
    def log_process_status(self, task_name: str, status: str, details: Optional[str] = None):
        """Логирование статуса процесса с переносами строк"""
        self.info(f"🔄 СТАТУС [{task_name}]: {status}")
        if details:
            self.info(f"   Детали: {details}")
        
        # Структурированное логирование
        self._log_structured({
            "event": "process_status",
            "task_name": task_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def _log_structured(self, data: Dict[str, Any]):
        """Структурированное логирование в JSON с переносами строк"""
        json_logger = logging.getLogger(f"{self.name}.json")
        json_logger.info(json.dumps(data, ensure_ascii=False))
    
    def cleanup_old_logs(self, days: int = 7):
        """Очистка старых логов"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                self.info(f"🗑️ Удален старый лог: {log_file.name}")

class JsonFormatter(logging.Formatter):
    """Форматтер для JSON логов с обязательными переносами строк"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Возвращаем JSON с переносом строки
        return json.dumps(log_data, ensure_ascii=False) + '\n'

# Глобальный экземпляр логгера
_global_logger = None
_logger_lock = threading.Lock()

def get_logger(component_name: Optional[str] = None):
    """Получить глобальный логгер с потокобезопасностью"""
    global _global_logger
    with _logger_lock:
        if _global_logger is None:
            _global_logger = CentralizedLogger()
        return _global_logger.get_logger(component_name)

def setup_logging(log_dir: str = "/home/microWakeWord/logs"):
    """Настройка централизованного логирования с потокобезопасностью"""
    global _global_logger
    with _logger_lock:
        _global_logger = CentralizedLogger(log_dir=log_dir)
        return _global_logger

def log_print(message: str, level: str = "info"):
    """Замена print() на централизованное логирование с переносами строк"""
    logger = get_logger()
    
    # Убираем существующие переносы строк
    clean_message = message.rstrip('\n\r')
    
    # Логируем с нужным уровнем
    if level.lower() == "debug":
        logger.debug(clean_message)
    elif level.lower() == "warning":
        logger.warning(clean_message)
    elif level.lower() == "error":
        logger.error(clean_message)
    elif level.lower() == "critical":
        logger.critical(clean_message)
    else:  # default to info
        logger.info(clean_message)

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logger_system = setup_logging()
    
    # Получение логгера для компонента
    train_logger = get_logger("training")
    data_logger = get_logger("data_generation")
    
    # Примеры использования
    logger_system.log_task_start("test_task", "python test.py", pid=12345)
    logger_system.log_training_progress(1, 100, 0.95, 0.1, 0.92, 0.98)
    logger_system.log_resource_usage(75.5, 1024.3, "python training.py")
    logger_system.log_data_generation("positives", 1000, 30.5)
    logger_system.log_task_end("test_task", 0, 120.5)
    
    # Тест замены print
    log_print("✅ Централизованное логирование настроено!")
    log_print("Тест переноса строк", "info")
    log_print("Тест ошибки", "error")