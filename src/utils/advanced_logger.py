#!/usr/bin/env python3
"""
Продвинутая система логирования для microWakeWord
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class AdvancedLogger:
    """Продвинутая система логирования с ротацией и структурированием"""
    
    def __init__(self, config_path: str = "config/base/system.yaml"):
        self.config = self._load_config(config_path)
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации логирования"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'system': {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file_rotation': True,
                    'max_file_size': '10MB',
                    'backup_count': 5
                },
                'paths': {
                    'logs_dir': '/home/microWakeWord/logs'
                }
            }
        }
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        log_config = self.config['system']['logging']
        logs_dir = Path(self.config['system']['paths']['logs_dir'])
        
        # Создание директорий для логов
        self._create_log_directories(logs_dir)
        
        # Настройка форматирования
        formatter = logging.Formatter(log_config['format'])
        
        # Настройка уровней логирования
        level = getattr(logging, log_config['level'].upper())
        
        # Создание логгеров для разных компонентов
        self._create_component_loggers(logs_dir, formatter, level)
    
    def _create_log_directories(self, logs_dir: Path):
        """Создание директорий для логов"""
        directories = [
            logs_dir / 'pipeline' / 'data_generation',
            logs_dir / 'pipeline' / 'augmentation',
            logs_dir / 'pipeline' / 'balancing',
            logs_dir / 'pipeline' / 'training',
            logs_dir / 'system' / 'performance',
            logs_dir / 'system' / 'errors',
            logs_dir / 'models' / 'training',
            logs_dir / 'models' / 'evaluation',
            logs_dir / 'models' / 'inference',
            logs_dir / 'archived' / 'daily',
            logs_dir / 'archived' / 'weekly',
            logs_dir / 'archived' / 'monthly'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _create_component_loggers(self, logs_dir: Path, formatter: logging.Formatter, level: int):
        """Создание логгеров для разных компонентов"""
        components = {
            'pipeline': {
                'data_generation': logs_dir / 'pipeline' / 'data_generation' / 'data_generation.log',
                'augmentation': logs_dir / 'pipeline' / 'augmentation' / 'augmentation.log',
                'balancing': logs_dir / 'pipeline' / 'balancing' / 'balancing.log',
                'training': logs_dir / 'pipeline' / 'training' / 'training.log'
            },
            'system': {
                'performance': logs_dir / 'system' / 'performance' / 'performance.log',
                'errors': logs_dir / 'system' / 'errors' / 'errors.log'
            },
            'models': {
                'training': logs_dir / 'models' / 'training' / 'training.log',
                'evaluation': logs_dir / 'models' / 'evaluation' / 'evaluation.log',
                'inference': logs_dir / 'models' / 'inference' / 'inference.log'
            }
        }
        
        for category, loggers in components.items():
            for name, log_file in loggers.items():
                logger = self._create_logger(name, log_file, formatter, level)
                self.loggers[f"{category}.{name}"] = logger
    
    def _create_logger(self, name: str, log_file: Path, formatter: logging.Formatter, level: int) -> logging.Logger:
        """Создание отдельного логгера"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Очистка существующих обработчиков
        logger.handlers.clear()
        
        # Файловый обработчик с ротацией
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Предотвращение дублирования логов
        logger.propagate = False
        
        return logger
    
    def get_logger(self, component: str) -> logging.Logger:
        """Получение логгера для компонента"""
        if component in self.loggers:
            return self.loggers[component]
        else:
            # Создание общего логгера
            logger = logging.getLogger(component)
            logger.setLevel(logging.INFO)
            return logger
    
    def log_structured(self, component: str, level: str, message: str, **kwargs):
        """Структурированное логирование"""
        logger = self.get_logger(component)
        
        # Создание структурированного сообщения
        structured_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'level': level,
            'message': message,
            **kwargs
        }
        
        # Логирование
        log_level = getattr(logging, level.upper())
        logger.log(log_level, json.dumps(structured_data, ensure_ascii=False))
    
    def log_performance(self, component: str, operation: str, duration: float, **metrics):
        """Логирование производительности"""
        self.log_structured(
            f"system.performance",
            "INFO",
            f"Performance metric: {operation}",
            component=component,
            operation=operation,
            duration=duration,
            **metrics
        )
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Логирование ошибок"""
        self.log_structured(
            f"system.errors",
            "ERROR",
            f"Error in {component}: {str(error)}",
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def archive_logs(self, archive_type: str = "daily"):
        """Архивирование логов"""
        logs_dir = Path(self.config['system']['paths']['logs_dir'])
        archive_dir = logs_dir / 'archived' / archive_type
        
        # Создание архива
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = archive_dir / f"logs_{timestamp}.tar.gz"
        
        # Архивирование (упрощенная версия)
        import tarfile
        with tarfile.open(archive_file, "w:gz") as tar:
            for log_file in logs_dir.rglob("*.log"):
                tar.add(log_file, arcname=log_file.relative_to(logs_dir))
        
        return archive_file


# Глобальный экземпляр логгера
advanced_logger = AdvancedLogger()


def get_logger(component: str) -> logging.Logger:
    """Получение логгера для компонента"""
    return advanced_logger.get_logger(component)


def log_performance(component: str, operation: str, duration: float, **metrics):
    """Логирование производительности"""
    advanced_logger.log_performance(component, operation, duration, **metrics)


def log_error(component: str, error: Exception, context: Dict[str, Any] = None):
    """Логирование ошибок"""
    advanced_logger.log_error(component, error, context)


if __name__ == "__main__":
    # Тестирование системы логирования
    logger = get_logger("pipeline.data_generation")
    logger.info("Тест системы логирования")
    
    log_performance("pipeline", "data_generation", 1.5, files_processed=1000)
    
    try:
        raise ValueError("Тестовая ошибка")
    except Exception as e:
        log_error("pipeline", e, {"test": True})