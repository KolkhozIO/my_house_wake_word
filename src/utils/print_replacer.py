#!/usr/bin/env python3
"""
Модуль для замены всех print() на централизованное логирование
Автоматически перехватывает print() и направляет в логгер
"""

import sys
import builtins
from typing import Any, Optional
from .centralized_logger import get_logger

class PrintReplacer:
    """Класс для замены стандартного print() на логирование"""
    
    def __init__(self, component_name: Optional[str] = None):
        self.logger = get_logger(component_name)
        self.original_print = builtins.print
        self._enabled = False
    
    def enable(self):
        """Включает перехват print()"""
        if not self._enabled:
            builtins.print = self._print_with_logging
            self._enabled = True
    
    def disable(self):
        """Отключает перехват print()"""
        if self._enabled:
            builtins.print = self.original_print
            self._enabled = False
    
    def _print_with_logging(self, *args, **kwargs):
        """Заменяет print() на логирование с переносами строк"""
        # Получаем сообщение как строку
        message = ' '.join(str(arg) for arg in args)
        
        # Убираем существующие переносы строк
        clean_message = message.rstrip('\n\r')
        
        # Определяем уровень логирования из kwargs
        level = kwargs.pop('level', 'info')
        
        # Логируем с нужным уровнем
        if level.lower() == "debug":
            self.logger.debug(clean_message)
        elif level.lower() == "warning":
            self.logger.warning(clean_message)
        elif level.lower() == "error":
            self.logger.error(clean_message)
        elif level.lower() == "critical":
            self.logger.critical(clean_message)
        else:  # default to info
            self.logger.info(clean_message)

# Глобальный экземпляр для автоматической замены
_global_print_replacer = None

def enable_print_replacement(component_name: Optional[str] = None):
    """Включает автоматическую замену print() на логирование"""
    global _global_print_replacer
    _global_print_replacer = PrintReplacer(component_name)
    _global_print_replacer.enable()

def disable_print_replacement():
    """Отключает автоматическую замену print()"""
    global _global_print_replacer
    if _global_print_replacer:
        _global_print_replacer.disable()
        _global_print_replacer = None

def log_print(message: str, level: str = "info", component_name: Optional[str] = None):
    """Прямое логирование вместо print() с переносами строк"""
    logger = get_logger(component_name)
    
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

# Контекстный менеджер для временной замены print()
class PrintReplacementContext:
    """Контекстный менеджер для временной замены print()"""
    
    def __init__(self, component_name: Optional[str] = None):
        self.component_name = component_name
        self.replacer = None
    
    def __enter__(self):
        self.replacer = PrintReplacer(self.component_name)
        self.replacer.enable()
        return self.replacer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.replacer:
            self.replacer.disable()

# Пример использования
if __name__ == "__main__":
    from .centralized_logger import setup_logging
    
    # Настройка логирования
    setup_logging()
    
    # Включаем замену print()
    enable_print_replacement("test_component")
    
    # Теперь все print() будут идти в логгер
    print("Это сообщение пойдет в логгер!")
    print("Тест переноса строк")
    print("Тест ошибки", level="error")
    
    # Отключаем замену
    disable_print_replacement()
    
    # Теперь print() работает как обычно
    print("Это обычный print()")
    
    # Используем контекстный менеджер
    with PrintReplacementContext("context_test"):
        print("Это сообщение в контексте")
    
    print("Это снова обычный print()")