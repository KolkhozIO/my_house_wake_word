#!/usr/bin/env python3
"""
Модульные тесты для системы управления
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import time
import threading

# Импорт модулей для тестирования
import sys
sys.path.append('src')

from management.task_manager import TaskManager
from management.parallel_pipeline_manager import ParallelPipelineManager
from management.error_recovery_system import ErrorRecoverySystem


class TestTaskManager:
    """Тесты для менеджера задач"""
    
    def test_init(self, temp_data_dir):
        """Тест инициализации менеджера задач"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        assert manager is not None
        assert manager.tasks_dir == temp_data_dir
        assert manager.tasks_dir.exists()
    
    def test_start_task(self, temp_data_dir):
        """Тест запуска задачи"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск простой задачи
        task_name = "test_task"
        command = "echo 'test'"
        
        result = manager.start_task(task_name, command)
        assert result is True
        
        # Проверка создания файлов задачи
        task_files = list(temp_data_dir.glob(f"{task_name}_*"))
        assert len(task_files) >= 2  # JSON и PID файлы
    
    def test_get_task_status(self, temp_data_dir):
        """Тест получения статуса задачи"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск задачи
        task_name = "status_test"
        command = "sleep 1"
        manager.start_task(task_name, command)
        
        # Проверка статуса
        status = manager.get_task_status(task_name)
        assert status is not None
        assert 'status' in status
        assert 'pid' in status
    
    def test_stop_task(self, temp_data_dir):
        """Тест остановки задачи"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск долгой задачи
        task_name = "stop_test"
        command = "sleep 10"
        manager.start_task(task_name, command)
        
        # Остановка задачи
        result = manager.stop_task(task_name)
        assert result is True
        
        # Проверка что задача остановлена
        time.sleep(1)  # Даем время на остановку
        status = manager.get_task_status(task_name)
        assert status['status'] == 'stopped'
    
    def test_cleanup_tasks(self, temp_data_dir):
        """Тест очистки завершенных задач"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск и завершение задачи
        task_name = "cleanup_test"
        command = "echo 'done'"
        manager.start_task(task_name, command)
        
        time.sleep(2)  # Ждем завершения
        
        # Очистка
        manager.cleanup_tasks()
        
        # Проверка что файлы удалены
        task_files = list(temp_data_dir.glob(f"{task_name}_*"))
        assert len(task_files) == 0


class TestParallelPipelineManager:
    """Тесты для параллельного менеджера пайплайна"""
    
    def test_init(self):
        """Тест инициализации параллельного менеджера"""
        manager = ParallelPipelineManager()
        assert manager is not None
        assert hasattr(manager, 'stage_workers')
        assert hasattr(manager, 'resource_manager')
    
    def test_create_stage_worker(self):
        """Тест создания воркера этапа"""
        manager = ParallelPipelineManager()
        
        worker = manager._create_stage_worker(
            stage_name="test_stage",
            command="echo 'test'",
            cycle_interval=5,
            cpu_cores=2,
            memory_gb=1
        )
        
        assert worker is not None
        assert worker.stage_name == "test_stage"
        assert worker.command == "echo 'test'"
        assert worker.cycle_interval == 5
    
    def test_start_stage(self):
        """Тест запуска этапа"""
        manager = ParallelPipelineManager()
        
        # Запуск тестового этапа
        manager.start_stage("test_stage", "echo 'test'")
        
        # Проверка что воркер создан
        assert "test_stage" in manager.stage_workers
        
        # Остановка этапа
        manager.stop_stage("test_stage")
    
    def test_resource_allocation(self):
        """Тест распределения ресурсов"""
        manager = ParallelPipelineManager()
        
        # Проверка конфигурации ресурсов
        config = manager.stage_configs['generate_data']
        assert 'cpu_cores' in config
        assert 'memory_gb' in config
        assert config['cpu_cores'] > 0
        assert config['memory_gb'] > 0
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        manager = ParallelPipelineManager()
        
        # Запуск этапа с неверной командой
        manager.start_stage("error_test", "invalid_command_12345")
        
        # Даем время на выполнение
        time.sleep(2)
        
        # Проверка что ошибка обработана
        worker = manager.stage_workers.get("error_test")
        if worker:
            assert worker.running is False
        
        # Очистка
        manager.stop_stage("error_test")


class TestErrorRecoverySystem:
    """Тесты для системы восстановления ошибок"""
    
    def test_init(self):
        """Тест инициализации системы восстановления"""
        recovery = ErrorRecoverySystem()
        assert recovery is not None
        assert hasattr(recovery, 'error_patterns')
        assert hasattr(recovery, 'recovery_actions')
    
    def test_analyze_error(self):
        """Тест анализа ошибок"""
        recovery = ErrorRecoverySystem()
        
        # Тест различных типов ошибок
        test_cases = [
            ("No space left on device", "no_space_left"),
            ("MemoryError: Unable to allocate", "memory_error"),
            ("FileNotFoundError: [Errno 2]", "file_not_found"),
            ("PermissionError: [Errno 13]", "permission_denied"),
            ("ModuleNotFoundError: No module named", "import_error"),
            ("TimeoutError: Operation timed out", "timeout_error"),
            ("Data corrupted or invalid", "corrupted_data")
        ]
        
        for error_message, expected_type in test_cases:
            error_type = recovery.analyze_error(error_message)
            assert error_type == expected_type
    
    def test_recover_from_error(self, temp_data_dir):
        """Тест восстановления от ошибок"""
        recovery = ErrorRecoverySystem()
        
        # Тест восстановления от ошибки нехватки места
        result = recovery.recover_from_error("no_space_left", temp_data_dir)
        assert result is True
        
        # Тест восстановления от ошибки памяти
        result = recovery.recover_from_error("memory_error", temp_data_dir)
        assert result is True
    
    def test_unknown_error(self):
        """Тест неизвестной ошибки"""
        recovery = ErrorRecoverySystem()
        
        # Анализ неизвестной ошибки
        error_type = recovery.analyze_error("Unknown error message")
        assert error_type == "unknown"
        
        # Попытка восстановления
        result = recovery.recover_from_error("unknown", "/tmp")
        assert result is False


class TestManagementIntegration:
    """Интеграционные тесты для системы управления"""
    
    def test_task_lifecycle(self, temp_data_dir):
        """Тест полного жизненного цикла задачи"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск задачи
        task_name = "lifecycle_test"
        command = "echo 'start'; sleep 2; echo 'end'"
        
        start_result = manager.start_task(task_name, command)
        assert start_result is True
        
        # Проверка статуса
        status = manager.get_task_status(task_name)
        assert status['status'] == 'running'
        
        # Ожидание завершения
        time.sleep(3)
        
        # Проверка что задача завершена
        status = manager.get_task_status(task_name)
        assert status['status'] == 'finished'
        
        # Очистка
        manager.cleanup_tasks()
    
    def test_parallel_pipeline_lifecycle(self):
        """Тест жизненного цикла параллельного пайплайна"""
        manager = ParallelPipelineManager()
        
        # Запуск нескольких этапов
        stages = [
            ("stage1", "echo 'stage1'"),
            ("stage2", "echo 'stage2'"),
            ("stage3", "echo 'stage3'")
        ]
        
        for stage_name, command in stages:
            manager.start_stage(stage_name, command)
        
        # Проверка что все этапы запущены
        assert len(manager.stage_workers) == 3
        
        # Остановка всех этапов
        for stage_name, _ in stages:
            manager.stop_stage(stage_name)
        
        # Проверка что все этапы остановлены
        for stage_name, _ in stages:
            worker = manager.stage_workers[stage_name]
            assert worker.running is False
    
    def test_error_recovery_integration(self, temp_data_dir):
        """Тест интеграции системы восстановления ошибок"""
        manager = ParallelPipelineManager()
        recovery = ErrorRecoverySystem()
        
        # Запуск этапа с потенциальной ошибкой
        manager.start_stage("error_recovery_test", "ls /nonexistent_directory")
        
        # Даем время на выполнение и ошибку
        time.sleep(2)
        
        # Проверка что ошибка обнаружена и обработана
        worker = manager.stage_workers.get("error_recovery_test")
        if worker:
            # Система должна попытаться восстановиться
            assert worker.running is False
        
        # Очистка
        manager.stop_stage("error_recovery_test")


class TestManagementPerformance:
    """Тесты производительности системы управления"""
    
    def test_concurrent_tasks(self, temp_data_dir):
        """Тест одновременного выполнения задач"""
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск нескольких задач одновременно
        tasks = []
        for i in range(5):
            task_name = f"concurrent_test_{i}"
            command = f"echo 'task {i}'; sleep 1"
            manager.start_task(task_name, command)
            tasks.append(task_name)
        
        # Проверка что все задачи запущены
        for task_name in tasks:
            status = manager.get_task_status(task_name)
            assert status['status'] == 'running'
        
        # Ожидание завершения
        time.sleep(2)
        
        # Проверка что все задачи завершены
        for task_name in tasks:
            status = manager.get_task_status(task_name)
            assert status['status'] == 'finished'
    
    def test_memory_efficiency(self, temp_data_dir):
        """Тест эффективности использования памяти"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        manager = TaskManager(tasks_dir=temp_data_dir)
        
        # Запуск множества задач
        for i in range(20):
            task_name = f"memory_test_{i}"
            command = "echo 'test'"
            manager.start_task(task_name, command)
        
        # Ожидание завершения
        time.sleep(3)
        
        # Очистка
        manager.cleanup_tasks()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Проверка что использование памяти разумное
        assert memory_increase < 100  # Менее 100MB увеличения


if __name__ == "__main__":
    pytest.main([__file__, "-v"])