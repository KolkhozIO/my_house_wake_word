#!/usr/bin/env python3
"""
Профайлер производительности для microWakeWord
"""

import time
import psutil
import threading
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import functools


class PerformanceProfiler:
    """Профайлер производительности для microWakeWord"""
    
    def __init__(self, config_path: str = "config/base/system.yaml"):
        self.config = self._load_config(config_path)
        self.metrics = {}
        self.active_profiles = {}
        self.monitoring_thread = None
        self.monitoring_active = False
        self.results_dir = Path("tools/performance/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'system': {
                'performance': {
                    'monitoring_interval': 1,
                    'max_monitoring_duration': 3600,
                    'alert_thresholds': {
                        'cpu_usage': 80,
                        'memory_usage': 85,
                        'disk_usage': 90
                    }
                }
            }
        }
    
    @contextmanager
    def profile_function(self, function_name: str, **kwargs):
        """Профилирование функции"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self._record_metric(function_name, {
                'duration': duration,
                'memory_delta': memory_delta,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'timestamp': datetime.now().isoformat(),
                **kwargs
            })
    
    def profile_pipeline_stage(self, stage_name: str, stage_function: Callable, *args, **kwargs):
        """Профилирование этапа пайплайна"""
        print(f"🔍 Профилирование этапа: {stage_name}")
        
        with self.profile_function(f"pipeline.{stage_name}"):
            result = stage_function(*args, **kwargs)
        
        return result
    
    def start_system_monitoring(self, duration: int = 300, interval: float = 1.0):
        """Запуск мониторинга системы"""
        if self.monitoring_active:
            print("⚠️ Мониторинг уже активен")
            return
        
        print(f"📊 Запуск мониторинга системы на {duration} секунд...")
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            args=(duration, interval)
        )
        self.monitoring_thread.start()
    
    def stop_system_monitoring(self):
        """Остановка мониторинга системы"""
        if not self.monitoring_active:
            print("⚠️ Мониторинг не активен")
            return
        
        print("🛑 Остановка мониторинга системы...")
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_system(self, duration: int, interval: float):
        """Мониторинг системных ресурсов"""
        start_time = time.time()
        end_time = start_time + duration
        
        while self.monitoring_active and time.time() < end_time:
            timestamp = datetime.now()
            
            # Сбор метрик
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Метрики процесса
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            # Запись метрик
            self._record_metric("system.monitoring", {
                'timestamp': timestamp.isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / 1024 / 1024 / 1024,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / 1024 / 1024 / 1024,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'process_memory_mb': process_memory,
                'process_cpu_percent': process_cpu
            })
            
            # Проверка алертов
            self._check_alerts(cpu_percent, memory.percent, disk.percent)
            
            time.sleep(interval)
        
        print("✅ Мониторинг системы завершен")
    
    def _check_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Проверка алертов"""
        thresholds = self.config['system']['performance']['alert_thresholds']
        
        alerts = []
        
        if cpu_percent > thresholds['cpu_usage']:
            alerts.append(f"Высокая загрузка CPU: {cpu_percent:.1f}%")
        
        if memory_percent > thresholds['memory_usage']:
            alerts.append(f"Высокое использование памяти: {memory_percent:.1f}%")
        
        if disk_percent > thresholds['disk_usage']:
            alerts.append(f"Мало места на диске: {disk_percent:.1f}%")
        
        if alerts:
            for alert in alerts:
                print(f"🚨 АЛЕРТ: {alert}")
                self._record_metric("system.alerts", {
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent
                })
    
    def _record_metric(self, metric_name: str, data: Dict[str, Any]):
        """Запись метрики"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(data)
    
    def analyze_performance(self, metric_name: str = None) -> Dict[str, Any]:
        """Анализ производительности"""
        if metric_name:
            metrics_to_analyze = {metric_name: self.metrics.get(metric_name, [])}
        else:
            metrics_to_analyze = self.metrics
        
        analysis = {}
        
        for name, data in metrics_to_analyze.items():
            if not data:
                continue
            
            analysis[name] = self._analyze_metric_data(name, data)
        
        return analysis
    
    def _analyze_metric_data(self, metric_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ данных метрики"""
        if not data:
            return {'error': 'Нет данных для анализа'}
        
        # Извлечение числовых значений
        numeric_fields = {}
        for field in data[0].keys():
            if isinstance(data[0][field], (int, float)):
                values = [item[field] for item in data if field in item]
                if values:
                    numeric_fields[field] = values
        
        analysis = {
            'total_samples': len(data),
            'time_range': {
                'start': data[0].get('timestamp', 'unknown'),
                'end': data[-1].get('timestamp', 'unknown')
            }
        }
        
        # Статистика для числовых полей
        for field, values in numeric_fields.items():
            analysis[f'{field}_stats'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # Специальный анализ для разных типов метрик
        if 'duration' in numeric_fields:
            analysis['performance_summary'] = self._analyze_duration_metrics(numeric_fields['duration'])
        
        if 'cpu_percent' in numeric_fields:
            analysis['system_summary'] = self._analyze_system_metrics(numeric_fields)
        
        return analysis
    
    def _analyze_duration_metrics(self, durations: List[float]) -> Dict[str, Any]:
        """Анализ метрик длительности"""
        return {
            'total_time': sum(durations),
            'average_time': np.mean(durations),
            'fastest_execution': min(durations),
            'slowest_execution': max(durations),
            'executions_per_second': len(durations) / sum(durations) if sum(durations) > 0 else 0
        }
    
    def _analyze_system_metrics(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Анализ системных метрик"""
        summary = {}
        
        if 'cpu_percent' in metrics:
            cpu_values = metrics['cpu_percent']
            summary['cpu'] = {
                'average_usage': np.mean(cpu_values),
                'peak_usage': np.max(cpu_values),
                'usage_stability': 1 - np.std(cpu_values) / np.mean(cpu_values) if np.mean(cpu_values) > 0 else 0
            }
        
        if 'memory_percent' in metrics:
            memory_values = metrics['memory_percent']
            summary['memory'] = {
                'average_usage': np.mean(memory_values),
                'peak_usage': np.max(memory_values),
                'usage_trend': 'increasing' if memory_values[-1] > memory_values[0] else 'decreasing'
            }
        
        return summary
    
    def generate_report(self, analysis: Dict[str, Any] = None) -> str:
        """Генерация отчета о производительности"""
        if analysis is None:
            analysis = self.analyze_performance()
        
        report = "# Отчет о производительности microWakeWord\n\n"
        
        for metric_name, metric_analysis in analysis.items():
            report += f"## {metric_name}\n\n"
            
            if 'error' in metric_analysis:
                report += f"❌ {metric_analysis['error']}\n\n"
                continue
            
            report += f"- **Образцов**: {metric_analysis['total_samples']}\n"
            
            if 'time_range' in metric_analysis:
                report += f"- **Временной диапазон**: {metric_analysis['time_range']['start']} - {metric_analysis['time_range']['end']}\n"
            
            # Статистика по числовым полям
            for field, stats in metric_analysis.items():
                if field.endswith('_stats'):
                    field_name = field.replace('_stats', '')
                    report += f"\n### {field_name}\n"
                    report += f"- **Среднее**: {stats['mean']:.2f}\n"
                    report += f"- **Стандартное отклонение**: {stats['std']:.2f}\n"
                    report += f"- **Минимум**: {stats['min']:.2f}\n"
                    report += f"- **Максимум**: {stats['max']:.2f}\n"
                    report += f"- **Медиана**: {stats['median']:.2f}\n"
            
            # Специальные сводки
            if 'performance_summary' in metric_analysis:
                perf = metric_analysis['performance_summary']
                report += f"\n### Сводка производительности\n"
                report += f"- **Общее время**: {perf['total_time']:.2f} сек\n"
                report += f"- **Среднее время**: {perf['average_time']:.2f} сек\n"
                report += f"- **Выполнений в секунду**: {perf['executions_per_second']:.2f}\n"
            
            if 'system_summary' in metric_analysis:
                sys = metric_analysis['system_summary']
                if 'cpu' in sys:
                    report += f"\n### CPU\n"
                    report += f"- **Средняя загрузка**: {sys['cpu']['average_usage']:.1f}%\n"
                    report += f"- **Пиковая загрузка**: {sys['cpu']['peak_usage']:.1f}%\n"
                    report += f"- **Стабильность**: {sys['cpu']['usage_stability']:.1%}\n"
                
                if 'memory' in sys:
                    report += f"\n### Память\n"
                    report += f"- **Среднее использование**: {sys['memory']['average_usage']:.1f}%\n"
                    report += f"- **Пиковое использование**: {sys['memory']['peak_usage']:.1f}%\n"
                    report += f"- **Тренд**: {sys['memory']['usage_trend']}\n"
            
            report += "\n"
        
        return report
    
    def save_results(self, filename: str = None):
        """Сохранение результатов профилирования"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_profile_{timestamp}"
        
        # Сохранение сырых данных
        raw_data_file = self.results_dir / f"{filename}_raw.json"
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        # Анализ и сохранение отчета
        analysis = self.analyze_performance()
        analysis_file = self.results_dir / f"{filename}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Текстовый отчет
        report = self.generate_report(analysis)
        report_file = self.results_dir / f"{filename}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📁 Результаты профилирования сохранены:")
        print(f"  - Сырые данные: {raw_data_file}")
        print(f"  - Анализ: {analysis_file}")
        print(f"  - Отчет: {report_file}")
    
    def clear_metrics(self):
        """Очистка собранных метрик"""
        self.metrics.clear()
        print("🧹 Метрики очищены")


def profile_function(func_name: str = None):
    """Декоратор для профилирования функций"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            profiler = PerformanceProfiler()
            
            with profiler.profile_function(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def main():
    """Основная функция для тестирования"""
    profiler = PerformanceProfiler()
    
    print("🔍 Профайлер производительности microWakeWord")
    
    # Пример профилирования функции
    @profile_function("test_function")
    def test_function():
        time.sleep(1)
        return sum(range(1000))
    
    result = test_function()
    print(f"Результат тестовой функции: {result}")
    
    # Пример мониторинга системы
    profiler.start_system_monitoring(duration=10, interval=0.5)
    time.sleep(12)  # Ждем завершения мониторинга
    
    # Анализ и сохранение результатов
    analysis = profiler.analyze_performance()
    profiler.save_results("test_profile")
    
    print("✅ Тестирование завершено")


if __name__ == "__main__":
    main()