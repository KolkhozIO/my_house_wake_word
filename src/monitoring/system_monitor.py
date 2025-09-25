#!/usr/bin/env python3
"""
Система мониторинга microWakeWord
"""

import time
import psutil
import threading
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging


class SystemMonitor:
    """Система мониторинга для microWakeWord"""
    
    def __init__(self, config_path: str = "config/base/system.yaml"):
        self.config = self._load_config(config_path)
        self.metrics = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.logger = self._setup_logger()
        
        # Пороги для алертов
        self.thresholds = {
            'cpu_usage': 80,
            'memory_usage': 85,
            'disk_usage': 90,
            'temperature': 80,
            'pipeline_errors': 5,
            'model_accuracy': 0.8
        }
    
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
                'monitoring': {
                    'enabled': True,
                    'interval': 30,
                    'alert_thresholds': {
                        'cpu_usage': 80,
                        'memory_usage': 85,
                        'disk_usage': 90
                    }
                }
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger('system_monitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self, interval: int = 30):
        """Запуск мониторинга"""
        if self.monitoring_active:
            self.logger.warning("Мониторинг уже активен")
            return
        
        self.logger.info(f"Запуск мониторинга с интервалом {interval} секунд")
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        if not self.monitoring_active:
            self.logger.warning("Мониторинг не активен")
            return
        
        self.logger.info("Остановка мониторинга")
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval: int):
        """Основной цикл мониторинга"""
        while self.monitoring_active:
            try:
                # Сбор метрик
                metrics = self._collect_metrics()
                
                # Проверка порогов
                alerts = self._check_thresholds(metrics)
                
                # Отправка алертов
                if alerts:
                    self._send_alerts(alerts)
                
                # Сохранение метрик
                self._save_metrics(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Сбор системных метрик"""
        timestamp = datetime.now().isoformat()
        
        # Системные метрики
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Метрики процесса
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        process_cpu = process.cpu_percent()
        
        # Температура (если доступна)
        try:
            temperatures = psutil.sensors_temperatures()
            cpu_temp = temperatures.get('coretemp', [{}])[0].current if temperatures.get('coretemp') else None
        except:
            cpu_temp = None
        
        # Сетевые метрики
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': timestamp,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / 1024 / 1024 / 1024,
                'memory_available_gb': memory.available / 1024 / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / 1024 / 1024 / 1024,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'cpu_temperature': cpu_temp
            },
            'process': {
                'memory_mb': process_memory,
                'cpu_percent': process_cpu,
                'pid': process.pid,
                'status': process.status()
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        }
        
        return metrics
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Проверка порогов и генерация алертов"""
        alerts = []
        
        system = metrics['system']
        
        # Проверка CPU
        if system['cpu_percent'] > self.thresholds['cpu_usage']:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"Высокая загрузка CPU: {system['cpu_percent']:.1f}%",
                'value': system['cpu_percent'],
                'threshold': self.thresholds['cpu_usage']
            })
        
        # Проверка памяти
        if system['memory_percent'] > self.thresholds['memory_usage']:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"Высокое использование памяти: {system['memory_percent']:.1f}%",
                'value': system['memory_percent'],
                'threshold': self.thresholds['memory_usage']
            })
        
        # Проверка диска
        if system['disk_percent'] > self.thresholds['disk_usage']:
            alerts.append({
                'type': 'disk_full',
                'severity': 'critical',
                'message': f"Мало места на диске: {system['disk_percent']:.1f}%",
                'value': system['disk_percent'],
                'threshold': self.thresholds['disk_usage']
            })
        
        # Проверка температуры
        if system['cpu_temperature'] and system['cpu_temperature'] > self.thresholds['temperature']:
            alerts.append({
                'type': 'temperature_high',
                'severity': 'warning',
                'message': f"Высокая температура CPU: {system['cpu_temperature']:.1f}°C",
                'value': system['cpu_temperature'],
                'threshold': self.thresholds['temperature']
            })
        
        return alerts
    
    def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Отправка алертов"""
        for alert in alerts:
            self.logger.warning(f"АЛЕРТ: {alert['message']}")
            
            # Сохранение алерта
            self.alerts.append({
                **alert,
                'timestamp': datetime.now().isoformat()
            })
            
            # Отправка email (если настроено)
            self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Отправка email алерта"""
        try:
            # Проверка настроек email
            smtp_config = self.config.get('email', {})
            if not smtp_config.get('enabled', False):
                return
            
            # Создание сообщения
            msg = MIMEMultipart()
            msg['From'] = smtp_config['username']
            msg['To'] = smtp_config['notification_email']
            msg['Subject'] = f"microWakeWord Alert: {alert['type']}"
            
            body = f"""
            Алерт от системы мониторинга microWakeWord
            
            Тип: {alert['type']}
            Серьезность: {alert['severity']}
            Сообщение: {alert['message']}
            Значение: {alert['value']}
            Порог: {alert['threshold']}
            Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Отправка
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email алерт отправлен: {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки email алерта: {e}")
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Сохранение метрик"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Сохранение в JSON
        metrics_file = Path("logs/system/metrics.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Загрузка существующих метрик
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            except:
                existing_metrics = []
        else:
            existing_metrics = []
        
        # Добавление новых метрик
        existing_metrics.append(metrics)
        
        # Сохранение (только последние 1000 записей)
        if len(existing_metrics) > 1000:
            existing_metrics = existing_metrics[-1000:]
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Получение состояния системы"""
        metrics = self._collect_metrics()
        
        # Определение состояния
        health_score = 100
        
        if metrics['system']['cpu_percent'] > 80:
            health_score -= 20
        if metrics['system']['memory_percent'] > 85:
            health_score -= 20
        if metrics['system']['disk_percent'] > 90:
            health_score -= 30
        
        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'health_score': health_score,
            'metrics': metrics,
            'alerts_count': len(self.alerts),
            'last_check': datetime.now().isoformat()
        }
    
    def get_alerts_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории алертов"""
        return self.alerts[-limit:] if self.alerts else []
    
    def clear_alerts(self):
        """Очистка истории алертов"""
        self.alerts.clear()
        self.logger.info("История алертов очищена")
    
    def generate_report(self) -> str:
        """Генерация отчета о состоянии системы"""
        health = self.get_system_health()
        alerts = self.get_alerts_history(10)
        
        report = f"""
# Отчет о состоянии системы microWakeWord

## 📊 Общее состояние
- **Статус**: {health['status'].upper()}
- **Оценка здоровья**: {health['health_score']}/100
- **Последняя проверка**: {health['last_check']}
- **Количество алертов**: {health['alerts_count']}

## 📈 Текущие метрики
- **CPU**: {health['metrics']['system']['cpu_percent']:.1f}%
- **Память**: {health['metrics']['system']['memory_percent']:.1f}%
- **Диск**: {health['metrics']['system']['disk_percent']:.1f}%
- **Температура CPU**: {health['metrics']['system']['cpu_temperature'] or 'N/A'}°C

## 🚨 Последние алерты
"""
        
        for alert in alerts:
            report += f"- **{alert['type']}**: {alert['message']} ({alert['timestamp']})\n"
        
        return report


def main():
    """Основная функция для тестирования"""
    monitor = SystemMonitor()
    
    print("🔍 Система мониторинга microWakeWord")
    
    # Запуск мониторинга
    monitor.start_monitoring(interval=10)
    
    # Ожидание
    time.sleep(30)
    
    # Получение состояния
    health = monitor.get_system_health()
    print(f"Состояние системы: {health['status']} ({health['health_score']}/100)")
    
    # Генерация отчета
    report = monitor.generate_report()
    print(report)
    
    # Остановка мониторинга
    monitor.stop_monitoring()


if __name__ == "__main__":
    main()