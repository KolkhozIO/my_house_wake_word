#!/usr/bin/env python3
"""
Менеджер единой конфигурации microWakeWord пайплайна
Обеспечивает синхронизацию параметров и валидацию совместимости
"""

import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from centralized_logger import setup_logging, get_logger

class ConfigManager:
    """Менеджер единой конфигурации пайплайна"""
    
    def __init__(self, config_path: str = "unified_config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.logger = get_logger("config_manager")
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """Загрузка конфигурации из файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"✅ Конфигурация загружена: {self.config_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
            raise
    
    def _validate_config(self):
        """Валидация конфигурации"""
        self.logger.info("🔍 Валидация конфигурации...")
        
        # Проверка обязательных секций
        required_sections = ['audio', 'model', 'augmentation', 'paths', 'validation']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Отсутствует обязательная секция: {section}")
        
        # Валидация аудио параметров
        self._validate_audio_params()
        
        # Валидация путей
        self._validate_paths()
        
        # Синхронизация параметров
        self._sync_parameters()
        
        self.logger.info("✅ Конфигурация валидирована успешно")
    
    def _validate_audio_params(self):
        """Валидация аудио параметров"""
        audio = self.config['audio']
        
        # Проверка совместимости параметров
        clip_duration_ms = audio['clip_duration_ms']
        window_step_ms = audio['window_step_ms']
        step_ms = audio['step_ms']
        
        # Расчет ожидаемой длины спектрограммы
        expected_frames = clip_duration_ms / window_step_ms
        model_input_shape = self.config['model']['input_shape']
        
        if abs(expected_frames - model_input_shape[0]) > 1:
            self.logger.warning(f"⚠️ Несоответствие длины спектрограммы:")
            self.logger.warning(f"   Ожидается: {expected_frames} кадров")
            self.logger.warning(f"   Модель: {model_input_shape[0]} кадров")
            self.logger.warning(f"   Разница: {abs(expected_frames - model_input_shape[0])} кадров")
        
        # Проверка синхронизации step_ms и window_step_ms
        if step_ms != window_step_ms:
            self.logger.warning(f"⚠️ Несоответствие параметров:")
            self.logger.warning(f"   step_ms: {step_ms}")
            self.logger.warning(f"   window_step_ms: {window_step_ms}")
            self.logger.warning("   Рекомендуется синхронизировать параметры")
    
    def _validate_paths(self):
        """Валидация путей к данным"""
        paths = self.config['paths']
        
        # Проверка существования базовых директорий
        base_dirs = ['data_root', 'models_root', 'logs_root']
        for dir_name in base_dirs:
            dir_path = Path(paths[dir_name])
            if not dir_path.exists():
                self.logger.warning(f"⚠️ Директория не найдена: {dir_path}")
                self.logger.info(f"   Создаю директорию: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Проверка источников данных
        sources = paths['sources']
        for source_name, source_path in sources.items():
            if not Path(source_path).exists():
                self.logger.warning(f"⚠️ Источник данных не найден: {source_name} -> {source_path}")
    
    def _sync_parameters(self):
        """Синхронизация параметров между компонентами"""
        self.logger.info("🔄 Синхронизация параметров...")
        
        # Синхронизация step_ms и window_step_ms
        audio = self.config['audio']
        if audio['step_ms'] != audio['window_step_ms']:
            self.logger.info(f"   Синхронизирую step_ms: {audio['step_ms']} -> {audio['window_step_ms']}")
            audio['step_ms'] = audio['window_step_ms']
        
        # Обновление формы модели на основе аудио параметров
        clip_duration_ms = audio['clip_duration_ms']
        window_step_ms = audio['window_step_ms']
        expected_frames = int(clip_duration_ms / window_step_ms)
        
        model_input_shape = self.config['model']['input_shape']
        if model_input_shape[0] != expected_frames:
            self.logger.info(f"   Обновляю форму модели: {model_input_shape[0]} -> {expected_frames}")
            self.config['model']['input_shape'] = [expected_frames, model_input_shape[1]]
        
        self.logger.info("✅ Параметры синхронизированы")
    
    def get_audio_params(self) -> Dict[str, Any]:
        """Получение аудио параметров"""
        return self.config['audio'].copy()
    
    def get_model_params(self) -> Dict[str, Any]:
        """Получение параметров модели"""
        return self.config['model'].copy()
    
    def get_augmentation_params(self) -> Dict[str, Any]:
        """Получение параметров аугментации"""
        return self.config['augmentation'].copy()
    
    def get_paths(self) -> Dict[str, Any]:
        """Получение путей"""
        return self.config['paths'].copy()
    
    def get_validation_params(self) -> Dict[str, Any]:
        """Получение параметров валидации"""
        return self.config['validation'].copy()
    
    def get_performance_params(self) -> Dict[str, Any]:
        """Получение параметров производительности"""
        return self.config['performance'].copy()
    
    def calculate_spectrogram_length(self) -> int:
        """Расчет длины спектрограммы на основе параметров"""
        audio = self.config['audio']
        clip_duration_ms = audio['clip_duration_ms']
        window_step_ms = audio['window_step_ms']
        return int(clip_duration_ms / window_step_ms)
    
    def validate_data_model_compatibility(self, data_shape: Tuple[int, int]) -> bool:
        """Валидация совместимости данных и модели"""
        model_input_shape = self.config['model']['input_shape']
        
        # Проверка совместимости форм
        if data_shape != tuple(model_input_shape):
            self.logger.error(f"❌ Несовместимость форм:")
            self.logger.error(f"   Данные: {data_shape}")
            self.logger.error(f"   Модель: {model_input_shape}")
            return False
        
        self.logger.info(f"✅ Совместимость форм подтверждена: {data_shape}")
        return True
    
    def generate_training_config(self, output_path: str) -> str:
        """Генерация конфигурации для обучения модели"""
        self.logger.info(f"📝 Генерация конфигурации обучения: {output_path}")
        
        # Создаем конфигурацию в формате microWakeWord
        training_config = {
            'batch_size': self.config['model']['batch_size'],
            'clip_duration_ms': self.config['audio']['clip_duration_ms'],
            'eval_step_interval': self.config['model']['eval_step_interval'],
            'features': [],
            'freq_mask_count': self.config['augmentation']['freq_mask_count'],
            'freq_mask_max_size': self.config['augmentation']['freq_mask_max_size'],
            'learning_rates': [self.config['model']['learning_rate']],
            'maximization_metric': self.config['model']['maximization_metric'],
            'minimization_metric': self.config['model']['minimization_metric'],
            'negative_class_weight': self.config['model']['negative_class_weight'],
            'positive_class_weight': self.config['model']['positive_class_weight'],
            'target_minimization': self.config['model']['target_minimization'],
            'time_mask_count': self.config['augmentation']['time_mask_count'],
            'time_mask_max_size': self.config['augmentation']['time_mask_max_size'],
            'train_dir': 'trained_models/wakeword',
            'training_steps': self.config['model']['training_steps'] if isinstance(self.config['model']['training_steps'], list) else [self.config['model']['training_steps']],
            'window_step_ms': self.config['audio']['window_step_ms'],
            'spectrogram_length': self.calculate_spectrogram_length(),
            'spectrogram_length_final_layer': self.calculate_spectrogram_length() - 28,  # Стандартное смещение
            'stride': 1,
            'training_input_shape': self.config['model']['input_shape']
        }
        
        # Добавляем источники данных
        paths = self.config['paths']['generated']
        for source_name, source_path in paths.items():
            # Определяем тип данных
            if 'positive' in source_name:
                truth = True
            else:
                truth = False
            
            training_config['features'].append({
                'features_dir': source_path,
                'penalty_weight': 1.0,
                'sampling_weight': 1.0,
                'truncation_strategy': 'random',
                'truth': truth,
                'type': 'mmap'
            })
        
        # Сохраняем конфигурацию
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"✅ Конфигурация обучения сохранена: {output_path}")
        return output_path
    
    def save_config(self, output_path: str = None):
        """Сохранение обновленной конфигурации"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"💾 Конфигурация сохранена: {output_path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Получение сводки конфигурации"""
        return {
            'version': self.config['version']['config_version'],
            'audio_params': self.get_audio_params(),
            'model_params': self.get_model_params(),
            'spectrogram_length': self.calculate_spectrogram_length(),
            'data_sources': len(self.config['paths']['sources']),
            'generated_sources': len(self.config['paths']['generated'])
        }

# Глобальный экземпляр менеджера конфигурации
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Получение глобального экземпляра менеджера конфигурации"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logger_system = setup_logging()
    
    # Создание менеджера конфигурации
    config_manager = ConfigManager()
    
    # Получение сводки
    summary = config_manager.get_config_summary()
    print("📊 Сводка конфигурации:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Генерация конфигурации обучения
    training_config_path = config_manager.generate_training_config("/home/microWakeWord_data/unified_training_config.yaml")
    print(f"✅ Конфигурация обучения создана: {training_config_path}")