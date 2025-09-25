#!/usr/bin/env python3
"""
Конфигурация pytest для microWakeWord
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import yaml
import json


@pytest.fixture
def temp_data_dir():
    """Временная директория для тестовых данных"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_audio_data():
    """Образцы аудио данных для тестов"""
    # Создание синтетических аудио данных
    sample_rate = 16000
    duration = 0.5
    samples = int(sample_rate * duration)
    
    # Синусоидальный сигнал
    frequency = 440  # A4
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    return {
        'data': audio_data,
        'sample_rate': sample_rate,
        'duration': duration,
        'samples': samples
    }


@pytest.fixture
def sample_spectrogram_data():
    """Образцы спектрограмм для тестов"""
    # Создание синтетических спектрограмм
    height = 40
    width = 3
    spectrogram = np.random.rand(height, width).astype(np.float32)
    
    return {
        'data': spectrogram,
        'shape': (height, width),
        'dtype': np.float32
    }


@pytest.fixture
def sample_model_config():
    """Конфигурация модели для тестов"""
    return {
        'model': {
            'name': 'Test Model',
            'size': 'small',
            'target_size_kb': 50,
            'architecture': {
                'type': 'MixedNet',
                'layers': 3,
                'hidden_units': 64
            },
            'training': {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.002
            }
        }
    }


@pytest.fixture
def sample_pipeline_config():
    """Конфигурация пайплайна для тестов"""
    return {
        'pipeline': {
            'name': 'Test Pipeline',
            'version': '2.0',
            'general': {
                'wake_word': 'test word',
                'language': 'en',
                'sample_rate': 16000,
                'duration': 0.5
            },
            'data_generation': {
                'quick_mode': True,
                'parallel_workers': 4
            },
            'training': {
                'architecture': 'MixedNet',
                'model_size': 'small'
            }
        }
    }


@pytest.fixture
def mock_logger():
    """Мок логгера для тестов"""
    class MockLogger:
        def __init__(self):
            self.logs = []
        
        def info(self, message):
            self.logs.append(('INFO', message))
        
        def warning(self, message):
            self.logs.append(('WARNING', message))
        
        def error(self, message):
            self.logs.append(('ERROR', message))
        
        def debug(self, message):
            self.logs.append(('DEBUG', message))
    
    return MockLogger()


@pytest.fixture
def sample_test_data():
    """Тестовые данные для моделей"""
    # Создание тестовых данных
    n_samples = 100
    n_features = 120  # 3 * 40 для спектрограмм
    
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int32)
    
    return {
        'X': X,
        'y': y,
        'n_samples': n_samples,
        'n_features': n_features
    }


@pytest.fixture
def temp_config_file(temp_data_dir, sample_pipeline_config):
    """Временный конфигурационный файл"""
    config_file = temp_data_dir / 'test_config.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(sample_pipeline_config, f, default_flow_style=False)
    
    return config_file


@pytest.fixture
def temp_model_file(temp_data_dir):
    """Временный файл модели"""
    model_file = temp_data_dir / 'test_model.tflite'
    # Создание пустого файла модели
    model_file.touch()
    return model_file


@pytest.fixture
def sample_metrics():
    """Образцы метрик для тестов"""
    return {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.88,
        'f1_score': 0.90,
        'roc_auc': 0.94,
        'false_reject_rate': 0.02,
        'false_accept_rate': 0.01,
        'false_accepts_per_hour': 0.5
    }


@pytest.fixture
def sample_performance_data():
    """Образцы данных производительности"""
    return {
        'cpu_usage': [45.2, 47.8, 52.1, 48.9, 46.3],
        'memory_usage': [1234.5, 1245.2, 1256.8, 1248.1, 1239.7],
        'duration': [1.2, 1.1, 1.3, 1.0, 1.2],
        'throughput': [850, 920, 780, 1000, 870]
    }


@pytest.fixture(scope="session")
def test_environment():
    """Настройка тестового окружения"""
    # Создание тестовых директорий
    test_dirs = [
        'tests/fixtures/sample_data',
        'tests/fixtures/mock_models',
        'tests/fixtures/test_configs'
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Очистка после тестов (если нужно)
    pass


@pytest.fixture
def mock_file_system(temp_data_dir):
    """Мок файловой системы для тестов"""
    # Создание структуры директорий
    dirs = [
        'audio/positives',
        'audio/negatives',
        'spectrograms/positives',
        'spectrograms/negatives',
        'models',
        'logs'
    ]
    
    for dir_path in dirs:
        (temp_data_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Создание тестовых файлов
    files = [
        'audio/positives/sample1.wav',
        'audio/negatives/sample1.wav',
        'spectrograms/positives/sample1.npy',
        'spectrograms/negatives/sample1.npy',
        'models/test_model.tflite',
        'logs/test.log'
    ]
    
    for file_path in files:
        (temp_data_dir / file_path).touch()
    
    return temp_data_dir


# Параметризованные фикстуры
@pytest.fixture(params=['small', 'medium', 'large'])
def model_size(request):
    """Размеры моделей для тестирования"""
    return request.param


@pytest.fixture(params=[16000, 22050, 44100])
def sample_rate(request):
    """Частоты дискретизации для тестирования"""
    return request.param


@pytest.fixture(params=[0.5, 1.0, 2.0])
def duration(request):
    """Длительности аудио для тестирования"""
    return request.param


# Утилиты для тестов
class TestUtils:
    """Утилиты для тестов"""
    
    @staticmethod
    def create_test_audio(sample_rate=16000, duration=0.5, frequency=440):
        """Создание тестового аудио"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        return np.sin(2 * np.pi * frequency * t)
    
    @staticmethod
    def create_test_spectrogram(height=40, width=3):
        """Создание тестовой спектрограммы"""
        return np.random.rand(height, width).astype(np.float32)
    
    @staticmethod
    def assert_audio_valid(audio_data, sample_rate=16000, duration=0.5):
        """Проверка валидности аудио данных"""
        expected_samples = int(sample_rate * duration)
        assert len(audio_data) == expected_samples
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.dtype in [np.float32, np.float64]
    
    @staticmethod
    def assert_spectrogram_valid(spectrogram, expected_shape=(40, 3)):
        """Проверка валидности спектрограммы"""
        assert spectrogram.shape == expected_shape
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.dtype == np.float32


@pytest.fixture
def test_utils():
    """Фикстура с утилитами для тестов"""
    return TestUtils()