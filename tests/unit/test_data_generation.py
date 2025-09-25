#!/usr/bin/env python3
"""
Модульные тесты для генерации данных
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Импорт модулей для тестирования
import sys
sys.path.append('src')

from pipeline.data_generation.quick_generate import QuickDataGenerator
from pipeline.data_generation.generate_both_phrases import BothPhrasesGenerator


class TestQuickDataGenerator:
    """Тесты для быстрого генератора данных"""
    
    def test_init(self):
        """Тест инициализации генератора"""
        generator = QuickDataGenerator()
        assert generator is not None
        assert generator.sample_rate == 16000
        assert generator.duration == 0.5
    
    def test_generate_audio(self, sample_audio_data):
        """Тест генерации аудио"""
        generator = QuickDataGenerator()
        
        # Генерация тестового аудио
        audio_data = generator._generate_sine_wave(
            frequency=440,
            sample_rate=sample_audio_data['sample_rate'],
            duration=sample_audio_data['duration']
        )
        
        assert len(audio_data) == sample_audio_data['samples']
        assert isinstance(audio_data, np.ndarray)
        assert audio_data.dtype == np.float32
    
    def test_generate_positives(self, temp_data_dir):
        """Тест генерации позитивных данных"""
        generator = QuickDataGenerator()
        output_dir = temp_data_dir / 'positives'
        
        # Генерация небольшого количества позитивных данных
        generator.generate_positives(
            output_dir=output_dir,
            count=5,
            phrases=['test phrase']
        )
        
        # Проверка создания файлов
        assert output_dir.exists()
        wav_files = list(output_dir.glob('*.wav'))
        assert len(wav_files) == 5
        
        # Проверка содержимого файлов
        for wav_file in wav_files:
            assert wav_file.stat().st_size > 0
    
    def test_generate_negatives(self, temp_data_dir):
        """Тест генерации негативных данных"""
        generator = QuickDataGenerator()
        output_dir = temp_data_dir / 'negatives'
        
        # Генерация небольшого количества негативных данных
        generator.generate_negatives(
            output_dir=output_dir,
            count=5
        )
        
        # Проверка создания файлов
        assert output_dir.exists()
        wav_files = list(output_dir.glob('*.wav'))
        assert len(wav_files) == 5
    
    def test_generate_dataset(self, temp_data_dir):
        """Тест генерации полного датасета"""
        generator = QuickDataGenerator()
        
        # Генерация небольшого датасета
        generator.generate_dataset(
            output_dir=temp_data_dir,
            positives_count=3,
            negatives_count=3
        )
        
        # Проверка структуры
        assert (temp_data_dir / 'positives').exists()
        assert (temp_data_dir / 'negatives').exists()
        
        # Проверка количества файлов
        pos_files = list((temp_data_dir / 'positives').glob('*.wav'))
        neg_files = list((temp_data_dir / 'negatives').glob('*.wav'))
        
        assert len(pos_files) == 3
        assert len(neg_files) == 3


class TestBothPhrasesGenerator:
    """Тесты для генератора обеих фраз"""
    
    def test_init(self):
        """Тест инициализации генератора"""
        generator = BothPhrasesGenerator()
        assert generator is not None
        assert 'милый дом' in generator.phrases
        assert 'любимый дом' in generator.phrases
    
    def test_generate_phrase_variations(self):
        """Тест генерации вариаций фраз"""
        generator = BothPhrasesGenerator()
        
        variations = generator._generate_phrase_variations('тест фраза')
        
        assert len(variations) > 0
        assert all(isinstance(v, str) for v in variations)
        assert 'тест фраза' in variations
    
    def test_generate_hard_negatives(self):
        """Тест генерации hard negatives"""
        generator = BothPhrasesGenerator()
        
        hard_negatives = generator._generate_hard_negatives()
        
        assert len(hard_negatives) > 0
        assert all(isinstance(neg, str) for neg in hard_negatives)
        
        # Проверка что это действительно негативные примеры
        for neg in hard_negatives:
            assert 'милый дом' not in neg.lower()
            assert 'любимый дом' not in neg.lower()


class TestDataGenerationIntegration:
    """Интеграционные тесты для генерации данных"""
    
    def test_full_pipeline(self, temp_data_dir):
        """Тест полного пайплайна генерации"""
        generator = QuickDataGenerator()
        
        # Генерация датасета
        generator.generate_dataset(
            output_dir=temp_data_dir,
            positives_count=10,
            negatives_count=20
        )
        
        # Проверка структуры
        assert (temp_data_dir / 'positives').exists()
        assert (temp_data_dir / 'negatives').exists()
        
        # Проверка метаданных
        metadata_file = temp_data_dir / 'metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            assert 'positives_count' in metadata
            assert 'negatives_count' in metadata
            assert metadata['positives_count'] == 10
            assert metadata['negatives_count'] == 20
    
    def test_data_quality(self, temp_data_dir):
        """Тест качества генерируемых данных"""
        generator = QuickDataGenerator()
        
        # Генерация тестовых данных
        generator.generate_dataset(
            output_dir=temp_data_dir,
            positives_count=5,
            negatives_count=5
        )
        
        # Проверка качества аудио файлов
        for audio_dir in ['positives', 'negatives']:
            audio_path = temp_data_dir / audio_dir
            wav_files = list(audio_path.glob('*.wav'))
            
            for wav_file in wav_files:
                # Проверка размера файла
                assert wav_file.stat().st_size > 1000  # Минимум 1KB
                
                # Проверка что файл не пустой
                assert wav_file.stat().st_size > 0


class TestDataGenerationPerformance:
    """Тесты производительности генерации данных"""
    
    def test_generation_speed(self, temp_data_dir):
        """Тест скорости генерации"""
        import time
        
        generator = QuickDataGenerator()
        
        start_time = time.time()
        
        generator.generate_dataset(
            output_dir=temp_data_dir,
            positives_count=50,
            negatives_count=50
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Проверка что генерация достаточно быстрая
        assert duration < 10  # Менее 10 секунд для 100 файлов
    
    def test_memory_usage(self, temp_data_dir):
        """Тест использования памяти"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        generator = QuickDataGenerator()
        
        generator.generate_dataset(
            output_dir=temp_data_dir,
            positives_count=100,
            negatives_count=100
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Проверка что использование памяти разумное
        assert memory_increase < 500  # Менее 500MB увеличения


if __name__ == "__main__":
    pytest.main([__file__, "-v"])