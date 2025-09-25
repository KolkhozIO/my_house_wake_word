#!/usr/bin/env python3
"""
Анализатор датасетов microWakeWord
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd


class DatasetAnalyzer:
    """Анализатор датасетов для microWakeWord"""
    
    def __init__(self, data_dir: str = "/home/microWakeWord_data"):
        self.data_dir = Path(data_dir)
        self.results = {}
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Полный анализ датасета"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
        
        print(f"🔍 Анализ датасета: {dataset_path.name}")
        
        # Базовый анализ
        basic_stats = self._analyze_basic_stats(dataset_path)
        
        # Анализ аудио файлов
        audio_stats = self._analyze_audio_files(dataset_path)
        
        # Анализ спектрограмм
        spectrogram_stats = self._analyze_spectrograms(dataset_path)
        
        # Анализ качества данных
        quality_stats = self._analyze_data_quality(dataset_path)
        
        # Сводка
        summary = self._create_summary(basic_stats, audio_stats, spectrogram_stats, quality_stats)
        
        # Сохранение результатов
        self._save_results(dataset_path.name, {
            'basic_stats': basic_stats,
            'audio_stats': audio_stats,
            'spectrogram_stats': spectrogram_stats,
            'quality_stats': quality_stats,
            'summary': summary
        })
        
        return summary
    
    def _analyze_basic_stats(self, dataset_path: Path) -> Dict[str, Any]:
        """Базовый анализ статистики"""
        print("📊 Анализ базовой статистики...")
        
        # Подсчет файлов
        audio_files = list(dataset_path.rglob("*.wav"))
        spectrogram_files = list(dataset_path.rglob("*.npy"))
        
        # Размеры файлов
        total_size = sum(f.stat().st_size for f in audio_files)
        
        # Типы файлов
        file_types = Counter(f.suffix for f in audio_files)
        
        return {
            'total_files': len(audio_files),
            'spectrogram_files': len(spectrogram_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': dict(file_types),
            'dataset_path': str(dataset_path)
        }
    
    def _analyze_audio_files(self, dataset_path: Path) -> Dict[str, Any]:
        """Анализ аудио файлов"""
        print("🎵 Анализ аудио файлов...")
        
        audio_files = list(dataset_path.rglob("*.wav"))
        
        if not audio_files:
            return {'error': 'Аудио файлы не найдены'}
        
        # Анализ первых 100 файлов (для производительности)
        sample_files = audio_files[:100]
        
        durations = []
        sample_rates = []
        channels = []
        
        for file_path in sample_files:
            try:
                y, sr = librosa.load(file_path, sr=None)
                durations.append(len(y) / sr)
                sample_rates.append(sr)
                channels.append(1 if y.ndim == 1 else y.shape[0])
            except Exception as e:
                print(f"Ошибка анализа файла {file_path}: {e}")
        
        return {
            'sample_size': len(sample_files),
            'total_files': len(audio_files),
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            },
            'sample_rate_stats': {
                'unique_rates': list(set(sample_rates)),
                'most_common_rate': Counter(sample_rates).most_common(1)[0][0]
            },
            'channel_stats': {
                'unique_channels': list(set(channels)),
                'most_common_channels': Counter(channels).most_common(1)[0][0]
            }
        }
    
    def _analyze_spectrograms(self, dataset_path: Path) -> Dict[str, Any]:
        """Анализ спектрограмм"""
        print("📈 Анализ спектрограмм...")
        
        spectrogram_files = list(dataset_path.rglob("*.npy"))
        
        if not spectrogram_files:
            return {'error': 'Спектрограммы не найдены'}
        
        # Анализ первых 50 файлов
        sample_files = spectrogram_files[:50]
        
        shapes = []
        values_stats = []
        
        for file_path in sample_files:
            try:
                spectrogram = np.load(file_path)
                shapes.append(spectrogram.shape)
                values_stats.append({
                    'mean': np.mean(spectrogram),
                    'std': np.std(spectrogram),
                    'min': np.min(spectrogram),
                    'max': np.max(spectrogram)
                })
            except Exception as e:
                print(f"Ошибка анализа спектрограммы {file_path}: {e}")
        
        return {
            'sample_size': len(sample_files),
            'total_files': len(spectrogram_files),
            'shape_stats': {
                'unique_shapes': list(set(shapes)),
                'most_common_shape': Counter(shapes).most_common(1)[0][0] if shapes else None
            },
            'value_stats': {
                'mean_of_means': np.mean([s['mean'] for s in values_stats]),
                'std_of_stds': np.std([s['std'] for s in values_stats]),
                'overall_min': min([s['min'] for s in values_stats]),
                'overall_max': max([s['max'] for s in values_stats])
            }
        }
    
    def _analyze_data_quality(self, dataset_path: Path) -> Dict[str, Any]:
        """Анализ качества данных"""
        print("🔍 Анализ качества данных...")
        
        issues = []
        
        # Проверка пустых файлов
        empty_files = []
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size == 0:
                empty_files.append(str(file_path))
        
        if empty_files:
            issues.append(f"Найдено {len(empty_files)} пустых файлов")
        
        # Проверка поврежденных файлов
        corrupted_files = []
        for file_path in dataset_path.rglob("*.wav"):
            try:
                librosa.load(file_path, sr=None)
            except Exception:
                corrupted_files.append(str(file_path))
        
        if corrupted_files:
            issues.append(f"Найдено {len(corrupted_files)} поврежденных аудио файлов")
        
        # Проверка спектрограмм
        corrupted_spectrograms = []
        for file_path in dataset_path.rglob("*.npy"):
            try:
                np.load(file_path)
            except Exception:
                corrupted_spectrograms.append(str(file_path))
        
        if corrupted_spectrograms:
            issues.append(f"Найдено {len(corrupted_spectrograms)} поврежденных спектрограмм")
        
        return {
            'issues': issues,
            'empty_files': empty_files,
            'corrupted_audio_files': corrupted_files,
            'corrupted_spectrograms': corrupted_spectrograms,
            'quality_score': max(0, 100 - len(issues) * 10)
        }
    
    def _create_summary(self, basic_stats: Dict, audio_stats: Dict, spectrogram_stats: Dict, quality_stats: Dict) -> Dict[str, Any]:
        """Создание сводки анализа"""
        return {
            'dataset_name': Path(basic_stats['dataset_path']).name,
            'total_files': basic_stats['total_files'],
            'total_size_mb': basic_stats['total_size_mb'],
            'audio_files': audio_stats.get('total_files', 0),
            'spectrogram_files': spectrogram_stats.get('total_files', 0),
            'quality_score': quality_stats['quality_score'],
            'issues_count': len(quality_stats['issues']),
            'recommendations': self._generate_recommendations(basic_stats, audio_stats, spectrogram_stats, quality_stats)
        }
    
    def _generate_recommendations(self, basic_stats: Dict, audio_stats: Dict, spectrogram_stats: Dict, quality_stats: Dict) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации по размеру
        if basic_stats['total_size_mb'] < 100:
            recommendations.append("Датасет слишком маленький, рекомендуется увеличить количество данных")
        
        # Рекомендации по качеству
        if quality_stats['quality_score'] < 80:
            recommendations.append("Низкое качество данных, рекомендуется очистка")
        
        # Рекомендации по формату
        if audio_stats.get('sample_rate_stats', {}).get('unique_rates'):
            unique_rates = audio_stats['sample_rate_stats']['unique_rates']
            if len(unique_rates) > 1:
                recommendations.append("Разные частоты дискретизации, рекомендуется стандартизация")
        
        # Рекомендации по спектрограммам
        if spectrogram_stats.get('shape_stats', {}).get('unique_shapes'):
            unique_shapes = spectrogram_stats['shape_stats']['unique_shapes']
            if len(unique_shapes) > 1:
                recommendations.append("Разные размеры спектрограмм, рекомендуется стандартизация")
        
        return recommendations
    
    def _save_results(self, dataset_name: str, results: Dict[str, Any]):
        """Сохранение результатов анализа"""
        output_dir = Path("tools/data_analysis/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение в JSON
        json_file = output_dir / f"{dataset_name}_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Сохранение в YAML
        yaml_file = output_dir / f"{dataset_name}_analysis.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📁 Результаты сохранены: {json_file}, {yaml_file}")
    
    def generate_report(self, dataset_name: str) -> str:
        """Генерация отчета в текстовом формате"""
        results_file = Path(f"tools/data_analysis/results/{dataset_name}_analysis.json")
        
        if not results_file.exists():
            return "Отчет не найден. Сначала выполните анализ датасета."
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        summary = results['summary']
        
        report = f"""
# Отчет по анализу датасета: {summary['dataset_name']}

## 📊 Общая статистика
- **Всего файлов**: {summary['total_files']}
- **Размер датасета**: {summary['total_size_mb']:.2f} MB
- **Аудио файлы**: {summary['audio_files']}
- **Спектрограммы**: {summary['spectrogram_files']}

## 🔍 Качество данных
- **Оценка качества**: {summary['quality_score']}/100
- **Найденные проблемы**: {summary['issues_count']}

## 💡 Рекомендации
"""
        
        for i, rec in enumerate(summary['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report


def main():
    """Основная функция для тестирования"""
    analyzer = DatasetAnalyzer()
    
    # Анализ основных датасетов
    datasets = [
        "/home/microWakeWord_data/positives_both",
        "/home/microWakeWord_data/negatives_both",
        "/home/microWakeWord_data/generated_features"
    ]
    
    for dataset in datasets:
        if Path(dataset).exists():
            try:
                results = analyzer.analyze_dataset(dataset)
                print(f"\n✅ Анализ завершен для {Path(dataset).name}")
                print(f"📊 Качество: {results['quality_score']}/100")
                print(f"💡 Рекомендаций: {len(results['recommendations'])}")
            except Exception as e:
                print(f"❌ Ошибка анализа {dataset}: {e}")


if __name__ == "__main__":
    main()