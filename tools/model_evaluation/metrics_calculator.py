#!/usr/bin/env python3
"""
Калькулятор метрик для моделей microWakeWord
"""

import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Калькулятор метрик для моделей wake word"""
    
    def __init__(self, model_path: str = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.results = {}
    
    def load_model(self, model_path: str):
        """Загрузка модели"""
        self.model_path = Path(model_path)
        
        if self.model_path.suffix == '.tflite':
            self.model = self._load_tflite_model(model_path)
        else:
            self.model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Модель загружена: {model_path}")
    
    def _load_tflite_model(self, model_path: str):
        """Загрузка TFLite модели"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """Расчет всех метрик"""
        print("📊 Расчет метрик...")
        
        # Базовые метрики
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Wake word специфичные метрики
        wake_word_metrics = self._calculate_wake_word_metrics(y_true, y_pred, y_prob)
        
        # Матрица ошибок
        confusion_mat = confusion_matrix(y_true, y_pred)
        
        # Детальный отчет
        classification_rep = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC AUC (если есть вероятности)
        roc_auc = None
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                roc_auc = None
        
        metrics = {
            'basic_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc is not None else None
            },
            'wake_word_metrics': wake_word_metrics,
            'confusion_matrix': confusion_mat.tolist(),
            'classification_report': classification_rep,
            'data_info': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true)),
                'negative_samples': int(len(y_true) - np.sum(y_true)),
                'positive_ratio': float(np.sum(y_true) / len(y_true))
            }
        }
        
        return metrics
    
    def _calculate_wake_word_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """Расчет wake word специфичных метрик"""
        
        # False Reject Rate (FRR) - пропуск wake word
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        frr = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # False Accept Rate (FAR) - ложное срабатывание
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # False Accepts per Hour (FA/h) - ложные срабатывания в час
        # Предполагаем 1 секунду на образец
        total_hours = len(y_true) / 3600
        fa_per_hour = false_positives / total_hours if total_hours > 0 else 0
        
        # Precision и Recall для wake word
        wake_word_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        wake_word_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 для wake word
        wake_word_f1 = 2 * (wake_word_precision * wake_word_recall) / (wake_word_precision + wake_word_recall) if (wake_word_precision + wake_word_recall) > 0 else 0
        
        return {
            'false_reject_rate': float(frr),
            'false_accept_rate': float(far),
            'false_accepts_per_hour': float(fa_per_hour),
            'wake_word_precision': float(wake_word_precision),
            'wake_word_recall': float(wake_word_recall),
            'wake_word_f1': float(wake_word_f1),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
    
    def evaluate_model_performance(self, test_data: Tuple[np.ndarray, np.ndarray], model_path: str = None) -> Dict[str, Any]:
        """Полная оценка производительности модели"""
        if model_path:
            self.load_model(model_path)
        
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        X_test, y_test = test_data
        
        print(f"🔍 Оценка модели на {len(X_test)} образцах...")
        
        # Предсказания
        if isinstance(self.model, tf.lite.Interpreter):
            y_pred, y_prob = self._predict_tflite(X_test)
        else:
            y_prob = self.model.predict(X_test)
            y_pred = (y_prob > 0.5).astype(int).flatten()
            y_prob = y_prob.flatten()
        
        # Расчет метрик
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Анализ производительности
        performance_analysis = self._analyze_performance(metrics)
        
        # Генерация рекомендаций
        recommendations = self._generate_recommendations(metrics)
        
        results = {
            'model_path': str(self.model_path) if self.model_path else None,
            'metrics': metrics,
            'performance_analysis': performance_analysis,
            'recommendations': recommendations,
            'evaluation_summary': self._create_summary(metrics, performance_analysis)
        }
        
        return results
    
    def _predict_tflite(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказания для TFLite модели"""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        predictions = []
        probabilities = []
        
        for sample in X_test:
            # Подготовка входных данных
            input_data = sample.reshape(input_details[0]['shape']).astype(input_details[0]['dtype'])
            
            # Установка входных данных
            self.model.set_tensor(input_details[0]['index'], input_data)
            
            # Выполнение инференса
            self.model.invoke()
            
            # Получение результатов
            output_data = self.model.get_tensor(output_details[0]['index'])
            probabilities.append(output_data[0])
            predictions.append(1 if output_data[0] > 0.5 else 0)
        
        return np.array(predictions), np.array(probabilities)
    
    def _analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ производительности модели"""
        wake_word_metrics = metrics['wake_word_metrics']
        basic_metrics = metrics['basic_metrics']
        
        analysis = {
            'overall_score': 0,
            'strengths': [],
            'weaknesses': [],
            'performance_level': 'unknown'
        }
        
        # Анализ FRR (должен быть низким)
        frr = wake_word_metrics['false_reject_rate']
        if frr < 0.01:
            analysis['strengths'].append("Отличный FRR (< 1%)")
            analysis['overall_score'] += 25
        elif frr < 0.05:
            analysis['strengths'].append("Хороший FRR (< 5%)")
            analysis['overall_score'] += 20
        else:
            analysis['weaknesses'].append(f"Высокий FRR ({frr:.1%})")
        
        # Анализ FA/h (должен быть низким)
        fa_per_hour = wake_word_metrics['false_accepts_per_hour']
        if fa_per_hour < 0.5:
            analysis['strengths'].append("Отличный FA/h (< 0.5)")
            analysis['overall_score'] += 25
        elif fa_per_hour < 2.0:
            analysis['strengths'].append("Хороший FA/h (< 2.0)")
            analysis['overall_score'] += 20
        else:
            analysis['weaknesses'].append(f"Высокий FA/h ({fa_per_hour:.1f})")
        
        # Анализ точности
        accuracy = basic_metrics['accuracy']
        if accuracy > 0.95:
            analysis['strengths'].append("Высокая точность (> 95%)")
            analysis['overall_score'] += 25
        elif accuracy > 0.90:
            analysis['strengths'].append("Хорошая точность (> 90%)")
            analysis['overall_score'] += 20
        else:
            analysis['weaknesses'].append(f"Низкая точность ({accuracy:.1%})")
        
        # Анализ F1 для wake word
        wake_word_f1 = wake_word_metrics['wake_word_f1']
        if wake_word_f1 > 0.9:
            analysis['strengths'].append("Отличный F1 для wake word (> 90%)")
            analysis['overall_score'] += 25
        elif wake_word_f1 > 0.8:
            analysis['strengths'].append("Хороший F1 для wake word (> 80%)")
            analysis['overall_score'] += 20
        else:
            analysis['weaknesses'].append(f"Низкий F1 для wake word ({wake_word_f1:.1%})")
        
        # Определение уровня производительности
        if analysis['overall_score'] >= 90:
            analysis['performance_level'] = 'excellent'
        elif analysis['overall_score'] >= 75:
            analysis['performance_level'] = 'good'
        elif analysis['overall_score'] >= 60:
            analysis['performance_level'] = 'fair'
        else:
            analysis['performance_level'] = 'poor'
        
        return analysis
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по улучшению"""
        recommendations = []
        wake_word_metrics = metrics['wake_word_metrics']
        basic_metrics = metrics['basic_metrics']
        
        # Рекомендации по FRR
        frr = wake_word_metrics['false_reject_rate']
        if frr > 0.05:
            recommendations.append("Высокий FRR - рассмотрите увеличение веса позитивного класса или улучшение качества данных")
        
        # Рекомендации по FA/h
        fa_per_hour = wake_word_metrics['false_accepts_per_hour']
        if fa_per_hour > 2.0:
            recommendations.append("Высокий FA/h - рассмотрите увеличение порога вероятности или улучшение негативных данных")
        
        # Рекомендации по точности
        accuracy = basic_metrics['accuracy']
        if accuracy < 0.90:
            recommendations.append("Низкая точность - рассмотрите увеличение размера модели или улучшение данных")
        
        # Рекомендации по дисбалансу
        positive_ratio = metrics['data_info']['positive_ratio']
        if positive_ratio < 0.01 or positive_ratio > 0.1:
            recommendations.append("Неоптимальный дисбаланс данных - рассмотрите балансировку датасета")
        
        # Рекомендации по ROC AUC
        roc_auc = basic_metrics['roc_auc']
        if roc_auc is not None and roc_auc < 0.85:
            recommendations.append("Низкий ROC AUC - рассмотрите улучшение архитектуры модели")
        
        return recommendations
    
    def _create_summary(self, metrics: Dict[str, Any], performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Создание сводки оценки"""
        wake_word_metrics = metrics['wake_word_metrics']
        basic_metrics = metrics['basic_metrics']
        
        return {
            'model_name': self.model_path.name if self.model_path else 'Unknown',
            'overall_score': performance_analysis['overall_score'],
            'performance_level': performance_analysis['performance_level'],
            'key_metrics': {
                'accuracy': basic_metrics['accuracy'],
                'frr': wake_word_metrics['false_reject_rate'],
                'fa_per_hour': wake_word_metrics['false_accepts_per_hour'],
                'wake_word_f1': wake_word_metrics['wake_word_f1']
            },
            'strengths_count': len(performance_analysis['strengths']),
            'weaknesses_count': len(performance_analysis['weaknesses']),
            'recommendations_count': len(self._generate_recommendations(metrics))
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Сохранение результатов оценки"""
        if output_path is None:
            model_name = self.model_path.stem if self.model_path else 'unknown_model'
            output_path = f"tools/model_evaluation/results/{model_name}_evaluation.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Сохранение в YAML
        yaml_path = output_path.with_suffix('.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"📁 Результаты сохранены: {output_path}, {yaml_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Генерация текстового отчета"""
        summary = results['evaluation_summary']
        metrics = results['metrics']
        wake_word_metrics = metrics['wake_word_metrics']
        basic_metrics = metrics['basic_metrics']
        
        report = f"""
# Отчет по оценке модели: {summary['model_name']}

## 📊 Общая оценка
- **Общий балл**: {summary['overall_score']}/100
- **Уровень производительности**: {summary['performance_level'].upper()}
- **Сильные стороны**: {summary['strengths_count']}
- **Слабые стороны**: {summary['weaknesses_count']}

## 🎯 Ключевые метрики
- **Точность**: {summary['key_metrics']['accuracy']:.1%}
- **FRR (False Reject Rate)**: {summary['key_metrics']['frr']:.1%}
- **FA/h (False Accepts per Hour)**: {summary['key_metrics']['fa_per_hour']:.1f}
- **F1 для wake word**: {summary['key_metrics']['wake_word_f1']:.1%}

## 📈 Детальные метрики
- **Precision**: {basic_metrics['precision']:.1%}
- **Recall**: {basic_metrics['recall']:.1%}
- **F1 Score**: {basic_metrics['f1_score']:.1%}
- **ROC AUC**: {basic_metrics['roc_auc']:.3f if basic_metrics['roc_auc'] else 'N/A'}

## 💡 Рекомендации
"""
        
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report


def main():
    """Основная функция для тестирования"""
    calculator = MetricsCalculator()
    
    # Пример использования
    print("🔍 Калькулятор метрик microWakeWord")
    print("Для полной оценки загрузите модель и тестовые данные")


if __name__ == "__main__":
    main()