#!/usr/bin/env python3
"""
Правильное тестирование обученной модели с оригинальной архитектурой
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import glob

# Добавляем путь к библиотеке microWakeWord
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths
from microwakeword import mixednet
from microwakeword.layers import modes

def create_model():
    """Создает модель с оригинальной архитектурой microWakeWord"""
    print("🏗️ Создание модели с оригинальной архитектурой...")
    
    # Параметры модели из training_config.yaml
    flags = {
        'attention_dim': 64,
        'attention_heads': 1,
        'first_conv_filters': 32,
        'first_conv_kernel_size': 3,
        'max_pool': False,
        'mixconv_kernel_sizes': '[5],[9],[13],[21]',
        'pointwise_filters': '48,48,48,48',
        'pooled': False,
        'repeat_in_block': '1,1,1,1',
        'residual_connection': '0,0,0,0',
        'spatial_attention': False,
        'stride': 1,
        'temporal_attention': False
    }
    
    # Создаем модель с правильными параметрами
    model = mixednet.model(
        flags=flags,
        shape=(194, 40),
        batch_size=1
    )
    
    print("✅ Модель создана")
    return model

def load_trained_weights(model):
    """Загружает обученные веса"""
    weights_path = "/home/microWakeWord_data/trained_models/wakeword_mixed_консервативный/best_weights.weights.h5"
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("✅ Веса загружены успешно")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки весов: {e}")
            return False
    else:
        print("❌ Файл весов не найден")
        return False

def preprocess_audio_proper(file_path, target_sr=16000, duration_ms=1500):
    """Правильная предобработка аудио в формате microWakeWord"""
    try:
        # Загружаем аудио
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Обрезаем до нужной длительности
        target_samples = int(duration_ms * target_sr / 1000)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            # Дополняем нулями если короткий
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # Создаем спектрограмму как в microWakeWord
        # Используем параметры из конфигурации
        n_fft = 512
        hop_length = 160
        n_mels = 40
        
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Mel-спектрограмма
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude**2, 
            sr=target_sr, 
            n_mels=n_mels,
            fmax=target_sr//2
        )
        
        # Логарифмическое масштабирование
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Обрезаем до нужного размера (194, 40)
        if log_mel_spec.shape[1] > 194:
            log_mel_spec = log_mel_spec[:, :194]
        else:
            # Дополняем нулями
            pad_width = 194 - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)))
        
        return log_mel_spec.T  # Транспонируем для соответствия (194, 40)
    
    except Exception as e:
        print(f"❌ Ошибка обработки {file_path}: {e}")
        return None

def test_model_on_files(model, file_paths, expected_label, label_name):
    """Тестирует модель на списке файлов"""
    print(f"\n🎯 Тестирование {label_name} данных:")
    print(f"📁 Файлов для тестирования: {len(file_paths)}")
    
    results = []
    
    for i, file_path in enumerate(file_paths[:5]):  # Тестируем первые 5 файлов
        print(f"  📄 {i+1}/5: {os.path.basename(file_path)}")
        
        # Предобрабатываем аудио
        spectrogram = preprocess_audio_proper(file_path)
        if spectrogram is None:
            continue
        
        # Подготавливаем данные для модели
        input_data = np.expand_dims(spectrogram, axis=0)  # Добавляем batch dimension
        
        # Предсказание
        try:
            prediction = model.predict(input_data, verbose=0)
            confidence = float(prediction[0][0])
            
            # Определяем предсказанный класс
            predicted_class = 1 if confidence > 0.5 else 0
            
            # Проверяем правильность
            is_correct = predicted_class == expected_label
            
            results.append({
                'file': os.path.basename(file_path),
                'confidence': confidence,
                'predicted': predicted_class,
                'expected': expected_label,
                'correct': is_correct
            })
            
            status = "✅" if is_correct else "❌"
            print(f"    {status} Confidence: {confidence:.3f}, Predicted: {predicted_class}, Expected: {expected_label}")
            
        except Exception as e:
            print(f"    ❌ Ошибка предсказания: {e}")
    
    return results

def main():
    """Основная функция тестирования"""
    print("🚀 ПРАВИЛЬНОЕ ТЕСТИРОВАНИЕ МОДЕЛИ НА ПРИМЕРАХ ДАННЫХ")
    print("=" * 60)
    
    # Создаем модель
    model = create_model()
    
    # Загружаем веса
    weights_loaded = load_trained_weights(model)
    
    if not weights_loaded:
        print("⚠️ Тестируем модель без обученных весов")
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    print(f"📁 Позитивные данные: {positives_dir}")
    print(f"📁 Негативные данные: {negatives_dir}")
    
    # Получаем списки файлов
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:5]
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:5]
    
    print(f"📊 Найдено позитивных файлов: {len(positive_files)}")
    print(f"📊 Найдено негативных файлов: {len(negative_files)}")
    
    # Тестируем на позитивных данных
    positive_results = test_model_on_files(model, positive_files, 1, "позитивных")
    
    # Тестируем на негативных данных
    negative_results = test_model_on_files(model, negative_files, 0, "негативных")
    
    # Анализируем результаты
    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)
    
    # Позитивные результаты
    if positive_results:
        correct_positives = sum(1 for r in positive_results if r['correct'])
        total_positives = len(positive_results)
        avg_confidence_positives = np.mean([r['confidence'] for r in positive_results])
        
        print(f"✅ Позитивные данные:")
        print(f"   Правильно: {correct_positives}/{total_positives} ({correct_positives/total_positives*100:.1f}%)")
        print(f"   Средняя уверенность: {avg_confidence_positives:.3f}")
    
    # Негативные результаты
    if negative_results:
        correct_negatives = sum(1 for r in negative_results if r['correct'])
        total_negatives = len(negative_results)
        avg_confidence_negatives = np.mean([r['confidence'] for r in negative_results])
        
        print(f"❌ Негативные данные:")
        print(f"   Правильно: {correct_negatives}/{total_negatives} ({correct_negatives/total_negatives*100:.1f}%)")
        print(f"   Средняя уверенность: {avg_confidence_negatives:.3f}")
    
    # Общие результаты
    all_results = positive_results + negative_results
    if all_results:
        total_correct = sum(1 for r in all_results if r['correct'])
        total_tests = len(all_results)
        overall_accuracy = total_correct / total_tests * 100
        
        print(f"\n🎯 ОБЩАЯ ТОЧНОСТЬ: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
        
        # Анализ уверенности
        positive_confidences = [r['confidence'] for r in positive_results]
        negative_confidences = [r['confidence'] for r in negative_results]
        
        if positive_confidences:
            print(f"📈 Позитивные: мин={min(positive_confidences):.3f}, макс={max(positive_confidences):.3f}")
        if negative_confidences:
            print(f"📉 Негативные: мин={min(negative_confidences):.3f}, макс={max(negative_confidences):.3f}")
        
        # Анализ для wake word
        print(f"\n🎤 АНАЛИЗ ДЛЯ WAKE WORD:")
        print(f"   FRR (False Reject Rate): {len([r for r in positive_results if not r['correct']])/len(positive_results)*100:.1f}%")
        print(f"   FAR (False Accept Rate): {len([r for r in negative_results if not r['correct']])/len(negative_results)*100:.1f}%")

if __name__ == "__main__":
    main()