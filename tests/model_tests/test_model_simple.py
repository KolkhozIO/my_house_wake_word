#!/usr/bin/env python3
"""
Простое тестирование модели на примерах данных
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

def analyze_audio_files(file_paths, label_name):
    """Анализирует аудио файлы без модели"""
    print(f"\n🎯 Анализ {label_name} данных:")
    print(f"📁 Файлов для анализа: {len(file_paths)}")
    
    results = []
    
    for i, file_path in enumerate(file_paths[:5]):  # Анализируем первые 5 файлов
        print(f"  📄 {i+1}/5: {os.path.basename(file_path)}")
        
        try:
            # Загружаем аудио
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            duration = len(audio) / sr
            
            # Базовый анализ
            rms = np.sqrt(np.mean(audio**2))
            zero_crossings = np.sum(librosa.zero_crossings(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            # Создаем простую спектрограмму
            stft = librosa.stft(audio, n_fft=512, hop_length=160)
            magnitude = np.abs(stft)
            
            # Mel-спектрограмма
            mel_spec = librosa.feature.melspectrogram(
                S=magnitude**2, 
                sr=sr, 
                n_mels=40,
                fmax=sr//2
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            results.append({
                'file': os.path.basename(file_path),
                'duration': duration,
                'rms': rms,
                'zero_crossings': zero_crossings,
                'spectral_centroid': spectral_centroid,
                'mel_shape': log_mel_spec.shape,
                'mel_mean': np.mean(log_mel_spec),
                'mel_std': np.std(log_mel_spec)
            })
            
            print(f"    📊 Длительность: {duration:.2f}с, RMS: {rms:.3f}, ZC: {zero_crossings}")
            print(f"    📈 Mel-спектрограмма: {log_mel_spec.shape}, mean: {np.mean(log_mel_spec):.2f}")
            
        except Exception as e:
            print(f"    ❌ Ошибка анализа: {e}")
    
    return results

def check_model_files():
    """Проверяет файлы модели"""
    model_path = "/home/microWakeWord_data/trained_models/wakeword_mixed_консервативный"
    
    print("🔍 ПРОВЕРКА ФАЙЛОВ МОДЕЛИ:")
    print("=" * 40)
    
    files_to_check = [
        "best_weights.weights.h5",
        "last_weights.weights.h5", 
        "training_config.yaml",
        "model_summary.txt"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {filename}: {size:,} bytes")
        else:
            print(f"❌ {filename}: не найден")
    
    # Проверяем директории
    dirs_to_check = ["logs", "train", "restore"]
    for dirname in dirs_to_check:
        dirpath = os.path.join(model_path, dirname)
        if os.path.exists(dirpath):
            files_count = len(os.listdir(dirpath))
            print(f"📁 {dirname}/: {files_count} файлов")
        else:
            print(f"❌ {dirname}/: не найдена")

def main():
    """Основная функция анализа"""
    print("🚀 АНАЛИЗ ДАННЫХ И МОДЕЛИ")
    print("=" * 50)
    
    # Проверяем файлы модели
    check_model_files()
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    print(f"\n📁 Позитивные данные: {positives_dir}")
    print(f"📁 Негативные данные: {negatives_dir}")
    
    # Получаем списки файлов
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:5]
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:5]
    
    print(f"📊 Найдено позитивных файлов: {len(positive_files)}")
    print(f"📊 Найдено негативных файлов: {len(negative_files)}")
    
    # Анализируем позитивные данные
    positive_results = analyze_audio_files(positive_files, "позитивных")
    
    # Анализируем негативные данные
    negative_results = analyze_audio_files(negative_files, "негативных")
    
    # Сравниваем результаты
    print("\n📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
    print("=" * 50)
    
    if positive_results and negative_results:
        # Длительность
        pos_durations = [r['duration'] for r in positive_results]
        neg_durations = [r['duration'] for r in negative_results]
        
        print(f"⏱️ Длительность:")
        print(f"   Позитивные: {np.mean(pos_durations):.2f}±{np.std(pos_durations):.2f}с")
        print(f"   Негативные: {np.mean(neg_durations):.2f}±{np.std(neg_durations):.2f}с")
        
        # RMS (громкость)
        pos_rms = [r['rms'] for r in positive_results]
        neg_rms = [r['rms'] for r in negative_results]
        
        print(f"🔊 RMS (громкость):")
        print(f"   Позитивные: {np.mean(pos_rms):.3f}±{np.std(pos_rms):.3f}")
        print(f"   Негативные: {np.mean(neg_rms):.3f}±{np.std(neg_rms):.3f}")
        
        # Спектральный центроид
        pos_centroid = [r['spectral_centroid'] for r in positive_results]
        neg_centroid = [r['spectral_centroid'] for r in negative_results]
        
        print(f"🎵 Спектральный центроид:")
        print(f"   Позитивные: {np.mean(pos_centroid):.0f}±{np.std(pos_centroid):.0f}Hz")
        print(f"   Негативные: {np.mean(neg_centroid):.0f}±{np.std(neg_centroid):.0f}Hz")
        
        # Mel-спектрограммы
        pos_mel_mean = [r['mel_mean'] for r in positive_results]
        neg_mel_mean = [r['mel_mean'] for r in negative_results]
        
        print(f"📊 Mel-спектрограмма (среднее):")
        print(f"   Позитивные: {np.mean(pos_mel_mean):.2f}±{np.std(pos_mel_mean):.2f}")
        print(f"   Негативные: {np.mean(neg_mel_mean):.2f}±{np.std(neg_mel_mean):.2f}")
        
        # Проверяем различия
        print(f"\n🔍 РАЗЛИЧИЯ:")
        duration_diff = abs(np.mean(pos_durations) - np.mean(neg_durations))
        rms_diff = abs(np.mean(pos_rms) - np.mean(neg_rms))
        centroid_diff = abs(np.mean(pos_centroid) - np.mean(neg_centroid))
        mel_diff = abs(np.mean(pos_mel_mean) - np.mean(neg_mel_mean))
        
        print(f"   Длительность: {duration_diff:.2f}с {'✅' if duration_diff > 0.1 else '⚠️'}")
        print(f"   RMS: {rms_diff:.3f} {'✅' if rms_diff > 0.01 else '⚠️'}")
        print(f"   Центроид: {centroid_diff:.0f}Hz {'✅' if centroid_diff > 100 else '⚠️'}")
        print(f"   Mel-спектрограмма: {mel_diff:.2f} {'✅' if mel_diff > 1.0 else '⚠️'}")

if __name__ == "__main__":
    main()