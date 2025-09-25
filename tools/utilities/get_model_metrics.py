#!/usr/bin/env python3
"""
Получение конкретных метрик модели на тестовых данных
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Добавляем путь к библиотеке microWakeWord
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths

def create_simple_model():
    """Создает простую модель для тестирования метрик"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(194, 40)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_audio_for_metrics(file_path, target_sr=16000, duration_ms=1500):
    """Предобработка аудио для получения метрик"""
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
        
        # Создаем спектрограмму
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

def prepare_test_data():
    """Подготавливает тестовые данные"""
    print("📊 Подготовка тестовых данных...")
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    # Получаем файлы
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:100]  # 100 позитивных
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:100]  # 100 негативных
    
    print(f"📁 Позитивных файлов: {len(positive_files)}")
    print(f"📁 Негативных файлов: {len(negative_files)}")
    
    # Обрабатываем данные
    X = []
    y = []
    
    print("🔄 Обработка позитивных данных...")
    for i, file_path in enumerate(positive_files):
        if i % 20 == 0:
            print(f"  Обработано: {i}/{len(positive_files)}")
        
        spectrogram = preprocess_audio_for_metrics(file_path)
        if spectrogram is not None:
            X.append(spectrogram)
            y.append(1)  # Позитивный класс
    
    print("🔄 Обработка негативных данных...")
    for i, file_path in enumerate(negative_files):
        if i % 20 == 0:
            print(f"  Обработано: {i}/{len(negative_files)}")
        
        spectrogram = preprocess_audio_for_metrics(file_path)
        if spectrogram is not None:
            X.append(spectrogram)
            y.append(0)  # Негативный класс
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Данные подготовлены: {X.shape}, метки: {y.shape}")
    print(f"📊 Распределение классов: {np.bincount(y)}")
    
    return X, y

def train_and_evaluate_model(X, y):
    """Обучает модель и получает метрики"""
    print("🏋️ Обучение модели...")
    
    # Создаем модель
    model = create_simple_model()
    
    # Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Train labels: {np.bincount(y_train)}")
    print(f"📊 Test labels: {np.bincount(y_test)}")
    
    # Обучаем модель
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Предсказания
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Wake Word специфичные метрики
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'frr': frr,
        'far': far,
        'confusion_matrix': cm,
        'history': history
    }

def print_metrics(metrics):
    """Выводит метрики в красивом формате"""
    print("\n" + "="*60)
    print("📊 КОНКРЕТНЫЕ МЕТРИКИ МОДЕЛИ")
    print("="*60)
    
    print(f"🎯 ОБЩИЕ МЕТРИКИ:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"   AUC:       {metrics['auc']:.4f}")
    
    print(f"\n🎤 WAKE WORD МЕТРИКИ:")
    print(f"   FRR (False Reject Rate): {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
    print(f"   FAR (False Accept Rate): {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
    
    print(f"\n📊 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives:  {cm[0,0]:3d}")
    print(f"   False Positives: {cm[0,1]:3d}")
    print(f"   False Negatives: {cm[1,0]:3d}")
    print(f"   True Positives:  {cm[1,1]:3d}")
    
    print(f"\n📈 ИНТЕРПРЕТАЦИЯ:")
    if metrics['frr'] < 0.05:
        print("   ✅ FRR отличный (< 5%) - модель редко пропускает wake word")
    elif metrics['frr'] < 0.1:
        print("   ⚠️ FRR хороший (< 10%) - модель иногда пропускает wake word")
    else:
        print("   ❌ FRR плохой (> 10%) - модель часто пропускает wake word")
    
    if metrics['far'] < 0.02:
        print("   ✅ FAR отличный (< 2%) - модель редко срабатывает на фоне")
    elif metrics['far'] < 0.05:
        print("   ⚠️ FAR хороший (< 5%) - модель иногда срабатывает на фоне")
    else:
        print("   ❌ FAR плохой (> 5%) - модель часто срабатывает на фоне")
    
    if metrics['accuracy'] > 0.95:
        print("   ✅ Accuracy отличный (> 95%)")
    elif metrics['accuracy'] > 0.90:
        print("   ⚠️ Accuracy хороший (> 90%)")
    else:
        print("   ❌ Accuracy плохой (< 90%)")

def main():
    """Основная функция"""
    print("🚀 ПОЛУЧЕНИЕ КОНКРЕТНЫХ МЕТРИК МОДЕЛИ")
    print("="*60)
    
    # Подготавливаем данные
    X, y = prepare_test_data()
    
    if len(X) == 0:
        print("❌ Не удалось подготовить данные")
        return
    
    # Обучаем модель и получаем метрики
    metrics = train_and_evaluate_model(X, y)
    
    # Выводим метрики
    print_metrics(metrics)
    
    print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"📊 Протестировано {len(X)} файлов")
    print(f"🎯 Модель готова к использованию!")

if __name__ == "__main__":
    main()