#!/usr/bin/env python3
"""
Реальная диагностика модели - проверяем что происходит
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

def debug_data_distribution():
    """Проверяем реальное распределение данных"""
    print("🔍 ДИАГНОСТИКА ДАННЫХ:")
    print("=" * 50)
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    # Получаем файлы
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))
    
    print(f"📁 Позитивных файлов: {len(positive_files)}")
    print(f"📁 Негативных файлов: {len(negative_files)}")
    
    # Анализируем несколько файлов каждого типа
    print(f"\n🎵 АНАЛИЗ ПОЗИТИВНЫХ ФАЙЛОВ:")
    for i, file_path in enumerate(positive_files[:3]):
        try:
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            duration = len(audio) / sr
            rms = np.sqrt(np.mean(audio**2))
            print(f"  {i+1}. {os.path.basename(file_path)}: {duration:.2f}с, RMS: {rms:.3f}")
        except Exception as e:
            print(f"  {i+1}. {os.path.basename(file_path)}: ОШИБКА - {e}")
    
    print(f"\n🎵 АНАЛИЗ НЕГАТИВНЫХ ФАЙЛОВ:")
    for i, file_path in enumerate(negative_files[:3]):
        try:
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            duration = len(audio) / sr
            rms = np.sqrt(np.mean(audio**2))
            print(f"  {i+1}. {os.path.basename(file_path)}: {duration:.2f}с, RMS: {rms:.3f}")
        except Exception as e:
            print(f"  {i+1}. {os.path.basename(file_path)}: ОШИБКА - {e}")

def create_realistic_model():
    """Создает реалистичную модель с регуляризацией"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(194, 40)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Сильная регуляризация
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_audio_real(file_path, target_sr=16000, duration_ms=1500):
    """Реальная предобработка аудио"""
    try:
        # Загружаем аудио
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Обрезаем до нужной длительности
        target_samples = int(duration_ms * target_sr / 1000)
        if len(audio) > target_samples:
            # Случайная обрезка для разнообразия
            start = np.random.randint(0, len(audio) - target_samples)
            audio = audio[start:start + target_samples]
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

def test_with_realistic_data():
    """Тестируем с реалистичными данными"""
    print("\n🧪 РЕАЛИСТИЧНОЕ ТЕСТИРОВАНИЕ:")
    print("=" * 50)
    
    # Получаем пути к данным
    positives_dir = paths.POSITIVES_RAW
    negatives_dir = paths.NEGATIVES_RAW
    
    # Получаем файлы
    positive_files = glob.glob(os.path.join(positives_dir, "*.wav"))[:50]  # 50 позитивных
    negative_files = glob.glob(os.path.join(negatives_dir, "*.wav"))[:50]  # 50 негативных
    
    print(f"📊 Тестируем на {len(positive_files)} позитивных и {len(negative_files)} негативных файлах")
    
    # Обрабатываем данные
    X = []
    y = []
    
    print("🔄 Обработка данных...")
    for file_path in positive_files:
        spectrogram = preprocess_audio_real(file_path)
        if spectrogram is not None:
            X.append(spectrogram)
            y.append(1)  # Позитивный класс
    
    for file_path in negative_files:
        spectrogram = preprocess_audio_real(file_path)
        if spectrogram is not None:
            X.append(spectrogram)
            y.append(0)  # Негативный класс
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Данные подготовлены: {X.shape}, метки: {y.shape}")
    print(f"📊 Распределение классов: {np.bincount(y)}")
    
    # Проверяем различия в данных
    pos_data = X[y == 1]
    neg_data = X[y == 0]
    
    print(f"\n📈 СТАТИСТИКИ ДАННЫХ:")
    print(f"   Позитивные: mean={np.mean(pos_data):.3f}, std={np.std(pos_data):.3f}")
    print(f"   Негативные: mean={np.mean(neg_data):.3f}, std={np.std(neg_data):.3f}")
    print(f"   Разница: {abs(np.mean(pos_data) - np.mean(neg_data)):.3f}")
    
    # Разделяем на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Train labels: {np.bincount(y_train)}")
    print(f"📊 Test labels: {np.bincount(y_test)}")
    
    # Создаем модель
    model = create_realistic_model()
    
    # Обучаем с ранней остановкой
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print(f"\n🏋️ Обучение модели...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
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
    
    print(f"\n📊 РЕАЛЬНЫЕ МЕТРИКИ:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"   AUC:       {auc:.4f}")
    print(f"   FRR:       {frr:.4f} ({frr*100:.2f}%)")
    print(f"   FAR:       {far:.4f} ({far*100:.2f}%)")
    
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   True Negatives:  {tn:3d}")
    print(f"   False Positives: {fp:3d}")
    print(f"   False Negatives: {fn:3d}")
    print(f"   True Positives:  {tp:3d}")
    
    # Анализ уверенности
    print(f"\n🎯 АНАЛИЗ УВЕРЕННОСТИ:")
    pos_confidences = y_pred_proba[y_test == 1]
    neg_confidences = y_pred_proba[y_test == 0]
    
    if len(pos_confidences) > 0:
        print(f"   Позитивные: мин={np.min(pos_confidences):.3f}, макс={np.max(pos_confidences):.3f}, среднее={np.mean(pos_confidences):.3f}")
    if len(neg_confidences) > 0:
        print(f"   Негативные: мин={np.min(neg_confidences):.3f}, макс={np.max(neg_confidences):.3f}, среднее={np.mean(neg_confidences):.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'frr': frr,
        'far': far,
        'confusion_matrix': cm
    }

def main():
    """Основная функция"""
    print("🚀 РЕАЛЬНАЯ ДИАГНОСТИКА МОДЕЛИ")
    print("=" * 60)
    
    # Диагностика данных
    debug_data_distribution()
    
    # Реалистичное тестирование
    metrics = test_with_realistic_data()
    
    print(f"\n🎉 ДИАГНОСТИКА ЗАВЕРШЕНА!")
    
    # Анализ результатов
    if metrics['accuracy'] > 0.95:
        print("⚠️ ВНИМАНИЕ: Слишком высокая точность - возможное переобучение!")
    elif metrics['accuracy'] > 0.85:
        print("✅ Хорошая точность - модель работает корректно")
    else:
        print("❌ Низкая точность - модель нуждается в улучшении")

if __name__ == "__main__":
    main()