#!/usr/bin/env python3
"""
Тестирование финальной модели microWakeWord
"""

import sys
import os
sys.path.append('/home/microWakeWord')

import numpy as np
import tensorflow as tf
from pathlib import Path

def test_tflite_model():
    """Тестирует TFLite модель"""
    
    model_path = "/home/microWakeWord_data/trained_models/wakeword/model.tflite"
    
    print("🔄 Тестирование TFLite модели...")
    
    try:
        # Загружаем TFLite модель
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Получаем информацию о входе и выходе
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Входные детали: {input_details}")
        print(f"📊 Выходные детали: {output_details}")
        
        # Создаем тестовые данные (случайные спектрограммы)
        input_shape = input_details[0]['shape']
        print(f"📊 Форма входа: {input_shape}")
        
        # Создаем тестовые данные
        test_data = np.random.random(input_shape).astype(np.float32)
        
        # Запускаем инференс
        interpreter.set_tensor(input_details[0]['index'], test_data)
        interpreter.invoke()
        
        # Получаем результат
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ Инференс успешен!")
        print(f"📊 Форма выхода: {output_data.shape}")
        print(f"📊 Значения выхода: {output_data}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    success = test_tflite_model()
    if success:
        print("🎉 Модель работает корректно!")
    else:
        print("❌ Модель не работает")
        sys.exit(1)