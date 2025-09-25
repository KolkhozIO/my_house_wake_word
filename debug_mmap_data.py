#!/usr/bin/env python3
"""
Отладка данных RaggedMmap
"""

import sys
import os
sys.path.append('/home/microWakeWord')

import numpy as np
from pathlib import Path

def debug_mmap_data():
    """Отлаживает данные RaggedMmap"""
    
    print("🔍 Отладка данных RaggedMmap")
    print("=" * 40)
    
    # Позитивные данные
    positives_dir = "/home/microWakeWord_data/features_positives/training/wakeword_mmap"
    print(f"📁 Позитивные данные: {positives_dir}")
    
    try:
        from microwakeword.data import RaggedMmap
        
        data = RaggedMmap(positives_dir)
        print(f"📊 Размер данных: {len(data)}")
        
        # Проверяем первые несколько образцов
        for i in range(min(5, len(data))):
            spec = data[i]
            print(f"  Образец {i}: форма = {spec.shape}, тип = {type(spec)}")
            if hasattr(spec, 'shape'):
                print(f"    Детали формы: {spec.shape}")
            else:
                print(f"    Это не массив: {spec}")
        
        # Проверяем случайные образцы
        import random
        indices = random.sample(range(len(data)), min(10, len(data)))
        print(f"\n📊 Проверка случайных образцов: {indices}")
        
        valid_count = 0
        for idx in indices:
            spec = data[idx]
            if hasattr(spec, 'shape') and spec.shape == (194, 40):
                valid_count += 1
            else:
                print(f"  Образец {idx}: форма = {spec.shape if hasattr(spec, 'shape') else 'не массив'}")
        
        print(f"✅ Валидных образцов: {valid_count}/{len(indices)}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mmap_data()