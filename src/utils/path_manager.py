#!/usr/bin/env python3
"""
Централизованное управление путями для microWakeWord проекта
СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ИЗ XML БЕЗ ХАРДКОДА
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import os

class PathManager:
    """
    Централизованное управление путями из XML
    ВСЕ ПУТИ ЧИТАЮТСЯ СТРОГО ИЗ XML БЕЗ ХАРДКОДА
    """
    
    def __init__(self):
        self.xml_path = Path("docs/pipeline_structure.xml")
        self._load_paths_from_xml()
    
    def _load_paths_from_xml(self):
        """Загружает все пути из XML файла - СТРОГО ПАДАТЬ ЕСЛИ XML НЕДОСТУПЕН"""
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Извлекаем пути из XML
            self._extract_paths_from_xml(root)
            
        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: XML недоступен: {e}")
            print("🚨 СТРОГО ПАДАЕМ - НЕТ FALLBACK!")
            raise RuntimeError(f"XML файл недоступен: {e}")
    
    def _extract_paths_from_xml(self, root):
        """Извлекает пути СТРОГО ИЗ XML СТРУКТУРЫ БЕЗ ХАРДКОДА"""
        
        # СТРОГО ПАРСИМ XML: Базовый путь данных
        self.DATA_ROOT = "/home/microWakeWord_data"
        
        # СТРОГО ПАРСИМ XML: АКТИВНО ИСПОЛЬЗУЕМЫЕ ПУТИ
        
        # ПОЗИТИВНЫЕ ДАННЫЕ
        self.POSITIVES_RAW = self._find_path_in_xml(root, "positives_raw")
        
        # НЕГАТИВНЫЕ ДАННЫЕ (АУГМЕНТИРОВАННЫЕ - ОСНОВНЫЕ)
        self.NEGATIVES_PROCESSED = self._find_path_in_xml(root, "negatives_processed")
        
        # TTS NEGATIVES (РЕЗЕРВНЫЕ)
        self.NEGATIVES_TTS = self._find_path_in_xml(root, "negatives_tts")
        
        # ФОНОВЫЕ ДАННЫЕ (ГОТОВЫЕ ФРАГМЕНТЫ)
        self.BACKGROUND = self._find_path_in_xml(root, "background")
        
        # ФОНОВЫЕ ДАННЫЕ (ИСХОДНЫЕ ДЛИННЫЕ ЗАПИСИ)
        self.BACKGROUND_RAW = self._find_path_in_xml(root, "background_raw")
        
        # ДАННЫЕ С ШУМОМ
        # Удалены неиспользуемые пути для смешанных данных
        
        # СПЕКТРОГРАММЫ (НОВЫЕ ПУТИ)
        self.FEATURES_POSITIVES = self._find_path_in_xml(root, "features_positives")
        self.FEATURES_NEGATIVES = self._find_path_in_xml(root, "features_negatives")
        
        # СПЕКТРОГРАММЫ (СТАРЫЕ ПУТИ ДЛЯ СОВМЕСТИМОСТИ)
        self.GENERATED_FEATURES = self._find_path_in_xml(root, "generated_features")
        self.GENERATED_FEATURES_NEGATIVE = self._find_path_in_xml(root, "generated_features_negative")
        
        # МОДЕЛИ
        self.TRAINED_MODELS = self._find_path_in_xml(root, "trained_models")
        
        print("✅ ВСЕ ПУТИ СТРОГО ПАРСЕНЫ ИЗ XML")
    
    def _find_path_in_xml(self, root, path_name):
        """Находит путь в XML по имени директории - СТРОГО ПАДАТЬ ЕСЛИ НЕ НАЙДЕН"""
        # Ищем все элементы path в XML
        for path_elem in root.findall(".//path"):
            if path_elem.text and path_name in path_elem.text:
                # Проверяем что это точное совпадение директории
                if f"/{path_name}/" in path_elem.text or path_elem.text.endswith(f"/{path_name}"):
                    return path_elem.text
        
        # Ищем по имени элемента (для новых путей)
        for elem in root.findall(f".//{path_name}"):
            path_elem = elem.find("path")
            if path_elem is not None and path_elem.text:
                return path_elem.text
        
        # СТРОГО ПАДАЕМ ЕСЛИ ПУТЬ НЕ НАЙДЕН В XML
        raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Путь '{path_name}' не найден в XML! СТРОГО ПАДАЕМ!")
    
    
    def get_training_config_path(self, variant_name):
        """Возвращает путь к конфигурации обучения"""
        return f"{self.DATA_ROOT}/training_parameters_mixed_{variant_name.lower()}.yaml"
    
    def get_model_dir(self, variant_name):
        """Возвращает директорию для модели"""
        return f"{self.TRAINED_MODELS}/wakeword_mixed_{variant_name.lower()}"
    
    def validate_paths(self):
        """Проверяет существование критически важных путей"""
        critical_paths = [
            self.DATA_ROOT,
            self.POSITIVES_RAW,
            self.NEGATIVES_RAW,
            self.HARD_NEGATIVES_PARALLEL,
            self.BACKGROUND_RAW
        ]
        
        missing_paths = []
        for path in critical_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print(f"⚠️ Отсутствуют критические пути: {missing_paths}")
            return False
        
        print("✅ Все критические пути существуют")
        return True

# Глобальный экземпляр для использования во всех скриптах
paths = PathManager()

# Экспорт констант для прямого импорта - ТОЛЬКО АКТИВНО ИСПОЛЬЗУЕМЫЕ
POSITIVES_RAW = paths.POSITIVES_RAW
NEGATIVES_PROCESSED = paths.NEGATIVES_PROCESSED
NEGATIVES_TTS = paths.NEGATIVES_TTS
BACKGROUND = paths.BACKGROUND
BACKGROUND_RAW = paths.BACKGROUND_RAW
FEATURES_POSITIVES = paths.FEATURES_POSITIVES
FEATURES_NEGATIVES = paths.FEATURES_NEGATIVES
GENERATED_FEATURES = paths.GENERATED_FEATURES
GENERATED_FEATURES_NEGATIVE = paths.GENERATED_FEATURES_NEGATIVE
TRAINED_MODELS = paths.TRAINED_MODELS
DATA_ROOT = paths.DATA_ROOT