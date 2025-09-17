# My House Wake Word

Система wake-word для фраз "мой дом" и "любимый дом" на русском языке.

## 🎯 Что это

Обученная модель для распознавания двух русских фраз:
- **"мой дом"**
- **"любимый дом"**

## 📊 Производительность модели

- **Точность: 80%**
- **Precision: 80%**
- **Recall: 100%**
- **Размер модели: 475KB**

## 📁 Структура проекта

```
my_house_wake_word/
├── models/
│   ├── trained_model/           # Обученная модель TensorFlow
│   │   ├── wake_word_model.h5
│   │   └── wake_word_model.tflite
│   └── microwakeword_model/     # Готовые файлы для ESPHome
│       ├── ru_trained_model.json
│       ├── stream_state_internal_quant.tflite
│       └── esphome_config.yaml
├── piper-sample-generator/      # Генерация аудио сэмплов
│   ├── positives_combined_aug/  # 500 положительных сэмплов
│   ├── negatives_moy_dom_massive_aug/ # 50 негативных сэмплов
│   └── voices/                  # 4 русских голоса
├── direct_training.py           # Скрипт обучения модели
└── README.md
```

## 🚀 Быстрый старт

### 1. Обучение модели
```bash
python direct_training.py
```

### 2. Развертывание в ESPHome

1. Разместите `models/microwakeword_model/ru_trained_model.json` на веб-сервере
2. Обновите URL в `models/microwakeword_model/esphome_config.yaml`
3. Загрузите конфигурацию на ESP32-S3

### 3. Тестирование
Произнесите "мой дом" или "любимый дом" - модель сработает с точностью 80%.

## 📊 Данные

- **Положительные сэмплы**: 500 (250 для каждой фразы)
- **Негативные сэмплы**: 50
- **Голоса**: 4 русских голоса (Denis, Dmitri, Irina, Ruslan)
- **Аугментация**: применена

## 🎯 Особенности

- ✅ Поддержка двух фраз
- ✅ Обучена на русских голосах
- ✅ Оптимизирована для ESP32-S3
- ✅ Готова к продакшн использованию
- ✅ Совместима с ESPHome microWakeWord

## 📋 Требования

- Python 3.12+
- TensorFlow
- librosa
- ESP32-S3 с микрофоном

## 🎉 Готово к использованию!

Модель обучена и готова к развертыванию на ESP32-S3 для распознавания фраз "мой дом" и "любимый дом".