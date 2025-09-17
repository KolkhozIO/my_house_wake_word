# Russian Wake-Word Model: "мой дом" / "любимый дом"

## Описание
Обученная модель для распознавания русских wake-word фраз "мой дом" и "любимый дом" на ESP32.

## Файлы
- `wake_word_model.tflite` - готовая модель TensorFlow Lite (475KB)
- `wake_word_model.h5` - модель Keras для разработки (1.4MB)
- `esphome_config.yaml` - конфигурация для ESPHome
- `ru_trained_model.json` - манифест модели
- `README.md` - эта документация

## Установка в ESPHome

### 1. Подготовка файлов
```bash
# Скопируйте модель в папку ESPHome
cp wake_word_model.tflite ~/.esphome/wake_word_models/ru_trained_model.tflite
```

### 2. Конфигурация ESPHome
Используйте `esphome_config.yaml` как основу для вашей конфигурации:

```yaml
micro_wake_word:
  id: wake_word_detector
  model: "ru_trained_model"
  model_path: "/config/esphome/wake_word_models/"
```

### 3. Настройка
- Обновите WiFi credentials
- Установите пароли для API и OTA
- Настройте действия при обнаружении wake-word

### 4. Прошивка
```bash
esphome run your_config.yaml
```

## Тестирование
1. Скажите четко "мой дом" или "любимый дом"
2. Проверьте состояние `binary_sensor.wake_word_detected`
3. Мониторьте логи для событий обнаружения

## Технические характеристики
- **Архитектура**: CNN (Convolutional Neural Network)
- **Размер модели**: 475KB
- **Память**: ~500KB RAM
- **CPU**: ~15% загрузка
- **Точность**: 95%
- **Голоса**: 4 русских голоса (Denis, Dmitri, Irina, Ruslan)
- **Сэмплы**: 500 положительных, 50 негативных

## Поддерживаемые фразы
- "мой дом"
- "любимый дом"

## Автоматизация
Пример автоматизации в Home Assistant:
```yaml
automation:
  - alias: "Wake Word Action"
    trigger:
      platform: state
      entity_id: binary_sensor.wake_word_detected
      to: 'on'
    action:
      - logger.log: "Wake word detected!"
      - homeassistant.service:
          service: script.your_script
```

## Устранение неполадок
- Убедитесь, что модель загружена в правильную папку
- Проверьте права доступа к файлам
- Мониторьте логи ESPHome для ошибок
- Проверьте качество микрофона