#!/bin/bash

echo "=== АВТОМАТИЧЕСКИЙ МОНИТОРИНГ ПАЙПЛАЙНА ==="
echo "Время: $(date)"
echo ""

# Проверка статуса задач
echo "📋 СТАТУС ЗАДАЧ:"
./manage_tasks.sh status
echo ""

# Проверка логов активных задач
echo "📊 ЛОГИ АКТИВНЫХ ЗАДАЧ:"
if ./manage_tasks.sh status | grep -q "train_model.*running"; then
    echo "🔄 train_model (обучение модели):"
    tail -3 training.log 2>/dev/null || echo "Лог не найден"
fi
echo ""

# Проверка ресурсов системы
echo "💻 РЕСУРСЫ СИСТЕМЫ:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "RAM: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "Диск: $(df -h /home/microWakeWord_data | tail -1 | awk '{print $3"/"$2" ("$5")"}')"
echo ""

# Проверка завершенных этапов
echo "✅ ЗАВЕРШЕННЫЕ ЭТАПЫ:"
echo "negatives_real: $(find /home/microWakeWord_data/negatives_real_sampled -name "*.wav" 2>/dev/null | wc -l) семплов"
echo "background_data: $(find /home/microWakeWord_data/background_data_sampled -name "*.wav" 2>/dev/null | wc -l) семплов"
echo ""

# Автоматические действия
echo "🤖 АВТОМАТИЧЕСКИЕ ДЕЙСТВИЯ:"

# Если обучение завершилось, проверяем результат
if ./manage_tasks.sh status | grep -q "train_model.*finished"; then
    echo "🎉 Обучение завершено! Проверяем результат..."
    if [ -f "/home/microWakeWord_data/original_library_model.tflite" ]; then
        echo "✅ Модель создана: $(ls -lh /home/microWakeWord_data/original_library_model.tflite | awk '{print $5}')"
        echo "🎯 Пайплайн полностью завершен!"
    else
        echo "❌ Модель не найдена!"
    fi
fi

echo ""

# Проверка ошибок системы
echo "🚨 ПРОВЕРКА ОШИБОК:"
if dmesg | tail -10 | grep -i "killed\|oom\|error" > /dev/null; then
    echo "⚠️ Найдены ошибки в системных логах!"
    dmesg | tail -5 | grep -i "killed\|oom\|error"
else
    echo "✅ Системных ошибок не найдено"
fi
echo ""

echo "=== СЛЕДУЮЩАЯ ПРОВЕРКА ЧЕРЕЗ 10 МИНУТ ==="
echo ""

