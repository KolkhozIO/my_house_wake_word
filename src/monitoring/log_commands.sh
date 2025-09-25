#!/bin/bash
# Удобные команды для просмотра логов microWakeWord

# Основные команды
alias logs='python view_logs.py --type main --lines 50'
alias logs-follow='python view_logs.py --type main --follow'
alias logs-errors='python view_logs.py --type errors --lines 20'
alias logs-training='python view_logs.py --type structured --event training_progress --hours 2'
alias logs-resources='python view_logs.py --type structured --event resource_usage --hours 1'
alias logs-tasks='python view_logs.py --type structured --event task_start --hours 24'
alias logs-stats='python view_logs.py --type stats'

# Функции для быстрого доступа
logs_last() {
    python view_logs.py --type main --lines ${1:-100}
}

logs_training() {
    python view_logs.py --type structured --event training_progress --hours ${1:-2}
}

logs_errors() {
    python view_logs.py --type errors --lines ${1:-50}
}

logs_follow() {
    python view_logs.py --type main --follow
}

# Показать справку
logs_help() {
    echo "📄 Команды для просмотра логов microWakeWord:"
    echo ""
    echo "Основные команды:"
    echo "  logs              - Последние 50 строк основных логов"
    echo "  logs-follow       - Следить за логами в реальном времени"
    echo "  logs-errors       - Последние 20 строк ошибок"
    echo "  logs-training     - Метрики обучения за последние 2 часа"
    echo "  logs-resources    - Использование ресурсов за последний час"
    echo "  logs-tasks        - Запуски задач за последние 24 часа"
    echo "  logs-stats        - Статистика файлов логов"
    echo ""
    echo "Функции с параметрами:"
    echo "  logs_last [N]     - Последние N строк (по умолчанию 100)"
    echo "  logs_training [H] - Метрики обучения за H часов (по умолчанию 2)"
    echo "  logs_errors [N]   - Последние N строк ошибок (по умолчанию 50)"
    echo "  logs_follow       - Следить за логами в реальном времени"
    echo ""
    echo "Примеры:"
    echo "  logs_last 200     - Последние 200 строк"
    echo "  logs_training 6   - Метрики обучения за 6 часов"
    echo "  logs_errors 100   - Последние 100 строк ошибок"
}

# Экспортируем функции
export -f logs_last logs_training logs_errors logs_follow logs_help

echo "✅ Команды для просмотра логов загружены!"
echo "💡 Используйте 'logs_help' для справки"