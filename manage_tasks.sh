#!/bin/bash

# Скрипт управления задачами microWakeWord
# НЕ РЕДАКТИРОВАТЬ - ТОЛЬКО ЧИТАТЬ!

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция показа помощи
show_help() {
    echo -e "${BLUE}microWakeWord Task Manager${NC}"
    echo "=========================="
    echo ""
    echo "Использование: $0 <команда> [задача]"
    echo ""
    echo "Команды:"
    echo "  start <task>     - Запустить задачу в фоновом режиме"
    echo "  stop <task>      - Остановить задачу"
    echo "  status           - Показать статус всех задач"
    echo "  logs <task>      - Показать логи задачи"
    echo "  list             - Показать список доступных задач"
    echo "  stop-all         - Остановить все задачи"
    echo "  cleanup          - Очистить временные файлы"
    echo ""
    echo "Доступные задачи:"
    echo "  Используйте '$0 list' для просмотра всех доступных задач"
    echo ""
    echo "Примеры:"
    echo "  $0 start train_model"
    echo "  $0 status"
    echo "  $0 logs train_model"
    echo "  $0 stop train_model"
}

# Функция запуска задачи
start_task() {
    local task="$1"
    
    if [ -z "$task" ]; then
        echo -e "${RED}❌ Ошибка: Не указана задача${NC}"
        show_help
        exit 1
    fi
    
    echo -e "${BLUE}🚀 Запуск задачи: $task${NC}"
    
    # Проверка существования задачи через task_manager.py
    if [ -f "src/management/task_manager.py" ]; then
        validation_result=$(source .venv/bin/activate && python3 src/management/task_manager.py validate "$task" 2>&1)
        if echo "$validation_result" | grep -q "✅ Задача"; then
            echo -e "${GREEN}✅ Задача $task найдена${NC}"
        else
            echo -e "${RED}❌ Ошибка: Неизвестная задача: $task${NC}"
            echo -e "${YELLOW}💡 Используйте '$0 list' для просмотра доступных задач${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
    
    # Запуск через Python менеджер
    if [ -f "src/management/task_manager.py" ]; then
        echo -e "${BLUE}📋 Запуск через task_manager.py${NC}"
        source .venv/bin/activate && python3 src/management/task_manager.py start "$task"
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция остановки задачи
stop_task() {
    local task="$1"
    
    if [ -z "$task" ]; then
        echo -e "${RED}❌ Ошибка: Не указана задача${NC}"
        show_help
        exit 1
    fi
    
    echo -e "${YELLOW}🛑 Остановка задачи: $task${NC}"
    
    if [ -f "src/management/task_manager.py" ]; then
        source .venv/bin/activate && python3 src/management/task_manager.py stop "$task"
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция показа статуса
show_status() {
    echo -e "${BLUE}📊 Статус задач microWakeWord${NC}"
    echo "=============================="
    
    if [ -f "src/management/task_manager.py" ]; then
        source .venv/bin/activate && python3 src/management/task_manager.py status
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция показа логов
show_logs() {
    local task="$1"
    
    if [ -z "$task" ]; then
        echo -e "${RED}❌ Ошибка: Не указана задача${NC}"
        show_help
        exit 1
    fi
    
    echo -e "${BLUE}📋 Логи задачи: $task${NC}"
    echo "========================"
    
    if [ -f "src/management/task_manager.py" ]; then
        source .venv/bin/activate && python3 src/management/task_manager.py logs "$task"
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция показа списка задач
show_list() {
    if [ -f "src/management/task_manager.py" ]; then
        source .venv/bin/activate && python3 src/management/task_manager.py list
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция остановки всех задач
stop_all_tasks() {
    echo -e "${YELLOW}🛑 Остановка всех задач${NC}"
    
    if [ -f "src/management/task_manager.py" ]; then
        source .venv/bin/activate && python3 src/management/task_manager.py stop-all
    else
        echo -e "${RED}❌ Ошибка: task_manager.py не найден${NC}"
        exit 1
    fi
}

# Функция очистки
cleanup() {
    echo -e "${YELLOW}🧹 Очистка временных файлов${NC}"
    
    # Очистка временных директорий
    if [ -d "temp" ]; then
        echo -e "${BLUE}📁 Очистка temp/...${NC}"
        rm -rf temp/*
    fi
    
    if [ -d "positives_both_aug_temp" ]; then
        echo -e "${BLUE}📁 Очистка positives_both_aug_temp/...${NC}"
        rm -rf positives_both_aug_temp/*
    fi
    
    echo -e "${GREEN}✅ Очистка завершена${NC}"
}

# Основная логика
case "$1" in
    "start")
        start_task "$2"
        ;;
    "stop")
        stop_task "$2"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "list")
        show_list
        ;;
    "stop-all")
        stop_all_tasks
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Ошибка: Неизвестная команда: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac