#!/bin/bash
# Управление циклическим пайплайном microWakeWord

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Функция для вывода заголовков
print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Функция для вывода статуса
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

case "$1" in
    "start-cyclic")
        print_header "🚀 ЗАПУСК ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА"
        
        # Проверяем что мы в правильной директории
        if [ ! -f "cyclic_pipeline_manager.py" ]; then
            print_error "cyclic_pipeline_manager.py не найден!"
            print_info "Запустите из директории microWakeWord"
            exit 1
        fi
        
        # Проверяем виртуальное окружение
        if [ ! -d ".venv" ]; then
            print_error "Виртуальное окружение не найдено!"
            print_info "Создайте виртуальное окружение: python -m venv .venv"
            exit 1
        fi
        
        print_info "Активация виртуального окружения..."
        source .venv/bin/activate
        
        print_info "Проверка зависимостей..."
        python -c "import psutil" 2>/dev/null || {
            print_warning "psutil не установлен, устанавливаем..."
            pip install psutil
        }
        
        print_status "Запуск циклического пайплайна..."
        print_info "Пайплайн будет работать непрерывно до остановки (Ctrl+C)"
        print_info "Логи: /home/microWakeWord_data/cyclic_pipeline.log"
        print_info "Статус задач: ./manage_cyclic_tasks.sh status"
        
        # Запускаем циклический пайплайн
        python cyclic_pipeline_manager.py
        ;;
        
    "start-cyclic-bg")
        print_header "🚀 ЗАПУСК ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА В ФОНЕ"
        
        # Проверяем что пайплайн не запущен уже
        if pgrep -f "cyclic_pipeline_manager.py" > /dev/null; then
            print_warning "Циклический пайплайн уже запущен!"
            print_info "PID: $(pgrep -f cyclic_pipeline_manager.py)"
            exit 1
        fi
        
        print_info "Запуск в фоновом режиме..."
        nohup python cyclic_pipeline_manager.py > /home/microWakeWord_data/cyclic_pipeline_bg.log 2>&1 &
        PIPELINE_PID=$!
        
        print_status "Циклический пайплайн запущен в фоне (PID: $PIPELINE_PID)"
        print_info "Логи: /home/microWakeWord_data/cyclic_pipeline_bg.log"
        print_info "Остановка: ./manage_cyclic_tasks.sh stop-cyclic"
        ;;
        
    "stop-cyclic")
        print_header "🛑 ОСТАНОВКА ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА"
        
        # Ищем процессы циклического пайплайна
        PIDS=$(pgrep -f "cyclic_pipeline_manager.py")
        
        if [ -z "$PIDS" ]; then
            print_warning "Циклический пайплайн не запущен"
            exit 0
        fi
        
        print_info "Найдены процессы: $PIDS"
        
        # Останавливаем процессы
        for PID in $PIDS; do
            print_info "Остановка процесса $PID..."
            kill -TERM $PID
            
            # Ждем graceful shutdown
            for i in {1..10}; do
                if ! kill -0 $PID 2>/dev/null; then
                    print_status "Процесс $PID остановлен"
                    break
                fi
                sleep 1
            done
            
            # Принудительная остановка если не остановился
            if kill -0 $PID 2>/dev/null; then
                print_warning "Принудительная остановка процесса $PID..."
                kill -KILL $PID
            fi
        done
        
        # Останавливаем все задачи через task_manager
        print_info "Остановка всех задач..."
        python task_manager.py stop-all > /dev/null 2>&1
        
        print_status "Циклический пайплайн остановлен"
        ;;
        
    "status")
        print_header "📊 СТАТУС СИСТЕМЫ"
        
        # Проверяем циклический пайплайн
        CYCLIC_PIDS=$(pgrep -f "cyclic_pipeline_manager.py")
        if [ -n "$CYCLIC_PIDS" ]; then
            print_status "Циклический пайплайн запущен (PID: $CYCLIC_PIDS)"
        else
            print_warning "Циклический пайплайн не запущен"
        fi
        
        # Проверяем обычные задачи
        print_info "Активные задачи:"
        python task_manager.py status 2>/dev/null || print_warning "task_manager недоступен"
        
        # Системные ресурсы
        print_info "Системные ресурсы:"
        echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
        echo "  RAM: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
        echo "  Диск: $(df -h /home/microWakeWord_data | tail -1 | awk '{print $5}')"
        
        # Проверяем логи
        if [ -f "/home/microWakeWord_data/cyclic_pipeline.log" ]; then
            LOG_SIZE=$(du -h /home/microWakeWord_data/cyclic_pipeline.log | cut -f1)
            print_info "Размер лога: $LOG_SIZE"
        fi
        ;;
        
    "logs")
        print_header "📋 ЛОГИ ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА"
        
        LOG_FILE="/home/microWakeWord_data/cyclic_pipeline.log"
        BG_LOG_FILE="/home/microWakeWord_data/cyclic_pipeline_bg.log"
        
        if [ -f "$LOG_FILE" ]; then
            print_info "Основной лог ($LOG_FILE):"
            echo ""
            tail -50 "$LOG_FILE"
        elif [ -f "$BG_LOG_FILE" ]; then
            print_info "Фоновый лог ($BG_LOG_FILE):"
            echo ""
            tail -50 "$BG_LOG_FILE"
        else
            print_warning "Логи не найдены"
        fi
        ;;
        
    "monitor")
        print_header "📈 МОНИТОРИНГ В РЕАЛЬНОМ ВРЕМЕНИ"
        
        print_info "Мониторинг циклического пайплайна (Ctrl+C для выхода)..."
        echo ""
        
        while true; do
            clear
            print_header "📊 МОНИТОРИНГ - $(date '+%H:%M:%S')"
            
            # Статус пайплайна
            CYCLIC_PIDS=$(pgrep -f "cyclic_pipeline_manager.py")
            if [ -n "$CYCLIC_PIDS" ]; then
                print_status "Циклический пайплайн: ЗАПУЩЕН (PID: $CYCLIC_PIDS)"
            else
                print_warning "Циклический пайплайн: ОСТАНОВЛЕН"
            fi
            
            # Активные задачи
            echo ""
            print_info "Активные задачи:"
            python task_manager.py status 2>/dev/null | grep -E "(🟢|🔴)" || echo "  Нет активных задач"
            
            # Системные ресурсы
            echo ""
            print_info "Системные ресурсы:"
            echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
            echo "  RAM: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
            
            # Последние логи
            echo ""
            print_info "Последние события:"
            if [ -f "/home/microWakeWord_data/cyclic_pipeline.log" ]; then
                tail -5 "/home/microWakeWord_data/cyclic_pipeline.log" | sed 's/^/  /'
            else
                echo "  Логи не найдены"
            fi
            
            sleep 5
        done
        ;;
        
    "config")
        print_header "⚙️ КОНФИГУРАЦИЯ ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА"
        
        print_info "Текущие настройки ресурсов:"
        echo ""
        echo "Этапы пайплайна и их ресурсы:"
        echo "  1. generate_data: 25% CPU, 25% RAM, таймаут 10 мин"
        echo "  2. generate_spectrograms: 50% CPU, 50% RAM, таймаут 30 мин"
        echo "  3. augmentations: 12.5% CPU, 12.5% RAM, таймаут 20 мин"
        echo "  4. balance_dataset: 12.5% CPU, 12.5% RAM, таймаут 15 мин"
        echo "  5. train_model: 25% CPU, 33% RAM, таймаут 60 мин"
        echo ""
        echo "Настройки цикла:"
        echo "  - Задержка между циклами: 30 секунд"
        echo "  - Максимум параллельных этапов: 2"
        echo "  - Порог ошибок для пропуска этапа: 3"
        echo ""
        print_info "Для изменения настроек отредактируйте cyclic_pipeline_manager.py"
        ;;
        
    "help"|"--help"|"")
        print_header "🆘 СПРАВКА ПО ЦИКЛИЧЕСКОМУ ПАЙПЛАЙНУ"
        
        echo "Использование: ./manage_cyclic_tasks.sh <команда>"
        echo ""
        echo "Команды:"
        echo "  start-cyclic      Запуск циклического пайплайна (интерактивный)"
        echo "  start-cyclic-bg   Запуск циклического пайплайна в фоне"
        echo "  stop-cyclic       Остановка циклического пайплайна"
        echo "  status            Статус системы и задач"
        echo "  logs              Просмотр логов пайплайна"
        echo "  monitor           Мониторинг в реальном времени"
        echo "  config            Показать конфигурацию"
        echo "  help              Показать эту справку"
        echo ""
        echo "Примеры:"
        echo "  ./manage_cyclic_tasks.sh start-cyclic-bg  # Запуск в фоне"
        echo "  ./manage_cyclic_tasks.sh monitor          # Мониторинг"
        echo "  ./manage_cyclic_tasks.sh stop-cyclic      # Остановка"
        echo ""
        echo "Особенности циклического пайплайна:"
        echo "  ✅ Непрерывное выполнение этапов"
        echo "  ✅ Фиксированные ресурсы для каждого этапа"
        echo "  ✅ Автоматическое восстановление после ошибок"
        echo "  ✅ Мониторинг системных ресурсов"
        echo "  ✅ Graceful shutdown по Ctrl+C"
        ;;
        
    *)
        print_error "Неизвестная команда: $1"
        print_info "Используйте: ./manage_cyclic_tasks.sh help"
        exit 1
        ;;
esac