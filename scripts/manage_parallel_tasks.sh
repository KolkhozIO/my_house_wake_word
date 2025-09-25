#!/bin/bash
# Управление параллельным пайплайном microWakeWord

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
    "start-parallel")
        print_header "🚀 ЗАПУСК ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        # Проверяем что мы в правильной директории
        if [ ! -f "parallel_pipeline_manager.py" ]; then
            print_error "parallel_pipeline_manager.py не найден!"
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
        
        print_status "Запуск параллельного пайплайна..."
        print_info "Все этапы будут работать параллельно и непрерывно"
        print_info "Логи: /home/microWakeWord_data/parallel_pipeline.log"
        print_info "Мониторинг: ./manage_parallel_tasks.sh monitor"
        
        # Запускаем параллельный пайплайн
        python parallel_pipeline_manager.py
        ;;
        
    "start-parallel-bg")
        print_header "🚀 ЗАПУСК ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА В ФОНЕ"
        
        # Проверяем что пайплайн не запущен уже
        if pgrep -f "parallel_pipeline_manager.py" > /dev/null; then
            print_warning "Параллельный пайплайн уже запущен!"
            print_info "PID: $(pgrep -f parallel_pipeline_manager.py)"
            exit 1
        fi
        
        print_info "Запуск в фоновом режиме..."
        nohup python parallel_pipeline_manager.py > /home/microWakeWord_data/parallel_pipeline_bg.log 2>&1 &
        PIPELINE_PID=$!
        
        print_status "Параллельный пайплайн запущен в фоне (PID: $PIPELINE_PID)"
        print_info "Логи: /home/microWakeWord_data/parallel_pipeline_bg.log"
        print_info "Остановка: ./manage_parallel_tasks.sh stop-parallel"
        ;;
        
    "stop-parallel")
        print_header "🛑 ОСТАНОВКА ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        # Ищем процессы параллельного пайплайна
        PIDS=$(pgrep -f "parallel_pipeline_manager.py")
        
        if [ -z "$PIDS" ]; then
            print_warning "Параллельный пайплайн не запущен"
            exit 0
        fi
        
        print_info "Найдены процессы: $PIDS"
        
        # Останавливаем процессы
        for PID in $PIDS; do
            print_info "Остановка процесса $PID..."
            kill -TERM $PID
            
            # Ждем graceful shutdown
            for i in {1..15}; do
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
        
        print_status "Параллельный пайплайн остановлен"
        ;;
        
    "status")
        print_header "📊 СТАТУС ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        # Проверяем параллельный пайплайн
        PARALLEL_PIDS=$(pgrep -f "parallel_pipeline_manager.py")
        if [ -n "$PARALLEL_PIDS" ]; then
            print_status "Параллельный пайплайн запущен (PID: $PARALLEL_PIDS)"
        else
            print_warning "Параллельный пайплайн не запущен"
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
        if [ -f "/home/microWakeWord_data/parallel_pipeline.log" ]; then
            LOG_SIZE=$(du -h /home/microWakeWord_data/parallel_pipeline.log | cut -f1)
            print_info "Размер лога: $LOG_SIZE"
        fi
        ;;
        
    "logs")
        print_header "📋 ЛОГИ ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        LOG_FILE="/home/microWakeWord_data/parallel_pipeline.log"
        BG_LOG_FILE="/home/microWakeWord_data/parallel_pipeline_bg.log"
        
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
        print_header "📈 МОНИТОРИНГ ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        print_info "Мониторинг параллельного пайплайна (Ctrl+C для выхода)..."
        echo ""
        
        while true; do
            clear
            print_header "📊 МОНИТОРИНГ ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА - $(date '+%H:%M:%S')"
            
            # Статус пайплайна
            PARALLEL_PIDS=$(pgrep -f "parallel_pipeline_manager.py")
            if [ -n "$PARALLEL_PIDS" ]; then
                print_status "Параллельный пайплайн: ЗАПУЩЕН (PID: $PARALLEL_PIDS)"
            else
                print_warning "Параллельный пайплайн: ОСТАНОВЛЕН"
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
            if [ -f "/home/microWakeWord_data/parallel_pipeline.log" ]; then
                tail -5 "/home/microWakeWord_data/parallel_pipeline.log" | sed 's/^/  /'
            else
                echo "  Логи не найдены"
            fi
            
            sleep 5
        done
        ;;
        
    "workers")
        print_header "👥 СТАТУС ВОРКЕРОВ"
        
        # Проверяем что пайплайн запущен
        if ! pgrep -f "parallel_pipeline_manager.py" > /dev/null; then
            print_warning "Параллельный пайплайн не запущен"
            exit 1
        fi
        
        print_info "Статус воркеров (из логов):"
        echo ""
        
        if [ -f "/home/microWakeWord_data/parallel_pipeline.log" ]; then
            # Ищем строки со статистикой воркеров
            grep "Статистика воркеров:" /home/microWakeWord_data/parallel_pipeline.log | tail -1
            echo ""
            grep "🔄.*- Цикл" /home/microWakeWord_data/parallel_pipeline.log | tail -10
        else
            print_warning "Логи не найдены"
        fi
        ;;
        
    "config")
        print_header "⚙️ КОНФИГУРАЦИЯ ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
        
        print_info "Конфигурация этапов (все работают параллельно):"
        echo ""
        echo "1. generate_data:"
        echo "   - Интервал циклов: 5 минут"
        echo "   - Ресурсы: 8 CPU, 2GB RAM"
        echo "   - Таймаут: 10 минут"
        echo "   - Зависимости: нет"
        echo ""
        echo "2. generate_spectrograms:"
        echo "   - Интервал циклов: 10 минут"
        echo "   - Ресурсы: 16 CPU, 4GB RAM"
        echo "   - Таймаут: 30 минут"
        echo "   - Зависимости: свежие данные"
        echo ""
        echo "3. augmentations:"
        echo "   - Интервал циклов: 15 минут"
        echo "   - Ресурсы: 4 CPU, 1GB RAM"
        echo "   - Таймаут: 20 минут"
        echo "   - Зависимости: готовые спектрограммы"
        echo ""
        echo "4. balance_dataset:"
        echo "   - Интервал циклов: 20 минут"
        echo "   - Ресурсы: 4 CPU, 1GB RAM"
        echo "   - Таймаут: 15 минут"
        echo "   - Зависимости: готовые спектрограммы"
        echo ""
        echo "5. train_model:"
        echo "   - Интервал циклов: 30 минут"
        echo "   - Ресурсы: 8 CPU, 3GB RAM"
        echo "   - Таймаут: 60 минут"
        echo "   - Зависимости: готовые спектрограммы"
        echo ""
        print_info "Для изменения настроек отредактируйте parallel_pipeline_manager.py"
        ;;
        
    "help"|"--help"|"")
        print_header "🆘 СПРАВКА ПО ПАРАЛЛЕЛЬНОМУ ПАЙПЛАЙНУ"
        
        echo "Использование: ./manage_parallel_tasks.sh <команда>"
        echo ""
        echo "Команды:"
        echo "  start-parallel      Запуск параллельного пайплайна (интерактивный)"
        echo "  start-parallel-bg   Запуск параллельного пайплайна в фоне"
        echo "  stop-parallel       Остановка параллельного пайплайна"
        echo "  status              Статус системы и задач"
        echo "  logs                Просмотр логов пайплайна"
        echo "  monitor             Мониторинг в реальном времени"
        echo "  workers             Статус воркеров"
        echo "  config              Показать конфигурацию"
        echo "  help                Показать эту справку"
        echo ""
        echo "Примеры:"
        echo "  ./manage_parallel_tasks.sh start-parallel-bg  # Запуск в фоне"
        echo "  ./manage_parallel_tasks.sh monitor             # Мониторинг"
        echo "  ./manage_parallel_tasks.sh workers            # Статус воркеров"
        echo "  ./manage_parallel_tasks.sh stop-parallel      # Остановка"
        echo ""
        echo "Особенности параллельного пайплайна:"
        echo "  ✅ Все этапы работают параллельно и непрерывно"
        echo "  ✅ Каждый этап работает в своем потоке"
        echo "  ✅ Независимые циклы выполнения"
        echo "  ✅ Автоматическое восстановление после ошибок"
        echo "  ✅ Мониторинг системных ресурсов"
        echo "  ✅ Graceful shutdown по Ctrl+C"
        echo ""
        echo "Этапы работают по кругу:"
        echo "  🔄 generate_data (каждые 5 мин)"
        echo "  🔄 generate_spectrograms (каждые 10 мин)"
        echo "  🔄 augmentations (каждые 15 мин)"
        echo "  🔄 balance_dataset (каждые 20 мин)"
        echo "  🔄 train_model (каждые 30 мин)"
        ;;
        
    *)
        print_error "Неизвестная команда: $1"
        print_info "Используйте: ./manage_parallel_tasks.sh help"
        exit 1
        ;;
esac