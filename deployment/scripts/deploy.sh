#!/bin/bash
# Скрипт развертывания microWakeWord

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Проверка зависимостей
check_dependencies() {
    log_info "Проверка зависимостей..."
    
    # Проверка Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker не установлен"
        exit 1
    fi
    
    # Проверка Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose не установлен"
        exit 1
    fi
    
    # Проверка Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 не установлен"
        exit 1
    fi
    
    log_success "Все зависимости установлены"
}

# Создание директорий
create_directories() {
    log_info "Создание директорий..."
    
    # Основные директории
    mkdir -p /home/microWakeWord_data/{datasets,models,results,cache}
    mkdir -p /home/microWakeWord/logs/{pipeline,system,models,archived}
    mkdir -p /home/microWakeWord/temp/{processing,training,cache}
    
    # Права доступа
    chmod 755 /home/microWakeWord_data
    chmod 755 /home/microWakeWord/logs
    chmod 755 /home/microWakeWord/temp
    
    log_success "Директории созданы"
}

# Сборка Docker образов
build_images() {
    log_info "Сборка Docker образов..."
    
    # Переход в директорию проекта
    cd "$(dirname "$0")/../.."
    
    # Сборка образа пайплайна
    log_info "Сборка образа пайплайна..."
    docker build -f deployment/docker/Dockerfile.pipeline -t microwakeword/pipeline:latest .
    
    # Сборка образа обучения (если есть GPU)
    if command -v nvidia-smi &> /dev/null; then
        log_info "Сборка образа обучения с GPU поддержкой..."
        docker build -f deployment/docker/Dockerfile.training -t microwakeword/training:latest .
    else
        log_warning "GPU не обнаружен, пропуск сборки образа обучения"
    fi
    
    log_success "Docker образы собраны"
}

# Запуск сервисов
start_services() {
    log_info "Запуск сервисов..."
    
    # Переход в директорию Docker Compose
    cd "$(dirname "$0")/../docker"
    
    # Запуск основных сервисов
    docker-compose up -d pipeline monitoring dashboard logging
    
    # Запуск сервиса обучения (если есть GPU)
    if command -v nvidia-smi &> /dev/null; then
        log_info "Запуск сервиса обучения с GPU..."
        docker-compose --profile gpu up -d training
    fi
    
    log_success "Сервисы запущены"
}

# Проверка статуса сервисов
check_services() {
    log_info "Проверка статуса сервисов..."
    
    cd "$(dirname "$0")/../docker"
    
    # Статус контейнеров
    docker-compose ps
    
    # Проверка логов
    log_info "Последние логи пайплайна:"
    docker-compose logs --tail=10 pipeline
    
    log_success "Проверка завершена"
}

# Остановка сервисов
stop_services() {
    log_info "Остановка сервисов..."
    
    cd "$(dirname "$0")/../docker"
    
    docker-compose down
    
    log_success "Сервисы остановлены"
}

# Очистка
cleanup() {
    log_info "Очистка..."
    
    cd "$(dirname "$0")/../docker"
    
    # Остановка и удаление контейнеров
    docker-compose down --volumes --remove-orphans
    
    # Удаление образов
    docker rmi microwakeword/pipeline:latest 2>/dev/null || true
    docker rmi microwakeword/training:latest 2>/dev/null || true
    
    log_success "Очистка завершена"
}

# Показать справку
show_help() {
    echo "Использование: $0 [КОМАНДА]"
    echo ""
    echo "Команды:"
    echo "  deploy     - Полное развертывание (по умолчанию)"
    echo "  start      - Запуск сервисов"
    echo "  stop       - Остановка сервисов"
    echo "  restart    - Перезапуск сервисов"
    echo "  status     - Проверка статуса"
    echo "  logs       - Просмотр логов"
    echo "  cleanup    - Очистка"
    echo "  help       - Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 deploy    # Полное развертывание"
    echo "  $0 start     # Запуск сервисов"
    echo "  $0 status    # Проверка статуса"
}

# Просмотр логов
show_logs() {
    log_info "Просмотр логов..."
    
    cd "$(dirname "$0")/../docker"
    
    echo "Выберите сервис для просмотра логов:"
    echo "1) pipeline"
    echo "2) training"
    echo "3) monitoring"
    echo "4) dashboard"
    echo "5) logging"
    echo "6) all"
    
    read -p "Введите номер (1-6): " choice
    
    case $choice in
        1) docker-compose logs -f pipeline ;;
        2) docker-compose logs -f training ;;
        3) docker-compose logs -f monitoring ;;
        4) docker-compose logs -f dashboard ;;
        5) docker-compose logs -f logging ;;
        6) docker-compose logs -f ;;
        *) log_error "Неверный выбор" ;;
    esac
}

# Основная функция развертывания
deploy() {
    log_info "Начало развертывания microWakeWord..."
    
    check_dependencies
    create_directories
    build_images
    start_services
    
    # Ожидание запуска сервисов
    log_info "Ожидание запуска сервисов..."
    sleep 10
    
    check_services
    
    log_success "Развертывание завершено!"
    echo ""
    echo "Доступные сервисы:"
    echo "  - Пайплайн: http://localhost:8080"
    echo "  - Мониторинг: http://localhost:9090"
    echo "  - Дашборд: http://localhost:3000 (admin/admin)"
    echo ""
    echo "Для просмотра логов: $0 logs"
    echo "Для проверки статуса: $0 status"
}

# Главная функция
main() {
    case "${1:-deploy}" in
        deploy)
            deploy
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 5
            start_services
            ;;
        status)
            check_services
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Неизвестная команда: $1"
            show_help
            exit 1
            ;;
    esac
}

# Запуск
main "$@"