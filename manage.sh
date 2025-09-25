#!/bin/bash
# Единый скрипт управления microWakeWord
# СОБЛЮДЕНИЕ ВСЕХ ПРАВИЛ ПРОЕКТА
# 🚨 ВСЕГДА ТОЛЬКО ЧЕРЕЗ СИСТЕМУ ТАСКОВ!

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

# Проверка виртуального окружения
check_venv() {
    if [ ! -d ".venv" ]; then
        print_error "Виртуальное окружение не найдено!"
        print_info "Создайте виртуальное окружение: python -m venv .venv"
        exit 1
    fi
}

# Активация виртуального окружения
activate_venv() {
    source .venv/bin/activate
}

# 🚨 КРИТИЧЕСКОЕ ПРАВИЛО: ВСЕГДА ТОЛЬКО ЧЕРЕЗ СИСТЕМУ ТАСКОВ!
# Функция для запуска задач через task_manager.py
start_task() {
    local task_name="$1"
    local command="$2"
    local description="$3"
    
    print_header "$description"
    check_venv
    activate_venv
    
    # 🚨 ВСЕГДА ТОЛЬКО ЧЕРЕЗ СИСТЕМУ ТАСКОВ!
    python src/management/task_manager.py start "$task_name" "$command"
    
    if [ $? -eq 0 ]; then
        print_status "Задача '$task_name' запущена в фоне"
        print_info "Для мониторинга: ./manage.sh pipeline status"
        print_info "Для логов: ./manage.sh pipeline logs $task_name"
    else
        print_error "Ошибка запуска задачи '$task_name'"
        exit 1
    fi
}

# Основная функция
main() {
    case "$1" in
        "pipeline")
            case "$2" in
                "start")
                    check_venv
                    activate_venv
                    
                    case "$3" in
                        "--parallel")
                            print_header "🚀 ЗАПУСК ПАРАЛЛЕЛЬНОГО ПАЙПЛАЙНА"
                            python src/management/parallel_pipeline_manager.py
                            ;;
                        "--cyclic")
                            print_header "🚀 ЗАПУСК ЦИКЛИЧЕСКОГО ПАЙПЛАЙНА"
                            python src/management/cyclic_pipeline_manager.py
                            ;;
                        "--sequential"|"")
                            print_header "🚀 ЗАПУСК ПОСЛЕДОВАТЕЛЬНОГО ПАЙПЛАЙНА"
                            python src/management/start_pipeline.py
                            ;;
                        *)
                            print_error "Неизвестный режим: $3"
                            print_info "Доступные режимы: --parallel, --cyclic, --sequential"
                            exit 1
                            ;;
                    esac
                    ;;
                "status")
                    print_header "📊 СТАТУС ПАЙПЛАЙНА"
                    activate_venv
                    python src/management/task_manager.py status
                    ;;
                "logs")
                    print_header "📋 ЛОГИ ПАЙПЛАЙНА"
                    activate_venv
                    if [ -n "$3" ]; then
                        python src/management/task_manager.py logs "$3"
                    else
                        print_info "Использование: ./manage.sh pipeline logs <task_name>"
                    fi
                    ;;
                "stop")
                    print_header "🛑 ОСТАНОВКА ПАЙПЛАЙНА"
                    activate_venv
                    if [ -n "$3" ]; then
                        python src/management/task_manager.py stop "$3"
                    else
                        python src/management/task_manager.py stop-all
                    fi
                    ;;
                "cleanup")
                    print_header "🧹 ОЧИСТКА ПАЙПЛАЙНА"
                    activate_venv
                    python src/management/task_manager.py cleanup
                    ;;
                *)
                    print_error "Неизвестная команда пайплайна: $2"
                    print_info "Доступные команды: start, status, logs, stop, cleanup"
                    exit 1
                    ;;
            esac
            ;;
        "data")
            case "$2" in
                "generate")
                    case "$3" in
                        "--quick")
                            start_task "generate_data_quick" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/quick_generate.py' \
                                "⚡ БЫСТРАЯ ГЕНЕРАЦИЯ ДАННЫХ"
                            ;;
                        "--tts"|"")
                            start_task "generate_data_tts" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/generate_both_phrases.py' \
                                "🎤 ГЕНЕРАЦИЯ TTS ДАННЫХ"
                            ;;
                        *)
                            print_error "Неизвестный режим генерации: $3"
                            print_info "Доступные режимы: --quick, --tts"
                            exit 1
                            ;;
                    esac
                    ;;
                "augment")
                    start_task "augmentations" \
                        'source .venv/bin/activate && python src/pipeline/augmentation/apply_augmentations.py' \
                        "🎨 ПРИМЕНЕНИЕ АУГМЕНТАЦИЙ"
                    ;;
                "balance")
                    start_task "balance_dataset" \
                        'source .venv/bin/activate && python src/pipeline/balancing/balance_dataset.py' \
                        "⚖️ БАЛАНСИРОВКА ДАТАСЕТА"
                    ;;
                "mix")
                    case "$3" in
                        "--conservative")
                            start_task "mix_conservative" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=conservative' \
                                "🔊 КОНСЕРВАТИВНОЕ СМЕШИВАНИЕ TTS С ШУМОМ"
                            ;;
                        "--moderate")
                            start_task "mix_moderate" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=moderate' \
                                "🔊 УМЕРЕННОЕ СМЕШИВАНИЕ TTS С ШУМОМ"
                            ;;
                        "--aggressive")
                            start_task "mix_aggressive" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=aggressive' \
                                "🔊 АГРЕССИВНОЕ СМЕШИВАНИЕ TTS С ШУМОМ"
                            ;;
                        "--extreme")
                            start_task "mix_extreme" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=extreme' \
                                "🔊 ЭКСТРЕМАЛЬНОЕ СМЕШИВАНИЕ TTS С ШУМОМ"
                            ;;
                        "--all")
                            start_task "mix_all_variants" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=conservative && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=moderate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=aggressive && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=extreme' \
                                "🔊 СМЕШИВАНИЕ ВСЕХ ВАРИАНТОВ TTS С ШУМОМ"
                            ;;
                        "--test")
                            start_task "mix_test" \
                                'source .venv/bin/activate && python src/pipeline/data_generation/mix_tts_with_real_noise.py --variant=moderate --test' \
                                "🧪 ТЕСТОВОЕ СМЕШИВАНИЕ TTS С ШУМОМ"
                            ;;
                        *)
                            print_error "Неизвестный вариант смешивания: $3"
                            print_info "Доступные варианты: --conservative, --moderate, --aggressive, --extreme, --all, --test"
                            exit 1
                            ;;
                    esac
                    ;;
                *)
                    print_error "Неизвестная команда данных: $2"
                    print_info "Доступные команды: generate, augment, balance, mix"
                    exit 1
                    ;;
            esac
            ;;
        "model")
            case "$2" in
                "train")
                    case "$3" in
                        "--size=small")
                            start_task "train_small" \
                                'source .venv/bin/activate && python src/pipeline/training/train_model_fast.py' \
                                "🧠 ОБУЧЕНИЕ МАЛЕНЬКОЙ МОДЕЛИ"
                            ;;
                        "--size=medium"|"")
                            start_task "train_model" \
                                'source .venv/bin/activate && python src/pipeline/training/use_original_library_correctly_fixed.py' \
                                "🧠 ОБУЧЕНИЕ СРЕДНЕЙ МОДЕЛИ"
                            ;;
                        "--size=large")
                            start_task "train_large" \
                                'source .venv/bin/activate && python src/pipeline/training/train_model_expanded.py' \
                                "🧠 ОБУЧЕНИЕ БОЛЬШОЙ МОДЕЛИ"
                            ;;
                        "--size=real")
                            start_task "train_with_real_data" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_real_data_optimized.py' \
                                "🎯 ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ С РЕАЛЬНЫМИ ДАННЫМИ (ИСПРАВЛЕНИЕ ПЕРЕОБУЧЕНИЯ)"
                            ;;
                        "--size=mixed-conservative")
                            start_task "train_mixed_conservative" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=conservative' \
                                "🧠 ОБУЧЕНИЕ НА КОНСЕРВАТИВНО СМЕШАННЫХ ДАННЫХ"
                            ;;
                        "--size=mixed-moderate")
                            start_task "train_mixed_moderate" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=moderate' \
                                "🧠 ОБУЧЕНИЕ НА УМЕРЕННО СМЕШАННЫХ ДАННЫХ"
                            ;;
                        "--size=mixed-aggressive")
                            start_task "train_mixed_aggressive" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=aggressive' \
                                "🧠 ОБУЧЕНИЕ НА АГРЕССИВНО СМЕШАННЫХ ДАННЫХ"
                            ;;
                        "--size=mixed-extreme")
                            start_task "train_mixed_extreme" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=extreme' \
                                "🧠 ОБУЧЕНИЕ НА ЭКСТРЕМАЛЬНО СМЕШАННЫХ ДАННЫХ"
                            ;;
                        "--size=mixed-all")
                            start_task "train_mixed_all" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=conservative && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=moderate && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=aggressive && python src/pipeline/training/train_with_mixed_noisy_data.py --variant=extreme' \
                                "🧠 ОБУЧЕНИЕ НА ВСЕХ ВАРИАНТАХ СМЕШАННЫХ ДАННЫХ"
                            ;;
                        "--size=mixed-fixed")
                            start_task "train_mixed_fixed" \
                                'source .venv/bin/activate && python src/pipeline/training/train_with_mixed_data_fixed.py --variant=conservative' \
                                "🧠 ОБУЧЕНИЕ НА ИСПРАВЛЕННЫХ СМЕШАННЫХ ДАННЫХ"
                            ;;
                        *)
                            print_error "Неизвестный размер модели: $3"
                            print_info "Доступные размеры: --size=small, --size=medium, --size=large, --size=real, --size=mixed-conservative, --size=mixed-moderate, --size=mixed-aggressive, --size=mixed-extreme, --size=mixed-all, --size=mixed-fixed"
                            exit 1
                            ;;
                    esac
                    ;;
                "test")
                    start_task "test_model" \
                        'source .venv/bin/activate && python src/utils/generate_unique_model_name.py' \
                        "🧪 ТЕСТИРОВАНИЕ МОДЕЛИ"
                    ;;
                "deploy")
                    print_header "🚀 РАЗВЕРТЫВАНИЕ МОДЕЛИ"
                    print_info "Копирование модели в ESPHome конфигурацию..."
                    
                    # Проверяем существование файлов
                    if [ -f "/home/microWakeWord_data/models/current/original_library_model.tflite" ]; then
                        cp /home/microWakeWord_data/models/current/original_library_model.tflite config/esp32/
                        cp /home/microWakeWord_data/models/current/original_library_model.json config/esp32/
                        print_status "Модель развернута в config/esp32/"
                    else
                        print_error "Модель не найдена! Сначала обучите модель."
                        print_info "Используйте: ./manage.sh model train"
                        exit 1
                    fi
                    ;;
                *)
                    print_error "Неизвестная команда модели: $2"
                    print_info "Доступные команды: train, test, deploy"
                    exit 1
                    ;;
            esac
            ;;
        "system")
            case "$2" in
                "monitor")
                    print_header "📈 МОНИТОРИНГ СИСТЕМЫ"
                    bash src/monitoring/auto_monitor.sh
                    ;;
                "health")
                    print_header "🏥 ПРОВЕРКА ЗДОРОВЬЯ СИСТЕМЫ"
                    echo "📊 Системные ресурсы:"
                    echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
                    echo "  RAM: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
                    echo "  Диск: $(df -h /home/microWakeWord_data | tail -1 | awk '{print $5}')"
                    ;;
                "backup")
                    print_header "💾 РЕЗЕРВНОЕ КОПИРОВАНИЕ"
                    timestamp=$(date +%Y%m%d_%H%M%S)
                    tar -czf "backup_${timestamp}.tar.gz" src/ config/ scripts/ docs/
                    print_status "Резервная копия создана: backup_${timestamp}.tar.gz"
                    ;;
                *)
                    print_error "Неизвестная команда системы: $2"
                    print_info "Доступные команды: monitor, health, backup"
                    exit 1
                    ;;
            esac
            ;;
        "help"|"--help"|"")
            print_header "🆘 СПРАВКА ПО MICROWAKEWORD"
            
            echo "Использование: ./manage.sh <категория> <команда> [опции]"
            echo ""
            echo "Категории и команды:"
            echo ""
            echo "📊 ПАЙПЛАЙН:"
            echo "  ./manage.sh pipeline start [--parallel|--cyclic|--sequential]"
            echo "  ./manage.sh pipeline status"
            echo "  ./manage.sh pipeline logs [task_name]"
            echo "  ./manage.sh pipeline stop [task_name|--all]"
            echo "  ./manage.sh pipeline cleanup"
            echo ""
            echo "📁 ДАННЫЕ:"
            echo "  ./manage.sh data generate [--quick|--tts]"
            echo "  ./manage.sh data augment"
            echo "  ./manage.sh data balance"
            echo ""
            echo "🧠 МОДЕЛЬ:"
            echo "  ./manage.sh model train [--size=small|medium|large]"
            echo "  ./manage.sh model test"
            echo "  ./manage.sh model deploy"
            echo ""
            echo "⚙️ СИСТЕМА:"
            echo "  ./manage.sh system monitor"
            echo "  ./manage.sh system health"
            echo "  ./manage.sh system backup"
            echo ""
            echo "Примеры:"
            echo "  ./manage.sh pipeline start --parallel    # Параллельный пайплайн"
            echo "  ./manage.sh data generate --quick       # Быстрая генерация"
            echo "  ./manage.sh model train --size=large    # Большая модель"
            echo "  ./manage.sh system health               # Проверка системы"
            echo ""
            echo "🚨 КРИТИЧЕСКИЕ ПРАВИЛА:"
            echo "  ✅ ВСЕГДА ТОЛЬКО ЧЕРЕЗ СИСТЕМУ ТАСКОВ!"
            echo "  ✅ ВСЕГДА ИСПОЛЬЗУЙТЕ VENV!"
            echo "  ✅ Wake Word специфика (дисбаланс 100:1 - это норма!)"
            echo "  ✅ НИКОГДА НЕ БЛОКИРУЙТЕ ТЕРМИНАЛ!"
            echo "  ✅ ВСЕ ЗАДАЧИ ВЫПОЛНЯЮТСЯ В ФОНЕ!"
            echo ""
            echo "📋 УПРАВЛЕНИЕ ЗАДАЧАМИ:"
            echo "  ./manage.sh pipeline status              # Статус всех задач"
            echo "  ./manage.sh pipeline logs <task_name>  # Логи конкретной задачи"
            echo "  ./manage.sh pipeline stop <task_name>   # Остановка задачи"
            echo "  ./manage.sh pipeline cleanup             # Очистка завершенных"
            ;;
        *)
            print_error "Неизвестная категория: $1"
            print_info "Используйте: ./manage.sh help"
            exit 1
            ;;
    esac
}

# Запуск основной функции
main "$@"