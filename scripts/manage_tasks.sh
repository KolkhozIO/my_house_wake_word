#!/bin/bash
# Управление задачами microWakeWord

case "$1" in
    "start")
        if [ -z "$2" ]; then
            echo "🚀 Запуск полного пайплайна (неблокирующий)..."
            source .venv/bin/activate && python start_pipeline.py
        else
            echo "🚀 Запуск задачи '$2'..."
            case "$2" in
                "generate_data")
                    source .venv/bin/activate && python task_manager.py start generate_data 'python generate_both_phrases.py > /home/microWakeWord_data/tasks/generate_data.log 2>&1'
                    ;;
                "augmentations")
                    source .venv/bin/activate && python task_manager.py start augmentations 'python apply_augmentations.py > /home/microWakeWord_data/tasks/augmentations.log 2>&1'
                    ;;
                "balance_dataset")
                    source .venv/bin/activate && python task_manager.py start balance_dataset 'python balance_dataset.py > /home/microWakeWord_data/tasks/balance_dataset.log 2>&1'
                    ;;
        "train_model")
            # Получаем описание изменений из аргументов
            shift  # Убираем "train_model" из аргументов
            description="$*"
            if [ -n "$description" ]; then
                echo "📝 Описание изменений: $description"
                source .venv/bin/activate && python task_manager.py start train_model "python use_original_library_correctly.py '$description' > /home/microWakeWord_data/tasks/train_model.log 2>&1"
            else
                echo "⚠️ Описание изменений не указано - будет использовано базовое имя"
                source .venv/bin/activate && python task_manager.py start train_model 'python use_original_library_correctly.py > /home/microWakeWord_data/tasks/train_model.log 2>&1'
            fi
            ;;
                "generate_spectrograms")
                    source .venv/bin/activate && python task_manager.py start generate_spectrograms 'python generate_spectrograms.py > /home/microWakeWord_data/tasks/generate_spectrograms.log 2>&1'
                    ;;
                "generate_spectrograms_parallel")
                    source .venv/bin/activate && python task_manager.py start generate_spectrograms_parallel 'python generate_spectrograms_parallel.py > generate_spectrograms_parallel.log 2>&1'
                    ;;
                "train_model_parallel")
                    source .venv/bin/activate && python task_manager.py start train_model_parallel 'python train_model_parallel.py > train_model_parallel.log 2>&1'
                    ;;
                "train_model_only")
                    source .venv/bin/activate && python task_manager.py start train_model_only 'python train_model_only.py > train_model_only.log 2>&1'
                    ;;
                "train_model_fixed")
                    # Получаем описание изменений из аргументов
                    shift  # Убираем "train_model_fixed" из аргументов
                    description="$*"
                    if [ -n "$description" ]; then
                        echo "📝 Описание изменений: $description"
                        source .venv/bin/activate && python task_manager.py start train_model_fixed "python use_original_library_correctly_fixed.py '$description' > train_model_fixed.log 2>&1"
                    else
                        echo "⚠️ Описание изменений не указано - будет использовано базовое имя"
                        source .venv/bin/activate && python task_manager.py start train_model_fixed 'python use_original_library_correctly_fixed.py > train_model_fixed.log 2>&1'
                    fi
                    ;;
                "generate_ambient")
                    source .venv/bin/activate && python task_manager.py start generate_ambient 'python generate_ambient_only.py > generate_ambient.log 2>&1'
                    ;;
                "generate_ambient_parallel")
                    source .venv/bin/activate && python task_manager.py start generate_ambient_parallel 'python generate_ambient_only.py > generate_ambient_parallel.log 2>&1'
                    ;;
                "fix_sample_rate")
                    source .venv/bin/activate && python task_manager.py start fix_sample_rate 'python fix_sample_rate.py > fix_sample_rate.log 2>&1'
                    ;;
                "fix_sample_rate_python")
                    source .venv/bin/activate && python task_manager.py start fix_sample_rate_python 'python fix_sample_rate_python.py > fix_sample_rate_python.log 2>&1'
                    ;;
                "fix_sample_rate_to_16000")
                    source .venv/bin/activate && python task_manager.py start fix_sample_rate_to_16000 'python fix_sample_rate_to_16000.py > fix_sample_rate_to_16000.log 2>&1'
                    ;;
                "train_larger")
                    source .venv/bin/activate && python task_manager.py start train_larger 'python train_larger_model.py > train_larger.log 2>&1'
                    ;;
                "generate_hard_negatives")
                    source .venv/bin/activate && python task_manager.py start generate_hard_negatives 'python generate_hard_negatives.py > hard_negatives.log 2>&1'
                    ;;
                "generate_enhanced_positives")
                    source .venv/bin/activate && python task_manager.py start generate_enhanced_positives 'python generate_enhanced_positives.py > enhanced_positives.log 2>&1'
                    ;;
                "generate_background")
                    source .venv/bin/activate && python task_manager.py start generate_background 'python generate_background_data.py > background_data.log 2>&1'
                    ;;
                "generate_hard_negatives_parallel")
                    source .venv/bin/activate && python task_manager.py start generate_hard_negatives_parallel 'python generate_hard_negatives_parallel.py > hard_negatives_parallel.log 2>&1'
                    ;;
                "normalize_audio_formats")
                    source .venv/bin/activate && python task_manager.py start normalize_audio_formats 'python normalize_audio_formats.py > normalize_audio_formats.log 2>&1'
                    ;;
                "generate_spectrograms_expanded")
                    source .venv/bin/activate && python task_manager.py start generate_spectrograms_expanded 'python generate_spectrograms_expanded.py > generate_spectrograms_expanded.log 2>&1'
                    ;;
                "train_model_expanded")
                    source .venv/bin/activate && python task_manager.py start train_model_expanded 'python train_model_expanded_logged.py > train_model_expanded.log 2>&1'
                    ;;
                "train_model_direct")
                    source .venv/bin/activate && python task_manager.py start train_model_direct 'rm -rf /home/microWakeWord_data/trained_models/wakeword/* && /home/microWakeWord/.venv/bin/python -m microwakeword.model_train_eval --training_config /home/microWakeWord_data/training_parameters.yaml --train 1 --restore_checkpoint 0 --test_tf_nonstreaming 0 --test_tflite_nonstreaming 0 --test_tflite_nonstreaming_quantized 0 --test_tflite_streaming 0 --test_tflite_streaming_quantized 0 inception > train_model_direct.log 2>&1'
                    ;;
                "generate_spectrograms_expanded_parallel")
                    source .venv/bin/activate && python task_manager.py start generate_spectrograms_expanded_parallel 'python generate_spectrograms_expanded_parallel.py > generate_spectrograms_expanded_parallel.log 2>&1'
                    ;;
                "train_model_fast")
                    source .venv/bin/activate && python task_manager.py start train_model_fast 'python train_model_fast.py > train_model_fast.log 2>&1'
                    ;;
                "generate_spectrograms_with_sampling")
                    source .venv/bin/activate && python task_manager.py start generate_spectrograms_with_sampling 'python generate_spectrograms_with_sampling.py > generate_spectrograms_with_sampling.log 2>&1'
                    ;;
                "generate_negatives_real")
                    source .venv/bin/activate && python task_manager.py start generate_negatives_real 'python generate_spectrograms_negatives_real.py > generate_negatives_real.log 2>&1'
                    ;;
        "generate_background_data")
            source .venv/bin/activate && python task_manager.py start generate_background_data 'python generate_spectrograms_background_data.py > generate_background_data.log 2>&1'
            ;;
        "generate_features")
            source .venv/bin/activate && python task_manager.py start generate_features 'python generate_features_from_samples.py > generate_features.log 2>&1'
            ;;
        "generate_model_name")
            if [ -z "$3" ]; then
                echo "❌ Использование: ./manage_tasks.sh start generate_model_name 'комментарий об изменениях'"
                echo "Пример: ./manage_tasks.sh start generate_model_name 'max_coverage_sampling_840k_samples'"
                exit 1
            fi
            source .venv/bin/activate && python generate_unique_model_name.py "$3" > generate_model_name.log 2>&1
            ;;
                *)
                    echo "❌ Неизвестная задача: $2"
                    echo "Доступные задачи: generate_data, augmentations, balance_dataset, train_model [описание_изменений], generate_spectrograms, generate_spectrograms_parallel, train_model_parallel, train_model_only, train_larger, generate_hard_negatives, generate_enhanced_positives, generate_background, generate_hard_negatives_parallel, normalize_audio_formats, generate_spectrograms_expanded, train_model_expanded, generate_spectrograms_expanded_parallel, generate_spectrograms_with_sampling, generate_negatives_real, generate_background_data, fix_sample_rate_to_16000"
    echo ""
    echo "Примеры использования train_model с описанием изменений:"
    echo "  ./manage_tasks.sh start train_model \"использование новых семплов\""
    echo "  ./manage_tasks.sh start train_model \"оптимизация batch_size\""
    echo "  ./manage_tasks.sh start train_model \"новые аугментации\""
                    exit 1
                    ;;
            esac
        fi
        ;;
    "status")
        echo "📋 Статус задач:"
        source .venv/bin/activate && python task_manager.py status
        ;;
    "logs")
        if [ -z "$2" ]; then
            echo "❌ Использование: $0 logs <имя_задачи>"
            echo "Доступные задачи: generate_data, augmentations, balance_dataset, train_model, train_larger, generate_hard_negatives, generate_enhanced_positives, generate_background, generate_spectrograms, generate_spectrograms_parallel, train_model_parallel, train_model_only, generate_ambient, generate_ambient_parallel, fix_sample_rate_to_16000"
            exit 1
        fi
        echo "📄 Логи задачи '$2':"
        source .venv/bin/activate && python task_manager.py logs "$2"
        ;;
    "stop")
        if [ -z "$2" ]; then
            echo "❌ Использование: $0 stop <имя_задачи>"
            echo "Доступные задачи: generate_data, augmentations, balance_dataset, train_model, train_larger, generate_hard_negatives, generate_enhanced_positives, generate_background, generate_spectrograms, generate_spectrograms_parallel, train_model_parallel, train_model_only, generate_ambient, generate_ambient_parallel"
            exit 1
        fi
        echo "🛑 Остановка задачи '$2'..."
        source .venv/bin/activate && python task_manager.py stop "$2"
        ;;
    "stop-all")
        echo "🛑 Остановка всех задач..."
        source .venv/bin/activate && python task_manager.py stop generate_data
        source .venv/bin/activate && python task_manager.py stop augmentations
        source .venv/bin/activate && python task_manager.py stop balance_dataset
        source .venv/bin/activate && python task_manager.py stop train_model
        source .venv/bin/activate && python task_manager.py stop train_larger
        source .venv/bin/activate && python task_manager.py stop generate_hard_negatives
        source .venv/bin/activate && python task_manager.py stop generate_enhanced_positives
        source .venv/bin/activate && python task_manager.py stop generate_background
        ;;
    "cleanup")
        echo "🧹 Очистка завершенных задач..."
        source .venv/bin/activate && python task_manager.py cleanup
        ;;
    "help")
        echo "📖 Команды управления задачами:"
        echo "  $0 start          - Запуск полного пайплайна"
        echo "  $0 status         - Статус всех задач"
        echo "  $0 logs <задача>  - Логи конкретной задачи"
        echo "  $0 stop <задача>  - Остановка конкретной задачи"
        echo "  $0 stop-all       - Остановка всех задач"
        echo "  $0 cleanup        - Очистка завершенных задач"
        echo "  $0 help           - Эта справка"
        echo ""
        echo "Доступные задачи:"
        echo "  - generate_data   - Генерация TTS данных"
        echo "  - augmentations   - Применение аугментаций"
        echo "  - balance_dataset - Балансировка датасета"
        echo "  - train_model     - Обучение модели"
        echo "  - train_larger    - Обучение большей модели"
        echo "  - generate_hard_negatives - Генерация hard negative примеров"
        echo "  - generate_enhanced_positives - Генерация расширенных позитивных данных"
        echo "  - generate_background - Генерация фоновых данных"
        echo "  - generate_spectrograms - Генерация спектрограмм (позитивные + негативные)"
        echo "  - train_model_only - Обучение модели (только обучение)"
        echo "  - generate_ambient - Генерация ambient данных (отдельно)"
        echo "  - generate_ambient_parallel - Генерация ambient данных (параллельно на всех ядрах)"
        echo "  - generate_negatives_real - Генерация семплов для negatives_real (с проверкой существующих)"
        echo "  - generate_background_data - Генерация семплов для background_data (с проверкой существующих)"
        ;;
    *)
        echo "❌ Неизвестная команда: $1"
        echo "Используйте '$0 help' для справки"
        exit 1
        ;;
esac