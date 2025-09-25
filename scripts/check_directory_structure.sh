#!/bin/bash

# Скрипт проверки фиксированной структуры директорий microWakeWord
# НЕ РЕДАКТИРОВАТЬ - ТОЛЬКО ЧИТАТЬ!

echo "🔍 ПРОВЕРКА ФИКСИРОВАННОЙ СТРУКТУРЫ ДИРЕКТОРИЙ"
echo "=============================================="

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция проверки директории
check_directory() {
    local dir="$1"
    local required="$2"
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ $dir${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ ОТСУТСТВУЕТ ОБЯЗАТЕЛЬНАЯ ДИРЕКТОРИЯ: $dir${NC}"
            return 1
        else
            echo -e "${YELLOW}⚠️  ОТСУТСТВУЕТ ОПЦИОНАЛЬНАЯ ДИРЕКТОРИЯ: $dir${NC}"
            return 0
        fi
    fi
}

# Функция проверки файла
check_file() {
    local file="$1"
    local required="$2"
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ ОТСУТСТВУЕТ ОБЯЗАТЕЛЬНЫЙ ФАЙЛ: $file${NC}"
            return 1
        else
            echo -e "${YELLOW}⚠️  ОТСУТСТВУЕТ ОПЦИОНАЛЬНЫЙ ФАЙЛ: $file${NC}"
            return 0
        fi
    fi
}

echo ""
echo "📁 ОСНОВНЫЕ ДИРЕКТОРИИ:"
echo "======================"

# Обязательные директории
check_directory "src" "true"
check_directory "src/pipeline" "true"
check_directory "src/pipeline/training" "true"
check_directory "src/pipeline/data_generation" "true"
check_directory "src/pipeline/augmentation" "true"
check_directory "src/pipeline/balancing" "true"
check_directory "src/management" "true"
check_directory "src/monitoring" "true"
check_directory "src/utils" "true"
check_directory "src/api" "true"
check_directory "src/cli" "true"
check_directory "src/web" "true"

check_directory "config" "true"
check_directory "docs" "true"
check_directory "scripts" "true"
check_directory "tests" "true"
check_directory "tools" "true"
check_directory "deployment" "true"

echo ""
echo "📄 ОСНОВНЫЕ ФАЙЛЫ:"
echo "=================="

# Обязательные файлы
check_file "manage.sh" "true"
check_file "manage_tasks.sh" "true"
check_file "requirements.txt" "true"
check_file "requirements-dev.txt" "true"

echo ""
echo "📊 ДАННЫЕ И МОДЕЛИ:"
echo "==================="

# Директории с данными
check_directory "logs" "false"
check_directory "temp" "false"
check_directory "models" "false"
check_directory "trained_models" "false"
check_directory "generated_features" "false"
check_directory "generated_features_negatives_real" "false"
check_directory "positives_both_aug_temp" "false"

echo ""
echo "🔄 РЕЗЕРВНЫЕ КОПИИ:"
echo "==================="

check_directory "backups" "false"
check_directory "mww_orig" "false"
check_directory "piper-sample-generator" "false"

echo ""
echo "🎯 КЛЮЧЕВЫЕ СКРИПТЫ ОБУЧЕНИЯ:"
echo "=============================="

# Основные скрипты обучения
check_file "src/pipeline/training/use_original_library_correctly.py" "true"
check_file "src/pipeline/training/train_model_only.py" "true"
check_file "src/pipeline/training/use_original_library_correctly_fixed.py" "false"

echo ""
echo "📈 СКРИПТЫ ГЕНЕРАЦИИ ДАННЫХ:"
echo "============================"

# Основные скрипты генерации
check_file "src/pipeline/data_generation/generate_spectrograms.py" "true"
check_file "src/pipeline/data_generation/generate_both_phrases.py" "true"
check_file "src/pipeline/data_generation/quick_generate.py" "true"

echo ""
echo "⚙️ МЕНЕДЖЕРЫ ЗАДАЧ:"
echo "==================="

# Менеджеры задач
check_file "src/management/task_manager.py" "true"
check_file "src/management/run_pipeline.py" "true"
check_file "src/management/start_pipeline.py" "true"

echo ""
echo "🔧 УТИЛИТЫ:"
echo "============"

# Утилиты
check_file "src/utils/config_manager.py" "true"
check_file "src/utils/generate_unique_model_name.py" "true"

echo ""
echo "📋 СТАТУС ПРОВЕРКИ:"
echo "==================="

# Подсчет ошибок
errors=0
if [ $? -ne 0 ]; then
    errors=$((errors + 1))
fi

if [ $errors -eq 0 ]; then
    echo -e "${GREEN}✅ СТРУКТУРА ДИРЕКТОРИЙ СООТВЕТСТВУЕТ ФИКСИРОВАННОЙ!${NC}"
    echo -e "${GREEN}✅ ВСЕ ОБЯЗАТЕЛЬНЫЕ КОМПОНЕНТЫ НА МЕСТЕ!${NC}"
    exit 0
else
    echo -e "${RED}❌ ОБНАРУЖЕНЫ ОТСУТСТВУЮЩИЕ КОМПОНЕНТЫ!${NC}"
    echo -e "${RED}❌ СТРУКТУРА НЕ СООТВЕТСТВУЕТ ФИКСИРОВАННОЙ!${NC}"
    exit 1
fi