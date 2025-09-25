#!/bin/bash

# Скрипт проверки контракта данных microWakeWord
# НЕ РЕДАКТИРОВАТЬ - ТОЛЬКО ЧИТАТЬ!

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🔍 ПРОВЕРКА КОНТРАКТА НА ДАННЫЕ"
echo "==============================="

# Функция проверки директории
check_data_directory() {
    local dir="$1"
    local expected_files="$2"
    local description="$3"
    
    if [ -d "$dir" ]; then
        local actual_files=$(find "$dir" -name "*.wav" 2>/dev/null | wc -l)
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        
        if [ "$actual_files" -ge "$expected_files" ]; then
            echo -e "${GREEN}✅ $description${NC}"
            echo -e "   📁 $dir"
            echo -e "   📄 $actual_files файлов (ожидалось: $expected_files+)"
            echo -e "   💾 $size"
            return 0
        else
            echo -e "${YELLOW}⚠️  $description${NC}"
            echo -e "   📁 $dir"
            echo -e "   📄 $actual_files файлов (ожидалось: $expected_files+)"
            echo -e "   💾 $size"
            return 1
        fi
    else
        echo -e "${RED}❌ ОТСУТСТВУЕТ: $description${NC}"
        echo -e "   📁 $dir"
        return 1
    fi
}

echo ""
echo "🎯 ПОЗИТИВНЫЕ ДАННЫЕ:"
echo "===================="

# Основные позитивные данные
check_data_directory "/home/microWakeWord_data/positives_final" 3000 "ОСНОВНЫЕ ПОЗИТИВНЫЕ ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/positives_both" 3000 "ПОЗИТИВНЫЕ ДАННЫЕ (ОБЕ ФРАЗЫ)"
check_data_directory "/home/microWakeWord_data/positives_enhanced" 1000 "УЛУЧШЕННЫЕ ПОЗИТИВНЫЕ ДАННЫЕ"

# Резервные копии
check_data_directory "/home/microWakeWord_data/positives_final_backup" 3000 "БЭКАП ПОЗИТИВНЫХ ДАННЫХ"
check_data_directory "/home/microWakeWord_data/positives_final_normalized" 3000 "НОРМАЛИЗОВАННЫЕ ПОЗИТИВНЫЕ ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/positives_final_temp" 3000 "ВРЕМЕННЫЕ ПОЗИТИВНЫЕ ДАННЫЕ"

echo ""
echo "🚫 НЕГАТИВНЫЕ ДАННЫЕ:"
echo "===================="

# Основные негативные данные
check_data_directory "/home/microWakeWord_data/negatives_real" 15000 "ОСНОВНЫЕ РЕАЛЬНЫЕ НЕГАТИВНЫЕ ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/negatives_real_sampled" 300000 "СЕМПЛИРОВАННЫЕ НЕГАТИВНЫЕ ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/negatives_final" 15000 "ФИНАЛЬНЫЕ НЕГАТИВНЫЕ ДАННЫЕ"

# Дополнительные негативные данные
check_data_directory "/home/microWakeWord_data/negatives_final_normalized" 15000 "НОРМАЛИЗОВАННЫЕ НЕГАТИВНЫЕ ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/negatives_both" 500 "НЕГАТИВНЫЕ ДАННЫЕ (ОБЕ ФРАЗЫ)"

echo ""
echo "🌊 BACKGROUND ДАННЫЕ:"
echo "===================="

# Основные background данные
check_data_directory "/home/microWakeWord_data/background_data" 50 "ОСНОВНЫЕ AMBIENT ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/background_data_sampled" 500000 "СЕМПЛИРОВАННЫЕ AMBIENT ДАННЫЕ"
check_data_directory "/home/microWakeWord_data/background" 5000 "ФОНОВЫЕ ДАННЫЕ"

echo ""
echo "📊 ОБЩАЯ СТАТИСТИКА:"
echo "===================="

# Подсчет общих данных
total_positives=$(find /home/microWakeWord_data/positives* -name "*.wav" 2>/dev/null | wc -l)
total_negatives=$(find /home/microWakeWord_data/negatives* -name "*.wav" 2>/dev/null | wc -l)
total_background=$(find /home/microWakeWord_data/background* -name "*.wav" 2>/dev/null | wc -l)
total_files=$((total_positives + total_negatives + total_background))

echo -e "${BLUE}📄 Общее количество WAV файлов: $total_files${NC}"
echo -e "${BLUE}🎯 Позитивные: $total_positives файлов${NC}"
echo -e "${BLUE}🚫 Негативные: $total_negatives файлов${NC}"
echo -e "${BLUE}🌊 Background: $total_background файлов${NC}"

# Подсчет размера
total_size=$(du -sh /home/microWakeWord_data/ 2>/dev/null | cut -f1)
echo -e "${BLUE}💾 Общий размер: $total_size${NC}"

echo ""
echo "🎯 ГОТОВЫЕ ДЛЯ ОБУЧЕНИЯ ДАННЫЕ:"
echo "==============================="

# Проверка готовых данных
ready_positives=$(find /home/microWakeWord_data/positives_final -name "*.wav" 2>/dev/null | wc -l)
ready_negatives=$(find /home/microWakeWord_data/negatives_real_sampled -name "*.wav" 2>/dev/null | wc -l)
ready_background=$(find /home/microWakeWord_data/background_data_sampled -name "*.wav" 2>/dev/null | wc -l)
ready_total=$((ready_positives + ready_negatives + ready_background))

echo -e "${GREEN}✅ Готовые позитивные: $ready_positives файлов${NC}"
echo -e "${GREEN}✅ Готовые негативные: $ready_negatives файлов${NC}"
echo -e "${GREEN}✅ Готовые background: $ready_background файлов${NC}"
echo -e "${GREEN}✅ Общее количество готовых: $ready_total файлов${NC}"

echo ""
echo "📋 СТАТУС ПРОВЕРКИ:"
echo "==================="

# Проверка критических директорий
critical_dirs=(
    "/home/microWakeWord_data/positives_final"
    "/home/microWakeWord_data/negatives_real"
    "/home/microWakeWord_data/negatives_real_sampled"
    "/home/microWakeWord_data/background_data"
    "/home/microWakeWord_data/background_data_sampled"
)

missing_critical=0
for dir in "${critical_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${RED}❌ ОТСУТСТВУЕТ КРИТИЧЕСКАЯ ДИРЕКТОРИЯ: $dir${NC}"
        missing_critical=$((missing_critical + 1))
    fi
done

if [ $missing_critical -eq 0 ]; then
    echo -e "${GREEN}✅ ВСЕ КРИТИЧЕСКИЕ ДИРЕКТОРИИ НА МЕСТЕ!${NC}"
    echo -e "${GREEN}✅ КОНТРАКТ НА ДАННЫЕ СОБЛЮДАЕТСЯ!${NC}"
    echo -e "${GREEN}✅ ГОТОВО К ОБУЧЕНИЮ: $ready_total файлов${NC}"
    exit 0
else
    echo -e "${RED}❌ ОБНАРУЖЕНЫ ОТСУТСТВУЮЩИЕ КРИТИЧЕСКИЕ ДИРЕКТОРИИ!${NC}"
    echo -e "${RED}❌ КОНТРАКТ НА ДАННЫЕ НАРУШЕН!${NC}"
    exit 1
fi