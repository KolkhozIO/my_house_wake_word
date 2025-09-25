#!/usr/bin/env python3
"""
Применение аугментаций к сгенерированным TTS данным
"""

import os
import subprocess
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import glob

def apply_tempo_augmentation(input_file, output_file, tempo_factor):
    """Применяет изменение темпа без изменения тона"""
    cmd = ['sox', input_file, output_file, 'tempo', str(tempo_factor)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_pitch_augmentation(input_file, output_file, pitch_semitones):
    """Применяет изменение тона"""
    cmd = ['sox', input_file, output_file, 'pitch', str(pitch_semitones)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_bandlimit_augmentation(input_file, output_file):
    """Применяет бандлимит (имитация телефонного канала)"""
    cmd = ['sox', input_file, output_file, 'highpass', '120', 'lowpass', '3800']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_reverb_augmentation(input_file, output_file):
    """Применяет реверберацию"""
    cmd = ['sox', input_file, output_file, 'reverb', '20', '50', '100']
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_volume_augmentation(input_file, output_file, volume_factor):
    """Применяет изменение громкости"""
    cmd = ['sox', input_file, output_file, 'vol', str(volume_factor)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_padding_augmentation(input_file, output_file, pad_before, pad_after):
    """Добавляет паузы до и после"""
    cmd = ['sox', input_file, output_file, 'pad', str(pad_before), str(pad_after)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_codec_augmentation(input_file, output_file):
    """Применяет кодек-аугментацию через ffmpeg"""
    temp_ogg = output_file.replace('.wav', '_temp.ogg')
    try:
        # Конвертируем в Opus с низким битрейтом
        subprocess.run([
            'ffmpeg', '-i', input_file, '-c:a', 'libopus', '-b:a', '16k', 
            '-y', temp_ogg
        ], check=True, capture_output=True)
        
        # Конвертируем обратно в WAV
        subprocess.run([
            'ffmpeg', '-i', temp_ogg, '-y', output_file
        ], check=True, capture_output=True)
        
        # Удаляем временный файл
        os.remove(temp_ogg)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        return f"ERROR: {output_file} - {e} (STDOUT: {e.stdout}, STDERR: {e.stderr})"

def apply_noise_augmentation(input_file, output_file, noise_level):
    """Добавляет розовый шум"""
    noise_file = output_file.replace('.wav', '_noise.wav')
    try:
        # Получаем параметры оригинального файла
        duration_cmd = ['soxi', '-D', input_file]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(duration_result.stdout.strip())
        
        sample_rate_cmd = ['soxi', '-r', input_file]
        sample_rate_result = subprocess.run(sample_rate_cmd, capture_output=True, text=True, check=True)
        sample_rate = int(sample_rate_result.stdout.strip())
        
        # Генерируем розовый шум с теми же параметрами
        subprocess.run([
            'sox', '-n', '-r', str(sample_rate), '-c', '1', noise_file, 
            'synth', str(duration), 'pinknoise'
        ], check=True, capture_output=True)
        
        # Микшируем с оригиналом
        subprocess.run([
            'sox', '-m', '-v', '1', input_file, '-v', str(noise_level), 
            noise_file, output_file
        ], check=True, capture_output=True)
        
        # Удаляем временный файл шума
        if os.path.exists(noise_file):
            os.remove(noise_file)
        return f"OK: {output_file}"
    except subprocess.CalledProcessError as e:
        # Очищаем временный файл при ошибке
        if os.path.exists(noise_file):
            os.remove(noise_file)
        return f"ERROR: {output_file} - {e}"

def apply_single_augmentation(args):
    """Применяет одну аугментацию к файлу"""
    input_file, output_file, aug_type, aug_params = args
    
    if aug_type == 'tempo':
        return apply_tempo_augmentation(input_file, output_file, aug_params['tempo'])
    elif aug_type == 'pitch':
        return apply_pitch_augmentation(input_file, output_file, aug_params['pitch'])
    elif aug_type == 'bandlimit':
        return apply_bandlimit_augmentation(input_file, output_file)
    elif aug_type == 'reverb':
        return apply_reverb_augmentation(input_file, output_file)
    elif aug_type == 'volume':
        return apply_volume_augmentation(input_file, output_file, aug_params['volume'])
    elif aug_type == 'padding':
        return apply_padding_augmentation(input_file, output_file, 
                                        aug_params['pad_before'], aug_params['pad_after'])
    elif aug_type == 'codec':
        return apply_codec_augmentation(input_file, output_file)
    elif aug_type == 'noise':
        return apply_noise_augmentation(input_file, output_file, aug_params['noise_level'])
    else:
        return f"ERROR: Unknown augmentation type {aug_type}"

def augment_directory(input_dir, output_dir, augmentation_types, progress_bar=None):
    """Применяет аугментации ко всем файлам в директории"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем все WAV файлы
    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    # Создаем задачи для аугментаций
    tasks = []
    for wav_file in wav_files:
        base_name = os.path.basename(wav_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        for aug_type in augmentation_types:
            if aug_type == 'tempo':
                # Темп: 0.9, 1.1
                for tempo in [0.9, 1.1]:
                    output_file = os.path.join(output_dir, f"{name_without_ext}_tempo_{tempo}.wav")
                    tasks.append((wav_file, output_file, 'tempo', {'tempo': tempo}))
            
            elif aug_type == 'pitch':
                # Тон: ±1 полутон
                for pitch in [100, -100]:
                    output_file = os.path.join(output_dir, f"{name_without_ext}_pitch_{pitch}.wav")
                    tasks.append((wav_file, output_file, 'pitch', {'pitch': pitch}))
            
            elif aug_type == 'bandlimit':
                output_file = os.path.join(output_dir, f"{name_without_ext}_bandlimit.wav")
                tasks.append((wav_file, output_file, 'bandlimit', {}))
            
            elif aug_type == 'reverb':
                output_file = os.path.join(output_dir, f"{name_without_ext}_reverb.wav")
                tasks.append((wav_file, output_file, 'reverb', {}))
            
            elif aug_type == 'volume':
                # Громкость: 0.4, 2.0
                for volume in [0.4, 2.0]:
                    output_file = os.path.join(output_dir, f"{name_without_ext}_vol_{volume}.wav")
                    tasks.append((wav_file, output_file, 'volume', {'volume': volume}))
            
            elif aug_type == 'padding':
                # Паузы: 100-300мс до и после
                pad_before = random.uniform(0.1, 0.3)
                pad_after = random.uniform(0.1, 0.3)
                output_file = os.path.join(output_dir, f"{name_without_ext}_pad_{pad_before:.2f}_{pad_after:.2f}.wav")
                tasks.append((wav_file, output_file, 'padding', {'pad_before': pad_before, 'pad_after': pad_after}))
            
            elif aug_type == 'codec':
                output_file = os.path.join(output_dir, f"{name_without_ext}_codec.wav")
                tasks.append((wav_file, output_file, 'codec', {}))
            
            elif aug_type == 'noise':
                # Шум: разные уровни
                for noise_level in [0.1, 0.2, 0.3]:
                    output_file = os.path.join(output_dir, f"{name_without_ext}_noise_{noise_level}.wav")
                    tasks.append((wav_file, output_file, 'noise', {'noise_level': noise_level}))
    
    # Копируем оригинальные файлы
    for wav_file in wav_files:
        base_name = os.path.basename(wav_file)
        shutil.copy2(wav_file, os.path.join(output_dir, base_name))
    
    # Применяем аугментации параллельно
    max_workers = cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(apply_single_augmentation, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("ERROR"):
                print(result)
            if progress_bar:
                progress_bar.update(1)

def main():
    print("Применение аугментаций к сгенерированным данным...")
    
    # Базовая директория для данных
    data_dir = "/home/microWakeWord_data"
    
    # Проверяем наличие исходных данных
    if not os.path.exists(os.path.join(data_dir, "positives_both")):
        print("ОШИБКА: Директория positives_both не найдена!")
        print("Сначала запустите generate_both_phrases.py")
        return
    
    if not os.path.exists(os.path.join(data_dir, "negatives_both")):
        print("ОШИБКА: Директория negatives_both не найдена!")
        print("Сначала запустите generate_both_phrases.py")
        return
    
    # Создаем временные директории для аугментированных данных
    temp_positives_aug = "positives_both_aug_temp"
    temp_negatives_aug = "negatives_both_aug_temp"
    
    # Очищаем временные директории если существуют
    if os.path.exists(temp_positives_aug):
        shutil.rmtree(temp_positives_aug, ignore_errors=True)
    if os.path.exists(temp_negatives_aug):
        shutil.rmtree(temp_negatives_aug, ignore_errors=True)
    
    # Определяем типы аугментаций
    augmentation_types = [
        'tempo',      # Изменение темпа
        'pitch',      # Изменение тона
        'bandlimit',  # Бандлимит
        'reverb',     # Реверберация
        'volume',     # Изменение громкости
        'padding',    # Паузы
        'codec',      # Кодек-аугментация
        'noise'       # Добавление шума
    ]
    
    # Подсчитываем общее количество задач
    pos_files = len(glob.glob(os.path.join(data_dir, "positives_both", "*.wav")))
    neg_files = len(glob.glob(os.path.join(data_dir, "negatives_both", "*.wav")))
    
    # Подсчитываем количество аугментаций на файл
    augs_per_file = 0
    for aug_type in augmentation_types:
        if aug_type == 'tempo':
            augs_per_file += 2  # 0.9, 1.1
        elif aug_type == 'pitch':
            augs_per_file += 2  # +100, -100
        elif aug_type == 'bandlimit':
            augs_per_file += 1
        elif aug_type == 'reverb':
            augs_per_file += 1
        elif aug_type == 'volume':
            augs_per_file += 2  # 0.4, 2.0
        elif aug_type == 'padding':
            augs_per_file += 1
        elif aug_type == 'codec':
            augs_per_file += 1
        elif aug_type == 'noise':
            augs_per_file += 3  # 0.1, 0.2, 0.3
    
    total_tasks = (pos_files + neg_files) * augs_per_file
    
    print(f"Используем {cpu_count()} ядер для {total_tasks} аугментаций...")
    print(f"Позитивных файлов: {pos_files}")
    print(f"Негативных файлов: {neg_files}")
    print(f"Аугментаций на файл: {augs_per_file}")
    
    # Создаем общий прогресс-бар
    with tqdm(total=total_tasks, desc="Применение аугментаций", unit="файл") as pbar:
        # Аугментируем позитивные данные
        print("Аугментируем позитивные данные...")
        augment_directory(os.path.join(data_dir, "positives_both"), temp_positives_aug, augmentation_types, pbar)
        
        # Аугментируем негативные данные
        print("Аугментируем негативные данные...")
        augment_directory(os.path.join(data_dir, "negatives_both"), temp_negatives_aug, augmentation_types, pbar)
    
    # Атомарная замена директорий
    print("Атомарная замена директорий...")
    
    # Удаляем старые директории если существуют
    final_positives = os.path.join(data_dir, "positives_both_augmented")
    final_negatives = os.path.join(data_dir, "negatives_both_augmented")
    
    if os.path.exists(final_positives):
        shutil.rmtree(final_positives)
    if os.path.exists(final_negatives):
        shutil.rmtree(final_negatives)
    
    # Переименовываем временные директории
    os.rename(temp_positives_aug, final_positives)
    os.rename(temp_negatives_aug, final_negatives)
    
    print("Аугментации завершены!")
    print(f"Аугментированных позитивных файлов: {len(os.listdir(final_positives))}")
    print(f"Аугментированных негативных файлов: {len(os.listdir(final_negatives))}")

if __name__ == "__main__":
    main()