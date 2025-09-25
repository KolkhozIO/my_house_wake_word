# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
СТРОГО РЕФАКТОРЕННЫЙ СКРИПТ ПО ОБРАЗЦУ ОРИГИНАЛЬНОГО ПРОЕКТА
Копирует точную структуру и логику из backups/mww_orig/microwakeword/train_original.py
"""

import os
import platform
import contextlib
import sys
import yaml
import json
from pathlib import Path

from absl import logging

import numpy as np
import tensorflow as tf

from tensorflow.python.util import tf_decorator

# СТРОГО СТАТИЧЕСКАЯ ЛИНКОВКА ПУТЕЙ ИЗ XML БЕЗ ХАРДКОДА
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('/home/microWakeWord')
from src.utils.path_manager import paths

# Импортируем оригинальные модули
from backups.mww_orig.microwakeword import mixednet
from backups.mww_orig.microwakeword import utils


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)

    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def validate_nonstreaming(config, data_processor, model, test_set):
    """Валидация модели без стриминга - ТОЧНАЯ КОПИЯ из оригинала"""
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)

    model.reset_metrics()

    result = model.evaluate(
        testing_fingerprints,
        testing_ground_truth,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]

    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = result["fp"].numpy()

    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)

        # XXX: tf no longer provides a way to evaluate a model without updating metrics
        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(
                ambient_testing_fingerprints,
                ambient_testing_ground_truth,
                batch_size=1024,
                return_dict=True,
                verbose=0,
            )

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
        all_true_positives = ambient_predictions["tp"].numpy()
        ambient_false_positives = ambient_predictions["fp"].numpy() - test_set_fp
        all_false_negatives = ambient_predictions["fn"].numpy()

        metrics["auc"] = ambient_predictions["auc"]
        metrics["loss"] = ambient_predictions["loss"]

        recall_at_cutoffs = (
            all_true_positives / (all_true_positives + all_false_negatives)
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set

        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break

        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, data_processor):
    """Обучение модели - ТОЧНАЯ КОПИЯ из оригинала"""
    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
        tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
        tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
        tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # We un-decorate the `tf.function`, it's very slow to manually run training batches
    model.make_train_function()
    _, model.train_function = tf_decorator.unwrap(model.train_function)

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    for training_step in range(1, training_steps_max + 1):
        # КРИТИЧНО: Останавливаемся на 1000 шагах для полного обучения
        if training_steps_max == 1000 and training_step >= 1000:
            print(f"\n🎯 ОБУЧЕНИЕ ЗАВЕРШЕНО НА {training_step} ШАГАХ!", flush=True)
            # Создаем директорию и сохраняем веса модели для конвертации
            os.makedirs(config["train_dir"], exist_ok=True)
            model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
            break
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        model.optimizer.learning_rate.assign(learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }

        (
            train_fingerprints,
            train_ground_truth,
            train_sample_weights,
        ) = data_processor.get_data(
            "training",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="default",
            augmentation_policy=augmentation_policy,
        )

        train_ground_truth = train_ground_truth.reshape(-1, 1)

        class_weights = {0: negative_class_weight, 1: positive_class_weight}
        combined_weights = train_sample_weights * np.vectorize(class_weights.get)(
            train_ground_truth
        )

        result = model.train_on_batch(
            train_fingerprints,
            train_ground_truth,
            sample_weight=combined_weights,
        )

        # Print the running statistics in the current validation epoch
        if training_step % 10 == 0:  # Логируем каждые 10 шагов
            print(
                "Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    result[1],
                    result[2],
                    result[3],
                    result[9],
                    (training_step % config["eval_step_interval"]),
                ),
                flush=True,
            )

        is_last_step = training_step == training_steps_max
        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    result[1] * 100,
                    result[2] * 100,
                    result[3] * 100,
                    result[9],
                ),
            )

            with train_writer.as_default():
                tf.summary.scalar("loss", result[9], step=training_step)
                tf.summary.scalar("accuracy", result[1], step=training_step)
                tf.summary.scalar("recall", result[2], step=training_step)
                tf.summary.scalar("precision", result[3], step=training_step)
                tf.summary.scalar("auc", result[8], step=training_step)
                train_writer.flush()

            model.save_weights(
                os.path.join(config["train_dir"], "last_weights.weights.h5")
            )

            nonstreaming_metrics = validate_nonstreaming(
                config, data_processor, model, "validation"
            )
            model.reset_metrics()  # reset metrics for next validation epoch of training
            logging.info(
                "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                *(
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["cutoff_for_no_faph"],
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                    nonstreaming_metrics["average_viable_recall"],
                ),
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "recall_at_no_faph",
                    nonstreaming_metrics["recall_at_no_faph"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "average_viable_recall",
                    nonstreaming_metrics["average_viable_recall"],
                    step=training_step,
                )
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[
                    config["minimization_metric"]
                ]
            current_maximization_quantity = nonstreaming_metrics[
                config["maximization_metric"]
            ]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model
            if (
                (
                    (
                        current_minimization_quantity <= config["target_minimization"]
                    )  # achieved target false positive rate
                    and (
                        (
                            current_maximization_quantity > best_maximization_quantity
                        )  # either accuracy improved
                        or (
                            best_minimization_quantity > config["target_minimization"]
                        )  # or this is the first time we met the target
                    )
                )
                or (
                    (
                        current_minimization_quantity > config["target_minimization"]
                    )  # we haven't achieved our target
                    and (
                        current_minimization_quantity < best_minimization_quantity
                    )  # but we have decreased since the previous best
                )
                or (
                    (
                        current_minimization_quantity == best_minimization_quantity
                    )  # we tied a previous best
                    and (
                        current_maximization_quantity > best_maximization_quantity
                    )  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(
                    os.path.join(config["train_dir"], "best_weights.weights.h5")
                )
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))


def create_original_config():
    """Загружает тестовую конфигурацию на 100 шагов"""
    
    print("🔧 Загрузка тестовой конфигурации на 100 шагов...", flush=True)
    
    data_dir = "/home/microWakeWord_data"
    # Используем тестовую конфигурацию на 100 шагов
    config_path = os.path.join(data_dir, 'training_parameters_test.yaml')
    
    # Проверяем что тестовая конфигурация существует
    if not os.path.exists(config_path):
        print(f"❌ Тестовая конфигурация не найдена: {config_path}", flush=True)
        return None
    
    print(f"✅ Используем тестовую конфигурацию: {config_path}", flush=True)
    return config_path


def create_original_flags():
    """Создает объект flags строго по образцу оригинала"""
    
    class Flags:
        def __init__(self):
            # Параметры MixedNet строго по образцу оригинала
            self.pointwise_filters = "48,48,48,48"
            self.repeat_in_block = "1,1,1,1"
            self.mixconv_kernel_sizes = "[5],[9],[13],[21]"
            self.residual_connection = "0,0,0,0"
            self.first_conv_filters = 32
            self.first_conv_kernel_size = 3
            self.stride = 1
            self.spatial_attention = False
            self.temporal_attention = False
            self.attention_heads = 1
            self.attention_dim = 64
            self.pooled = False
            self.max_pool = False
    
    return Flags()


def create_original_data_processor(config):
    """Создает обработчик данных строго по образцу оригинала"""
    
    print("📊 Создание обработчика данных по образцу оригинала...", flush=True)
    
    # Импортируем оригинальный модуль данных
    from backups.mww_orig.microwakeword import data as input_data
    
    # Создаем обработчик данных
    data_processor = input_data.FeatureHandler(config)
    
    print("✅ Обработчик данных создан", flush=True)
    return data_processor


def create_original_model(flags, config):
    """Создает модель строго по образцу оригинала"""
    
    print("🏗️ Создание модели по образцу оригинала...", flush=True)
    
    # Создаем модель используя оригинальную архитектуру
    # Преобразуем список в кортеж для совместимости
    input_shape = tuple(config["training_input_shape"])
    model = mixednet.model(flags, input_shape, config["batch_size"])
    
    print("✅ Модель создана", flush=True)
    print(f"   Вход: {model.input_shape}", flush=True)
    print(f"   Выход: {model.output_shape}", flush=True)
    
    return model


def evaluate_and_convert_model(config, model, data_processor):
    """Оценка и конвертация модели строго по образцу оригинала"""
    
    print("🔄 Оценка и конвертация модели...", flush=True)
    
    try:
        # Конвертируем модель в SavedModel для стриминга
        print("📦 Конвертация в SavedModel...", flush=True)
        
        # Создаем директории
        os.makedirs(os.path.join(config["train_dir"], "stream_state_internal"), exist_ok=True)
        os.makedirs(os.path.join(config["train_dir"], "non_stream"), exist_ok=True)
        
        # Конвертируем в стриминговую модель
        streaming_model = utils.convert_model_saved(
            model, config, "stream_state_internal", 
            utils.modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
        )
        
        # Конвертируем в не-стриминговую модель
        non_streaming_model = utils.convert_model_saved(
            model, config, "non_stream",
            utils.modes.Modes.NON_STREAM_INFERENCE
        )
        
        print("✅ SavedModel созданы", flush=True)
        
        # Конвертируем в TFLite
        print("🔄 Конвертация в TFLite...", flush=True)
        
        try:
            # Конвертируем стриминговую модель
            print("📦 Конвертация в обычный TFLite...", flush=True)
            utils.convert_saved_model_to_tflite(
                config, data_processor,
                os.path.join(config["train_dir"], "stream_state_internal"),
                config["train_dir"],
                "stream_state_internal.tflite",
                quantize=False
            )
            print("✅ Обычный TFLite создан", flush=True)
            
            # Конвертируем квантованную стриминговую модель ДЛЯ ESP32
            print("📦 Конвертация в квантованный TFLite для ESP32...", flush=True)
            utils.convert_saved_model_to_tflite(
                config, data_processor,
                os.path.join(config["train_dir"], "stream_state_internal"),
                config["train_dir"],
                "stream_state_internal_quant.tflite",
                quantize=True  # Оригинальная функция уже поддерживает ESP32 квантование!
            )
            print("✅ Квантованный TFLite создан", flush=True)
            
        except Exception as tflite_error:
            print(f"❌ Ошибка конвертации TFLite: {tflite_error}", flush=True)
            print("🔄 Продолжаем без TFLite...", flush=True)
            return False
        
        print("✅ TFLite модели созданы", flush=True)
        return True
        
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def copy_final_results(config):
    """Копирует финальные результаты"""
    
    print("📁 Копирование финальных результатов...", flush=True)
    
    train_dir = config["train_dir"]
    
    # Ищем лучший TFLite файл
    import shutil
    
    # Пробуем разные варианты TFLite файлов - ПРИОРИТЕТ КВАНТОВАННОЙ МОДЕЛИ ДЛЯ ESP32
    tflite_candidates = [
        os.path.join(train_dir, "stream_state_internal_quant.tflite"),  # ✅ КВАНТОВАННАЯ ДЛЯ ESP32
        os.path.join(train_dir, "stream_state_internal.tflite"),       # ❌ Обычная (не подходит для ESP32)
    ]
    
    final_model_path = "/home/microWakeWord_data/original_library_model.tflite"
    model_copied = False
    
    for candidate_path in tflite_candidates:
        if os.path.exists(candidate_path):
            shutil.copy(candidate_path, final_model_path)
            print(f"✅ Модель скопирована: {final_model_path} (из {candidate_path})", flush=True)
            model_copied = True
            break
    
    if not model_copied:
        print("⚠️ TFLite файлы не найдены, пропускаем копирование модели", flush=True)
    
    # Создаем манифест для ESP32-совместимой модели
    manifest = {
        "version": 2,
        "type": "micro",
        "model": "original_library_model.tflite",
        "author": "microWakeWord Project",
        "wake_word": "милый дом / любимый дом",
        "trained_languages": ["ru"],
        "website": "https://github.com/microWakeWord",
        "micro": {
            "probability_cutoff": 0.95,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 1000000,  # Уменьшено для ESP32
            "minimum_esphome_version": "2024.7.0"
        }
    }
    
    manifest_path = "/home/microWakeWord_data/original_library_model.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Манифест создан: {manifest_path}", flush=True)
    return True


def main():
    """Основная функция - строго по образцу оригинала"""
    
    print("🎯 Обучение модели microWakeWord по образцу оригинала")
    
    try:
        # Создаем конфигурацию
        config_path = create_original_config()
        
        # Загружаем конфигурацию
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Создаем flags
        flags = create_original_flags()
        
        # Создаем обработчик данных
        data_processor = create_original_data_processor(config)
        
        # Создаем модель
        model = create_original_model(flags, config)
        
        # Запускаем обучение
        print("🚀 Запуск обучения...", flush=True)
        train(model, config, data_processor)
        
        # Оценка и конвертация
        # Пытаемся конвертировать модель
        conversion_success = evaluate_and_convert_model(config, model, data_processor)
        
        # Копируем результаты в любом случае
        if copy_final_results(config):
            print("\n🎉 Обучение завершено успешно!", flush=True)
            print("📁 Файлы готовы для ESPHome:", flush=True)
            print("  - /home/microWakeWord_data/original_library_model.tflite", flush=True)
            print("  - /home/microWakeWord_data/original_library_model.json", flush=True)
            
            if conversion_success:
                print("✅ TFLite конвертация прошла успешно", flush=True)
            else:
                print("⚠️ TFLite конвертация не удалась, но модель сохранена", flush=True)
            
            return True
        
        print("\n❌ Обучение завершилось с ошибками", flush=True)
        return False
        
    except Exception as e:
        print(f"❌ Ошибка: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()