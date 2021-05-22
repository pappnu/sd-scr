import json
import timeit

import numpy as np
import tensorflow as tf
from keras_flops import get_flops
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from audio_processing import (prepare_mfcc_dataset, prepare_mel_spectrogram_dataset)
from visualization import (draw_spectrogram_from_tensors)


def train_model(
    model, train_dataset, validation_dataset, loss, optimizer, epochs, callbacks
):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return [model, history]


def test_model(
    model,
    test_dataset,
    false_dataset=None,
    thresholds=[],
    num_timer_samples=0,
    timer_iters=10
):
    test_times = []
    for test_batch, label in test_dataset.take(num_timer_samples):
        predict_duration = timeit.timeit(
            lambda: model.predict(test_batch), number=timer_iters
        )
        test_times.append(predict_duration)

    test_loss, test_acc = model.evaluate(test_dataset)

    predictions = np.array([])
    prediction_labels = np.array([])
    labels = np.array([])
    for test_batch, label in test_dataset:
        prediction = model.predict(test_batch)
        predictions = np.concatenate([predictions, prediction],
                                     axis=0) if predictions.size else prediction
        prediction_labels = np.concatenate([
            prediction_labels, np.argmax(prediction, axis=-1)
        ])
        labels = np.concatenate([labels, label.numpy()])

    # columns represent the predictions and rows represent the real labels
    confusion_matrix = tf.math.confusion_matrix(labels, prediction_labels)

    thresholding_results = {
        'thresholds': thresholds,
        'false_alarm_rates': [],
        'miss_rates': [],
    }
    if false_dataset:
        predictions_false = np.array([])
        for batch, label in false_dataset:
            predictions_false = np.concatenate(
                [predictions_false, model.predict(batch)]
            ) if predictions_false.size else model.predict(batch)

        false_set_n = np.shape(predictions_false)[0]

        for threshold in thresholds:
            false_alarms = 0
            correct_classifications = 0
            missed_classifications = 0
            for prediction, prediction_label, label in zip(
                predictions, prediction_labels, labels
            ):
                if np.equal(prediction_label, label):
                    correct_classifications += 1
                    if not (np.max(prediction) >= threshold):
                        missed_classifications += 1

            for prediction in predictions_false:
                if np.max(prediction) > threshold:
                    false_alarms += 1

            miss_rate = 0
            if correct_classifications:
                miss_rate = missed_classifications / correct_classifications

            thresholding_results['false_alarm_rates'].append(false_alarms / false_set_n)
            thresholding_results['miss_rates'].append(miss_rate)

    return [
        test_loss,
        test_acc,
        test_times,
        confusion_matrix.numpy(),
        thresholding_results,
    ]


def prepare_dataset(dataset_type, file_paths, batch_size, cache_path='', **keywords):
    dataset = None

    if dataset_type == 'mel_spectrogram':
        dataset = prepare_mel_spectrogram_dataset(
            file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    elif dataset_type == 'mfcc':
        dataset = prepare_mfcc_dataset(
            file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    return dataset


def prepare_train_val_test_datasets(
    dataset_type,
    train_file_paths,
    val_file_paths,
    test_file_paths,
    batch_size,
    cache_path='',
    **keywords
):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if dataset_type == 'mel_spectrogram':
        train_dataset = prepare_mel_spectrogram_dataset(
            train_file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(train_file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        test_dataset = prepare_mel_spectrogram_dataset(
            test_file_paths, **keywords
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = prepare_mel_spectrogram_dataset(
            val_file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(val_file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    elif dataset_type == 'mfcc':
        train_dataset = prepare_mfcc_dataset(
            train_file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(train_file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        test_dataset = prepare_mfcc_dataset(test_file_paths,
                                            **keywords).batch(batch_size).prefetch(
                                                tf.data.experimental.AUTOTUNE
                                            )
        val_dataset = prepare_mfcc_dataset(
            val_file_paths, **keywords
        ).batch(batch_size).cache(cache_path).shuffle(len(val_file_paths)).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    return [train_dataset, val_dataset, test_dataset]


def train_test_and_collect_data(
    model_constructor,
    model_kwargs,
    loss,
    optimizer,
    epochs,
    callbacks,
    train_ds,
    val_ds,
    test_ds,
    labels,
    false_ds=None,
    thresholds=[],
    num_timer_samples=0,
    timer_iters=100,
    load_weights='',
    model_save_path='',
):
    num_labels = len(labels)

    test_samples = [sample for sample in test_ds.take(8)]
    # draw_spectrogram_from_tensors(test_samples)

    for sample, _ in train_ds.take(1):
        # exclude batch dimension
        input_shape = sample.shape[1:]

    print('Number of classes', num_labels)
    print('Input shape', input_shape)

    model = model_constructor(input_shape, num_labels, **model_kwargs)

    trainable_count = np.sum([
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    ])
    non_trainable_count = np.sum([
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    ])
    flops = get_flops(model, batch_size=1)

    print('trainable params: ', trainable_count)
    print('non-trainable params', non_trainable_count)
    print('total params', trainable_count + non_trainable_count)
    print('flops', flops)
    model.summary()

    if load_weights:
        try:
            with open(load_weights + '.json', 'r') as f:
                progress = json.load(f)
                epochs = epochs - int(progress['epoch']) - 1
        except IOError:
            pass

        try:
            model.load_weights(load_weights)
        except tf.errors.NotFoundError:
            print('Weights not found.')

    model, history = train_model(
        model,
        train_ds,
        val_ds,
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        callbacks=callbacks
    )

    test_loss, test_acc, test_times, confusion_matrix, thresholding_results = test_model(
        model, test_ds, false_ds, thresholds, num_timer_samples, timer_iters
    )

    print('Test accuracy: ', test_acc)

    if model_save_path:
        model.save(model_save_path)

    return {
        'model': model,
        'epoch': history.epoch,
        'history': history.history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_times': test_times,
        'confusion_matrix': confusion_matrix.tolist(),
        'thresholding_results': thresholding_results,
        'total_params': int(trainable_count + non_trainable_count),
        'trainable_params': int(trainable_count),
        'non_trainable_params': int(non_trainable_count),
        'flops': flops
    }
