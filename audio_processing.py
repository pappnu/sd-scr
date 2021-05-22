import os

import tensorflow as tf


def get_label(file_path):
    # each file's label is its directory's name
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def prepare_label_dataset(file_paths):
    # create dataset by splitting input tensor to individual items
    label_ds = tf.data.Dataset.from_tensor_slices(file_paths)

    # extract labels from filepaths
    # AUTOTUNE automatically optimizes data prefetching
    return label_ds.map(get_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def add_labels_to_dataset(dataset, file_paths, label_list=[]):
    label_ds = prepare_label_dataset(file_paths)

    if len(label_list) > 0:
        label_ds = label_ds.map(
            lambda label: tf.argmax(label == label_list),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    return tf.data.Dataset.zip((dataset, label_ds))


def get_stft(waveform, frame_length=512, frame_step=256):
    # apply short-time Fourier transform
    # splits signal into frames and applies Fourier transform on those
    # by default uses smallest power of 2 enclosing frame_length for fft size
    # uses hann window, an alternative would be hamming window
    # https://www.tensorflow.org/api_docs/python/tf/signal/stft
    return tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )


def get_mel_spectrogram(
    stft,
    sample_rate,
    num_mel_bins=40,
    lower_edge_hertz=20.0,
    upper_edge_hertz=4000.0,
    log=False,
    add_energy=False
):
    # spectrograms need only magnitude from stft
    # https://www.tensorflow.org/tutorials/audio/simple_audio#spectrogram
    spectrogram = tf.abs(stft)

    # the number of bins in the source spectrogram
    # understood to be fft_size // 2 + 1
    # // == floordiv
    # https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix#args
    num_spectrogram_bins = spectrogram.shape[-1]

    # calculate a weight matrix that can be used to re-weight a spectrogram to mel-scale
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz
    )

    # convert spectrogram to mel-scale
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    # print('mel spectrogram shape before: ', mel_spectrogram.shape)
    # print('mel spectrogram shape before: ', mel_spectrogram.shape[:-1])
    # # https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms#for_example
    # # why is this needed?
    # mel_spectrogram.set_shape(
    # spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print('mel spectrogram shape after: ', mel_spectrogram.shape)

    if log:
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    if add_energy:
        # Compute power spectrum of each frame
        audio_power = tf.math.square(spectrogram)

        # Compute total energy of each frame and collect them to a column vector
        energy = tf.reshape(tf.reduce_sum(audio_power, 1), [audio_power.shape[0], 1])

        mel_spectrogram = tf.concat([mel_spectrogram, energy], 1)

    return mel_spectrogram


# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas
# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
# edited to work with tf.tensors
def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A tensor of shape (NUMFRAMES, features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A tensor of shape (NUMFRAMES, features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = feat.shape[0]
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = tf.reshape((), (0, feat.shape[1]))
    padded = tf.pad(
        feat, tf.constant([[N, N], [0, 0]]), 'CONSTANT', 0
    )  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat = tf.concat([
            delta_feat,
            tf.reshape(
                tf.tensordot(
                    tf.range(-N, N + 1, 1, tf.float32), padded[t:t + 2 * N + 1], 1
                ) / denominator, (1, feat.shape[1])
            )
        ], 0)  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def get_mfcc(
    log_mel_spectrogram,
    num_mel_bins_to_pick=12,
    add_energy=False,
    add_first_delta=False,
    add_second_delta=False,
    symmetric_zero_padding=0,
):
    # If add_energy, assume that the last bin in log mel spectrograms represents energy and separate it
    if (add_energy):
        energy = tf.slice(
            log_mel_spectrogram, [0, log_mel_spectrogram.shape[1] - 1],
            [log_mel_spectrogram.shape[0], 1]
        )
        log_mel_spectrogram = tf.slice(
            log_mel_spectrogram, [0, 0],
            [log_mel_spectrogram.shape[0], log_mel_spectrogram.shape[1] - 1]
        )
    # https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms#for_example
    # Compute MFCCs from log mel spectrograms
    # Take num_mel_bins_to_pick bins
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[
        ..., :num_mel_bins_to_pick]

    # add symmetric_zero_padding vectors of zeroes to both ends of the time dimension
    if symmetric_zero_padding > 0:
        zero_pad = tf.zeros([symmetric_zero_padding, num_mel_bins_to_pick])
        mfcc = tf.concat([zero_pad, mfcc, zero_pad], 0)

    # Add energy back if it was separated
    if add_energy:
        mfcc = tf.concat([mfcc, energy], 1)

    if add_first_delta:
        mfcc_delta = delta(mfcc, 1)
        mfcc = tf.concat([mfcc, mfcc_delta], 1)

    if add_second_delta:
        mfcc_double_delta = delta(mfcc_delta, 1)
        mfcc = tf.concat([mfcc, mfcc_double_delta], 1)

    return mfcc


def load_audio(audio_file_path, sample_rate, clip_duration):
    audio_binary = tf.io.read_file(audio_file_path)

    # works only with 16bit wav files
    # audio file is assumed to have sample rate equal to sample_rate
    # scales to [-1.0, 1.0]
    # takes clip_duration seconds of audio
    # adds zero padding if clip is too short
    tensor, _ = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=int(sample_rate * clip_duration)
    )

    # remove last dimension, in this case the number of channels
    return tf.squeeze(tensor, axis=-1)


def prepare_waveform_dataset(
    file_paths,
    sample_rate=16000,
    clip_duration=1,
    add_labels=True,
    labels_to_integers=[],
    add_channels=False
):
    waveform_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    waveform_ds = waveform_ds.map(
        lambda file_path: load_audio(file_path, sample_rate, clip_duration),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if add_channels:
        waveform_ds = waveform_ds.map(
            lambda tensor: tf.expand_dims(tensor, -1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if add_labels:
        return add_labels_to_dataset(waveform_ds, file_paths, labels_to_integers)

    return waveform_ds


def prepare_mel_spectrogram_dataset(
    file_paths,
    sample_rate=16000,
    clip_duration=1,
    fft_frame_length=512,
    fft_frame_step=256,
    num_mel_bins=40,
    lower_edge_hertz=20.0,
    upper_edge_hertz=4000.0,
    log=False,
    add_energy=False,
    add_labels=True,
    labels_to_integers=[],
    add_channels=False
):
    waveform_ds = prepare_waveform_dataset(file_paths, sample_rate, clip_duration, False)

    # apply short time fourier transform to each waveform
    stft_ds = waveform_ds.map(
        lambda waveform:
        get_stft(waveform, frame_length=fft_frame_length, frame_step=fft_frame_step),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # get mel spectrograms
    mel_spectrogram_ds = stft_ds.map(
        lambda stft: get_mel_spectrogram(
            stft, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, log,
            add_energy
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if add_channels:
        mel_spectrogram_ds = mel_spectrogram_ds.map(
            lambda tensor: tf.expand_dims(tensor, -1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if add_labels:
        return add_labels_to_dataset(mel_spectrogram_ds, file_paths, labels_to_integers)

    return mel_spectrogram_ds


def prepare_mfcc_dataset(
    file_paths,
    sample_rate=16000,
    clip_duration=1,
    fft_frame_length=512,
    fft_frame_step=256,
    num_mel_bins=40,
    num_mel_bins_to_pick=12,
    lower_edge_hertz=20.0,
    upper_edge_hertz=4000.0,
    add_energy=False,
    add_first_delta=False,
    add_second_delta=False,
    symmetric_zero_padding=0,
    add_labels=True,
    labels_to_integers=[],
    add_channels=False
):
    mel_spectrogram_ds = prepare_mel_spectrogram_dataset(
        file_paths, sample_rate, clip_duration, fft_frame_length, fft_frame_step,
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, True, add_energy, False, [],
        False
    )

    mfcc_ds = mel_spectrogram_ds.map(
        lambda mel: get_mfcc(
            mel, num_mel_bins_to_pick, add_energy, add_first_delta,
            add_second_delta, symmetric_zero_padding
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if add_channels:
        mfcc_ds = mfcc_ds.map(
            lambda tensor: tf.expand_dims(tensor, -1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if add_labels:
        return add_labels_to_dataset(mfcc_ds, file_paths, labels_to_integers)

    return mfcc_ds
