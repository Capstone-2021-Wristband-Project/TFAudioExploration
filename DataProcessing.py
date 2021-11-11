"""
    Common functions for audio Processing.
"""

import tensorflow as tf
import math
import os
import numpy as np
from typing import Tuple


def get_hw_parallelism():
    return os.cpu_count()


def wav_to_tensor(audio_binary: tf.Tensor, target_effective_bit_width: int = 16) -> tf.Tensor:
    """
    Reads a Tensor of type string containing binary data in WAV format and convert it to a Tensor of type float32
    between 1 and -1.


    :param audio_binary Tensor containing binary blob in WAV format
    :param target_effective_bit_width: the effective sample bit_with to emulate, must be between 2 and 16 or a
           ValueError will be raised
    :raise ValueError: if the supplied target_effective_bit_with is invalid.
    :return: a Tensor of type float32 containing the sound data.
    """

    if not 2 <= target_effective_bit_width <= 16:
        raise ValueError("Only bit-widths between 2 and 16 inclusive are supported")

    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)

    # adjusting samples to simulate 8 bit samples
    # comment out the following 5 lines to revert to 16 bit samples
    audio = tf.math.multiply(tf.fill(tf.shape(audio), 32768.0), audio)
    audio = tf.cast(audio, tf.int32)
    audio = tf.math.divide(audio, tf.fill(tf.shape(audio), 2 ** (16 - target_effective_bit_width)))
    audio = tf.cast(audio, tf.float32)
    audio = tf.math.divide(audio, tf.fill(tf.shape(audio), math.pow(2, target_effective_bit_width - 1)))

    return audio


def get_label(file_path: tf.Tensor) -> tf.RaggedTensor:
    """
    Retrieves the label part of the file path.

    :param file_path: Tensor of type string containing the file path.
    :return: Tensor of type string containing the label.
    """
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path, effective_bit_width):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = wav_to_tensor(audio_binary, effective_bit_width)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram

def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def get_spectrogram_and_label_id(audio, label, commands):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


def get_spectrogram_and_expected_array(audio, label, commands):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_array = tf.cast((label == commands), tf.float32)
    constant_array = tf.constant([1.0 / len(commands)] * len(commands))
    condition = tf.tile(
        tf.expand_dims(tf.math.count_nonzero(label_array) == 0, axis=0),
        tf.expand_dims(tf.constant(len(commands)), axis=0)
    )
    label_array = tf.where(
        x=constant_array, y=label_array, condition=condition)
    # label_array = tf.transpose(tf.expand_dims(label_array, axis=1))
    return spectrogram, label_array


def preprocess_dataset(files, commands, effective_bit_width=16,
                       use_array=False, num_parallel_calls=get_hw_parallelism()):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(lambda x: get_waveform_and_label(x, effective_bit_width),
                             num_parallel_calls=num_parallel_calls)
    # if use_array:
    #     output_ds = output_ds.map(
    #         get_spectrogram_and_expected_array, num_parallel_calls=AUTOTUNE)
    # else:
    #     output_ds = output_ds.map(
    #         get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(lambda x, y: get_spectrogram_and_label_id(x, y, commands),
                              num_parallel_calls=num_parallel_calls)
    return output_ds


def training_prep(data_dir: str, effective_bit_width: int = 16):

    # data_dir = pathlib.Path('data/mini_speech_commands')
    # data_dir = pathlib.Path('data/speech_commands_aligned')
    # if not data_dir.exists():
    #     tf.keras.utils.get_file(
    #         'mini_speech_commands.zip',
    #         origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    #         extract=True,
    #         cache_dir='.', cache_subdir='data')

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    # commands = commands[commands != 'noise']
    print('Commands:', commands)

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
          len(tf.io.gfile.listdir(str(data_dir + '/' + commands[0]))))
    print('Example file tensor:', filenames[0])

    num_train_samples = int(num_samples * 0.8)
    num_val_samples = int(num_samples * 0.1)

    train_files = filenames[:num_train_samples]
    val_files = filenames[num_train_samples: num_train_samples + num_val_samples]
    test_files = filenames[num_train_samples + num_val_samples:]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    train_ds = preprocess_dataset(train_files, commands, effective_bit_width)
    val_ds = preprocess_dataset(val_files, commands, effective_bit_width)
    test_ds = preprocess_dataset(test_files, commands, effective_bit_width)

    return (train_ds, val_ds, test_ds), commands


def train_model(
        datasets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        model: tf.keras.models.Model,
        epoch_limit: int = 30,
        batch_size: int = 30,
        num_parallel_runs: int = get_hw_parallelism(),
        early_stopping_patience: int = 2
):
    train_ds = datasets[0]
    val_ds = datasets[1]
    test_ds = datasets[2]

    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(num_parallel_runs)
    val_ds = val_ds.cache().prefetch(num_parallel_runs)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch_limit,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=early_stopping_patience),
    )

    # metrics = history.history
    # plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    # plt.legend(['loss', 'val_loss'])
    # plt.show()

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    return model, test_acc
