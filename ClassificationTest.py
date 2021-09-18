import itertools
import os
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import json

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import math

AUTOTUNE = 8


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


# def get_spectrogram_and_expected_array(audio, label):
#     spectrogram = get_spectrogram(audio)
#     spectrogram = tf.expand_dims(spectrogram, -1)
#     label_array = tf.cast((label == commands), tf.float32)
#     constant_array = tf.constant([1.0 / len(commands)] * len(commands))
#     condition = tf.tile(
#         tf.expand_dims(tf.math.count_nonzero(label_array) == 0, axis=0),
#         tf.expand_dims(tf.constant(len(commands)), axis=0)
#     )
#     label_array = tf.where(x=constant_array, y=label_array, condition=condition)
#     # label_array = tf.transpose(tf.expand_dims(label_array, axis=1))
#     return spectrogram, label_array


def preprocess_dataset(files, commands, effective_bit_width = 16, use_array=False):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(lambda x: get_waveform_and_label(x, effective_bit_width), num_parallel_calls=AUTOTUNE)
    # if use_array:
    #     output_ds = output_ds.map(
    #         get_spectrogram_and_expected_array, num_parallel_calls=AUTOTUNE)
    # else:
    #     output_ds = output_ds.map(
    #         get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(lambda x, y: get_spectrogram_and_label_id(x, y, commands), num_parallel_calls=AUTOTUNE)
    return output_ds


# Set seed for experiment reproducibility
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

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


def train_model(data_sets,
                commands,
                conv1_filters: int = 32,
                conv1_kernel: int = 5,
                conv2_filters: int = 64,
                conv2_kernel: int = 5,
                dense1_units: int = 128,
                ):

    train_ds = data_sets[0]
    val_ds = data_sets[1]
    test_ds = data_sets[2]

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_ds.map(lambda x, _: x))

    input_shape = None
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(conv1_filters, conv1_kernel, activation='relu'),
        layers.Conv2D(conv2_filters, conv2_kernel, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(dense1_units, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 30
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
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

def main():

    effective_bit_width = 8
    data_dir = 'data/speech_commands_aligned'

    datasets, commands = training_prep(data_dir, effective_bit_width)

    conv_filters = (16, 32, 64)
    conv_kernels = (3, 5, 7)
    dense_units = (32, 64, 128, 256)

    model_configs = itertools.product(conv_filters, conv_kernels, conv_filters, conv_kernels, dense_units)

    result_file_name = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

    with open(result_file_name, "w") as result_file:
        result_file.write(
            "data_path, bit_width, conv1_filters, conv1_kernel, conv2_filters, conv2_kernel,"
            " dense1_units, model_path, num_param, accuracy\n"
        )

    for model_config in model_configs:
        print(f"Running Config: {model_config}.")

        model, accuracy = train_model(datasets, commands, *model_config)
        num_params = model.count_params()

        model_path = f"model/bruteforce_experiment/model-{'-'.join(map(str, model_config))}" \
                     f"-{num_params}-{int(accuracy * 10000)}"
        model.save(model_path)

        model_dict = {
            "data_dir": data_dir,
            "bit_width": effective_bit_width,
            "categories": commands.tolist(),
            "model_config": model_config,
            "num_param": num_params,
            "accuracy": accuracy
        }

        with open(model_path + "/info.json", "w") as info_file:
            json.dump(model_dict, info_file)

        with open(result_file_name, "a") as result_file:
            result_file.write(
                f"{data_dir}, {effective_bit_width}, {', '.join(map(str, model_config))}, "
                f"{model_path}, {num_params}, {accuracy}\n"
            )
        pass



# model.save("model/audio_classify_test")
# model.save("model/audio_classify_aligned_8bit_test")
#
# test_audio = []
# test_labels = []
#
# for audio, label in test_ds:
#     test_audio.append(audio.numpy())
#     test_labels.append(label.numpy())
#
# test_audio = np.array(test_audio)
# test_labels = np.array(test_labels)
#
# y_pred = np.argmax(model.predict(test_audio), axis=1)
# y_true = test_labels
#
# test_acc = sum(y_pred == y_true) / len(y_true)
# print(f'Test set accuracy: {test_acc:.0%}')
#
# confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
#             annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# plt.show()

if __name__ == "__main__":
    main()