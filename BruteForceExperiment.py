import itertools
import os
import pathlib
from datetime import datetime

# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
# import tensorflow_datasets as tf_ds
import json

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
# from IPython import display
import math

from DataProcessing import *

AUTOTUNE = 8


# Set seed for experiment reproducibility
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)


def train_parametrized_model(data_sets,
                commands,
                conv1_filters: int = 32,
                conv1_kernel: int = 5,
                conv2_filters: int = 64,
                conv2_kernel: int = 5,
                dense1_units: int = 128,
                ):

    train_ds = data_sets[0]

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_ds.map(lambda x, _: x))

    input_shape = None
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)

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

    EPOCHS = 30
    return train_model(data_sets, model, EPOCHS)


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

        model, accuracy = train_parametrized_model(datasets, commands, *model_config)
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

# confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 8))
# # sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
# #             annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# plt.show()

if __name__ == "__main__":
    main()
