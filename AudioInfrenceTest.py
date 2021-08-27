import numpy as np
import pyaudio
from typing import Iterable
import struct
from collections import deque
import tensorflow as tf

CHUNK = 1000  # samples
WIDTH = 2  # bytes per sample
BIT_WIDTH = WIDTH * 8  # bits per sample
SAMPLE_MAX = 1 << BIT_WIDTH
CHANNELS = 1  #
SAMPLE_RATE = 16000

commands = ('down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes')

def decode_raw_samples(raw_samples: bytes) -> Iterable[float]:
    samples = struct.unpack("h" * CHUNK, raw_samples)
    return [x / 32768 for x in samples]


p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

sample_buffer = deque(maxlen=int(SAMPLE_RATE * 1))

model = tf.keras.Sequential([tf.keras.models.load_model("model/audio_classify_test"), tf.keras.layers.Softmax()])

print("* recording")
while True:
    raw_samples = stream.read(CHUNK)
    samples = decode_raw_samples(raw_samples)
    sample_buffer.extend(samples)

    if len(sample_buffer) == int(SAMPLE_RATE * 1):
        sample_tensor = tf.convert_to_tensor(sample_buffer, dtype=tf.float32)
        spectrogram = tf.signal.stft(
            sample_tensor, frame_length=255, frame_step=128)

        spectrogram = tf.abs(spectrogram)

        prediction: np.ndarray = model.predict(tf.expand_dims(spectrogram, axis=0))
        likely_command_index = prediction.argmax(1)[0]
        likely_command_chance = prediction[0, likely_command_index]
        # print(dict(zip(commands, prediction.tolist()[0])))
        if likely_command_chance > 0.9 and commands[likely_command_index] != 'noise':
            print(f"Detected command {commands[likely_command_index]}, {likely_command_chance}")
            sample_buffer.clear()

print("* done")

stream.stop_stream()
stream.close()

p.terminate()