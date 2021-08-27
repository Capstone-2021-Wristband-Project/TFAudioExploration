import numpy as np
import pyaudio
from typing import Iterable
import struct
from collections import deque
import tensorflow as tf
import wave
from utils import prompt_device_index, byte_width_to_format_char

CHUNK = 1000  # samples per chunk
MICROPHONE_WIDTH = 2  # bytes per sample at the microphone
MICROPHONE_BIT_WIDTH = MICROPHONE_WIDTH * 8
MICROPHONE_MAX = 1 << (MICROPHONE_BIT_WIDTH - 1)
TARGET_BIT_WIDTH = 8  # target bit with to simulate
TARGET_MAX = TARGET_BIT_WIDTH << (TARGET_BIT_WIDTH - 1)
MICROPHONE_TO_TARGET_FACTOR = MICROPHONE_MAX // TARGET_MAX
CHANNELS = 1  # number of Channels, one for mono, 2 for stereo
SAMPLE_RATE = 16000  # in Hz

PEAK_PERCENT = 0.05
PEAK_LEVEL = int(PEAK_PERCENT * TARGET_MAX)

commands = ('down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes')


def unpack_raw_samples(raw_samples: bytes, format_char: str) -> Iterable[int]:
    unpacked_values = struct.unpack("=" + format_char * CHUNK, raw_samples)
    unpacked_values = [x // MICROPHONE_TO_TARGET_FACTOR for x in unpacked_values]
    return unpacked_values


def decode_raw_samples(unpacked_samples: Iterable[int], sample_max: int) -> Iterable[float]:
    return [x / sample_max for x in unpacked_samples]


def main():
    p = pyaudio.PyAudio()
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #             print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    input_device_index = prompt_device_index(p)

    stream = p.open(format=p.get_format_from_width(MICROPHONE_WIDTH),
                    input_device_index=input_device_index,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    sample_buffer = deque()

    model = tf.keras.Sequential(
        [tf.keras.models.load_model("model/audio_classify_aligned_8bit_test"), tf.keras.layers.Softmax()])

    print("* recording")

    detected = False
    count = 0

    while True:
        raw_samples = stream.read(CHUNK)
        unpacked_samples = tuple(unpack_raw_samples(raw_samples, byte_width_to_format_char(MICROPHONE_WIDTH)))
        if detected:
            sample_buffer.extend(decode_raw_samples(unpacked_samples, TARGET_MAX))
            if len(sample_buffer) >= int(SAMPLE_RATE * 1):
                samples_of_interest = tuple(sample_buffer)[:int(SAMPLE_RATE * 1)]
                with wave.open(f"sample_{count}.wav", "wb") as out:
                    out.setnchannels(CHANNELS)
                    out.setsampwidth(MICROPHONE_WIDTH)
                    out.setframerate(SAMPLE_RATE)
                    out.writeframes(struct.pack("h" * int(SAMPLE_RATE * 1),
                                                *tuple(map(lambda x: int(x * MICROPHONE_MAX), samples_of_interest))))
                sample_tensor = tf.convert_to_tensor(samples_of_interest, dtype=tf.float32)
                spectrogram = tf.signal.stft(
                    sample_tensor, frame_length=255, frame_step=128)

                spectrogram = tf.abs(spectrogram)

                prediction: np.ndarray = model.predict(tf.expand_dims(spectrogram, axis=0))
                likely_command_index = prediction.argmax(1)[0]
                likely_command_chance = prediction[0, likely_command_index]
                # print(dict(zip(commands, prediction.tolist()[0])))
                if likely_command_chance > 0.5 and commands[likely_command_index] != 'noise':
                    print(f"Detected command {commands[likely_command_index]}, {likely_command_chance}")
                sample_buffer.clear()
                detected = False
                count = count + 1
        else:
            max_value = max(unpacked_samples)
            if max_value > PEAK_LEVEL:
                # index = unpacked_samples.index(max_value)
                index = next(x for x, val in enumerate(unpacked_samples) if val > PEAK_LEVEL)
                sample_buffer.extend(decode_raw_samples(unpacked_samples, TARGET_MAX)[index:])
                detected = True
                print(f"{max_value=}")
                print("detected!")

    print("* done")

    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == '__main__':
    main()
