import shutil
import wave
import struct
import os

def process_data(src:str, dst:str):
    if os.path.splitext(src)[-1] != ".wav":
        return shutil.copy(src=src, dst=dst)

    print(f"{src=}, {dst=}")
    in_file: wave.Wave_read
    with wave.open(src, "rb") as in_file:
        raw_data = in_file.readframes(in_file.getnframes())
        data = struct.unpack("h" * in_file.getnframes(), raw_data)
        threshold = int(0.3 * max(max(data), abs(min(data))))
        start_index = next(x for x, val in enumerate(data) if abs(val) > threshold)
        data = data[start_index:]

        if len(data) < 16000:
            data = data + (0,) * (16000 - len(data))

        data = data[:16000]

        out_file: wave.Wave_write
        with wave.open(dst, "wb") as out_file:
            out_file.setnchannels(1)
            out_file.setsampwidth(2)
            out_file.setframerate(16000)
            out_file.writeframes(struct.pack("h" * len(data), *data))


shutil.copytree(src="data/mini_speech_commands", dst="data/speech_commands_aligned", copy_function=process_data)
