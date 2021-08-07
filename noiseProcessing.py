import os
import pydub
import random

SPLITS = 2000

for file in os.listdir("noise"):
    if file.endswith(".wav"):
        file_name = os.path.splitext(os.path.basename(file))[0]
        audio = pydub.AudioSegment.from_file(f"noise/{file}")
        latest_start = len(audio) - 1000

        for i in range(SPLITS):
            start = random.randrange(0, latest_start)
            segment = audio[start:start + 1000]
            segment.export(f"data/mini_speech_commands/noise/{file_name}_segment{i}.wav", format="wav")
