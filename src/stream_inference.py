import io
import os
import re
import json
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import warnings

warnings.filterwarnings("ignore")


args = {"energy_threshold": 1000, "record_timeout": 1, "phrase_timeout": 2.5}

rus_reg = r"[а-яА-Я0-9]"

last_sample = bytes()
data_queue = Queue()
recorder = sr.Recognizer()
recorder.energy_threshold = args["energy_threshold"]
recorder.dynamic_energy_threshold = False

source = sr.Microphone(sample_rate=16000)

record_timeout = args["record_timeout"]
phrase_timeout = args["phrase_timeout"]

temp_file = NamedTemporaryFile().name
transcription = [""]

with source:
    recorder.adjust_for_ambient_noise(source, duration=1)
    print("Start")


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to recieve audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)
    print("|")


recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
idx = 0
phrases = {}  # saves texts

# Flush stdout.
print("", end="", flush=True)

# Clear the console
os.system("cls" if os.name == "nt" else "clear")

phrase_time = datetime.utcnow()

while True:
    try:
        now = datetime.utcnow()
        # Pull raw audio from the queue.
        if not data_queue.empty():
            phrase_complete = False
            # If enough time has passed between recordings => the phrase is complete.
            if now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            phrase_time = now

            # Concatenate our current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Convert the raw data to wav data.
            audio_data = sr.AudioData(
                last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH
            )
            wav_data = io.BytesIO(audio_data.get_wav_data())

            # Write wav data to the temporary file as bytes.
            with open(temp_file, "w+b") as f:
                f.write(wav_data.read())

            if phrase_complete:
                # classify. get prediction
                # print prediction
                print(temp_file)
                last_sample = bytes()

            sleep(1)

    except KeyboardInterrupt:
        break
