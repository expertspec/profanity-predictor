import io
import os

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
import warnings

warnings.filterwarnings("ignore")

import whisper
import speech_recognition as sr
from decord import AudioReader, bridge

bridge.set_bridge(new_bridge="torch")

import torch
from torch.utils.data import DataLoader

from src.features.tools import speech_to_text
from src.preprocessing import dataset_preparation
from src.models.prediction_model import PredictionModel
from src.features.feature_dataset import FeatureDataset


args = {"energy_threshold": 1000, "record_timeout": 2.5, "phrase_timeout": 3}

last_sample = bytes()
data_queue = Queue()
recorder = sr.Recognizer()
recorder.energy_threshold = args["energy_threshold"]
recorder.dynamic_energy_threshold = False

record_timeout = args["record_timeout"]
phrase_timeout = args["phrase_timeout"]

# Download models
_device = torch.device("cpu")
stt_model = whisper.load_model("small", device=_device)
weights = "./weights/model_attention_asr.pt"

prediction_model = PredictionModel(221, 1024, 2, 3).to(_device)
prediction_model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
prediction_model.eval()

path_to_banned_words = "./data/banned_words.txt"

with open(path_to_banned_words) as f:
    banned_words = f.readlines()
    banned_words = [word.strip() for word in banned_words]

temp_file = NamedTemporaryFile().name

source = sr.Microphone(sample_rate=16000)

with source:
    recorder.adjust_for_ambient_noise(source, duration=1)
    print("Start")


def signal_processing(temp_path):
    signal = AudioReader(temp_path, sample_rate6=1000, mono=True)
    transcribe = speech_to_text.transcribe_signal(stt_model, signal=signal[:][0])
    timemarks = speech_to_text.get_all_words(transcribe)
    samples = dataset_preparation.annotation_to_features(
        annotation={temp_path: timemarks}, signal=signal, banned_words=banned_words
    )
    samples = dataset_preparation.get_samples(samples)
    dataset = FeatureDataset(samples, 17, 7)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

    return next(iter(dataloader))[0]


def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to recieve audio data when recordings finish.
    Grab the raw bytes and push it into the thread safe queue.
    audio: An AudioData containing the recorded bytes.
    """
    data = audio.get_raw_data()
    data_queue.put(data)
    print("|")


recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

# Clear the console
os.system("cls" if os.name == "nt" else "clear")

phrase_time = datetime.utcnow()

# Stream processing
while True:
    try:
        now = datetime.utcnow()
        # Pull raw audio from the queue.
        if not data_queue.empty():
            phrase_complete = False
            # Condition to complete the phrase.
            if now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            phrase_time = now

            # Concatenate current audio data with the latest audio data.
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Convert the raw data to wav data.
            audio_data = sr.AudioData(
                last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH
            )
            wav_data = io.BytesIO(audio_data.get_wav_data())

            temp_path = temp_file + ".wav"
            with open(temp_path, "w+b") as f:
                f.write(wav_data.read())

            if phrase_complete:
                features = signal_processing(temp_path)
                print(prediction_model(features))
                last_sample = bytes()

            sleep(1)

    except KeyboardInterrupt:
        break
