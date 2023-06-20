from __future__ import annotations

import json
import os
from collections import defaultdict
from os import PathLike
from typing import Dict, List, Tuple

import torch
import torchaudio
import whisper
from decord import AudioReader, bridge
from src.features.tools import speech_to_text
from tqdm import tqdm

bridge.set_bridge(new_bridge="torch")


with open("banned_words.txt") as f:
    banned_words = f.readlines()
    banned_words = [word.strip() for word in banned_words]
    
def get_annotation(
    dir_path: str | PathLike,
    lang: str = "en",
    model: str = "server",
    device: torch.device | None = None
    )->Dict:
    """Creates dict with annotations

    Args:
        video_path (str | Pathlike): Path to the local video file.
        lang (str, optional): Language for speech recognition ['ru', 'en']. Defaults to 'en'.
        model (str, optional): Model configuration for speech recognition ['server', 'local']. Defaults to 'server'.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.

    Raises:
        NotImplementedError: If 'lang' is not equal to 'en' or 'ru'.
        NotImplementedError: If 'model' is not equal to 'server' or 'local'.
    """
    if lang not in ["en", "ru"]:
        raise NotImplementedError("'lang' must be 'en' or 'ru'.")
    if model not in ["server", "local"]:
        raise NotImplementedError("'model' must be 'server' or 'local'.")

    _device = torch.device("cpu")
    if device is not None:
        _device = device

    if model == "server":
        stt_model = whisper.load_model("medium", device=_device)
    elif model == "local":
        stt_model = whisper.load_model("small", device=_device)
        
    records = os.listdir(dir_path)
    timemarks_for_targets = {}
    for record_name in tqdm(records):        
        file_path = os.path.join(dir_path, record_name)
        res = speech_to_text.transcribe_video(stt_model=stt_model, record_path=file_path, lang=lang)
        timemarks_for_targets[file_path] = speech_to_text.get_all_words(res)
    
    return timemarks_for_targets

def annotation_to_features(annotation: Dict, output_path: str | PathLike = None):
    """Extract features for every records from the list

    Args:
        annotation (Dict): _description_
        output_path (str | PathLike, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    files_features = defaultdict(list)

    for name in tqdm(annotation.keys()):
        if annotation[name]:
            files_features[name] = words_to_features(annotation[name], name)
    
    if output_path:
        for name in files_features:
            for i in range(len(files_features[name])):
                # convert tensors to list for saving in json file
                files_features[name][i]["features"] = files_features[name][i]["features"].numpy().tolist()
        with open(output_path, "w") as f:
            json.dump(files_features, f)
    
    return files_features

def words_to_features(timestamps: Dict,
                      file_path: str | PathLike,
                      sr: int = 16000) -> Dict:
    """Function to convert that extracts MFCCs based on timestamps for annotation

    Args:
        timestamps (_type_): _description_
        file_path (_type_): _description_
        sr (_type_, optional): _description_. Defaults to sample_rate.
        output_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    to_mfcc = torchaudio.transforms.MFCC(sr, n_mfcc=13)
    features = []
    signal = AudioReader(file_path, sample_rate=sr, mono=True)
    if timestamps[0]["start"] != 0:
        fragment = signal[:][0][:int(timestamps[0]["start"]*sr)]
        feature = to_mfcc(fragment)
        features.append({"start": 0, "end": timestamps[0]["start"], "features": feature,
                         "text": "", "mask": 0})
    for elem in timestamps[0:]:
        fragment = signal[:][0][int(elem["start"]*sr):int(elem["end"]*sr)]
        try:
            if elem["text"] in banned_words:
                mask = 2
            else:
                mask = 1
            feature = to_mfcc(fragment)
            features.append({"start": elem["start"], "end": elem["end"], "features": feature,
                             "text": elem["text"], "mask": mask})
        except RuntimeError:
            features.append({"start": elem["start"], "end": elem["end"], "features": [],
                             "text": "", "mask": 0})
        

    return features