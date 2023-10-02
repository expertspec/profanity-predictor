from __future__ import annotations

import json
import os
from collections import defaultdict
from os import PathLike
from typing import Union, Dict, List, Tuple

import torch
import torchaudio
import whisper
from decord import AudioReader, bridge
from src.features.tools import speech_to_text
from tqdm import tqdm

bridge.set_bridge(new_bridge="torch")


def get_annotation(
    signal: torch.Tensor = None,
    dir_path: str | PathLike = None,
    model: str = "local",
    device: torch.device | None = None,
) -> Dict:
    """Creates dict with annotations

    Args:
        signal (torch.Tensor): Audio signal,
        dir_path (str | Pathlike): Path to the dir with records.
        model (str, optional): Model configuration for speech recognition ['server', 'local']. Defaults to 'local'.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.

    Raises:
        NotImplementedError: If 'model' is not equal to 'server' or 'local'.
    """
    if model not in ["server", "local"]:
        raise NotImplementedError("'model' must be 'server' or 'local'.")

    _device = torch.device("cpu")
    if device is not None:
        _device = device

    if model == "server":
        stt_model = whisper.load_model("medium", device=_device)
    elif model == "local":
        stt_model = whisper.load_model("small", device=_device)

    # if signal:

    if dir_path:
        records = os.listdir(dir_path)
        if ".gitkeep" in records:
            records.remove(".gitkeep")

        timemarks_for_targets = {}
        for record_name in tqdm(records):
            file_path = os.path.join(dir_path, record_name)
            res = speech_to_text.transcribe_signal(
                stt_model=stt_model, record_path=file_path
            )
            timemarks_for_targets[file_path] = speech_to_text.get_all_words(res)

    return timemarks_for_targets


def get_samples(files_features: Union[str, PathLike, List]) -> List:
    """Function for getting samples with from record with sliding window

    Args:
        files_features (Union[str, PathLike, List]): _description_

    Returns:
        List: _description_
    """
    if isinstance(files_features, str):
        with open(files_features) as f:
            files_features = json.load(f)
    else:
        samples = []
        for elem in files_features:
            for num in range(0, len(files_features[elem]), 2):
                if files_features[elem][num : num + 7][-1]["mask"] == 1:
                    try:
                        samples.append(files_features[elem][num : num + 7])
                    except IndexError:
                        samples.append(files_features[elem][num:])
                elif files_features[elem][num : num + 7][-1]["mask"] == 0:
                    try:
                        samples.append(files_features[elem][num : num + 7])
                    except IndexError:
                        samples.append(files_features[elem][num:])
                elif files_features[elem][num : num + 7][-1]["mask"] == 2:
                    try:
                        samples.append(files_features[elem][num : num + 7])
                    except IndexError:
                        samples.append(files_features[elem][num:])
                elif files_features[elem][num:][-1]["mask"] == 2:
                    try:
                        samples.append(files_features[elem][num : num + 7])
                    except IndexError:
                        samples.append(files_features[elem][num:])
    return samples


def annotation_to_features(
    annotation: Dict | List,
    signal: torch.Tensor = None,
    output_path: str | PathLike = None,
    banned_words=None,
):
    """Extract features for every records from the list

    Args:
        annotation (Dict): _description_
        output_path (str | PathLike, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    files_features = defaultdict(list)
    if isinstance(annotation, dict):
        for name in tqdm(annotation.keys()):
            if annotation[name]:
                if signal:
                    files_features[name] = words_to_features(
                        timestamps=annotation[name],
                        signal=signal,
                        file_path=name,
                        banned_words=banned_words,
                    )
                else:
                    files_features[name] = words_to_features(
                        timestamps=annotation[name],
                        file_path=name,
                        banned_words=banned_words,
                    )

    if output_path:
        for name in files_features:
            for i in range(len(files_features[name])):
                # convert tensors to list for saving in json file
                files_features[name][i]["features"] = (
                    files_features[name][i]["features"].numpy().tolist()
                )
        with open(output_path, "w") as f:
            json.dump(files_features, f)

    return files_features


def words_to_features(
    timestamps: Dict,
    signal: torch.Tensor = None,
    file_path: str | PathLike = None,
    sr: int = 16000,
    banned_words=None,
) -> Dict:
    """Function to convert that extracts MFCCs based on timestamps for annotation

    Args:
        timestamps (_type_): _description_
        file_path (_type_): _description_
        sr (_type_, optional): _description_. Defaults to sample_rate.

    Returns:
        _type_: _description_
    """
    to_mfcc = torchaudio.transforms.MFCC(sr, n_mfcc=13)
    features = []
    if signal is None and file_path:
        signal = AudioReader(file_path, sample_rate=sr, mono=True)
    if timestamps[0]["start"] != 0:
        fragment = signal[:][0][: int(timestamps[0]["start"] * sr)]
        feature = to_mfcc(fragment)
        features.append(
            {
                "start": 0,
                "end": timestamps[0]["start"],
                "features": feature,
                "text": "",
                "mask": 0,
            }
        )
    for elem in timestamps:
        fragment = signal[:][0][int(elem["start"] * sr) : int(elem["end"] * sr)]
        try:
            if elem["text"] in banned_words:
                mask = 2
            else:
                mask = 1
            feature = to_mfcc(fragment)
            features.append(
                {
                    "start": elem["start"],
                    "end": elem["end"],
                    "features": feature,
                    "text": elem["text"],
                    "mask": mask,
                }
            )
        except RuntimeError:
            features.append(
                {
                    "start": elem["start"],
                    "end": elem["end"],
                    "features": [],
                    "text": "",
                    "mask": 0,
                }
            )

    return features
