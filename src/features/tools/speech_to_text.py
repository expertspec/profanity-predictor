from __future__ import annotations

from os import PathLike
from typing import Dict, List, Tuple

import src.features.tools.transcribe as transcribe
import torch
import whisper


def transcribe_video(
    stt_model,
    record_path: str | PathLike,
    lang: str = "en",
) -> Dict:
    """Speech recognition module from video.

    Args:
        stt_model: whisper model for transcribation.
        record_path (str | Pathlike): Path to the local video file.
        lang (str): Language
    """

    transribation = transcribe.transcribe_timestamped(
        model=stt_model, audio=record_path, language=lang
    )

    return transribation


def get_all_words(transcribation: Dict) -> Tuple[List, str]:
    """Get all stamps with words from the transcribed text.

    Args:
        transcribation (Dict): Speech recognition module results.
    """
    all_words = []
    for segment in transcribation["segments"]:
        for word in segment["words"]:
            all_words.append(word)

    return all_words