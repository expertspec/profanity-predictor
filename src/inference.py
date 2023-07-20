from __future__ import annotations

from os import PathLike

import click
import torch
from src.preprocessing import dataset_preparation
from src.features.feature_dataset import FeatureDataset
from src.models.prediction_model import LSTM_attention
from torch.utils.data import DataLoader
from tqdm import tqdm


@click.command()
@click.argument("path_to_data", type=str)
@click.option("--device", type=str, help="Specify device to use")
@click.option("--weights", type=str, default="./weights/model_attention_asr.pt", help="Specify path to weights file")
@click.option("--path_to_banned_words", default="./data/banned_words.txt", type=str, help="Specify path to banned words file")

def get_samples_predictions(
    path_to_data: str | PathLike,
    device: torch.device | None = None,
    weights: str | PathLike = "./weights/model_attention_asr.pt",
    path_to_banned_words: str | PathLike = "./data/banned_words.txt",
) -> list:
    """Inference for predictions for samples from the dataset

    Args:
        path_to_data (str): path to the folder with the records for prediction
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        weights (str, optional): path to the model's weights. Defaults to "./weights".
        path_to_banned_words (str, optional): path to banned_words. Defaults to "./data/banned_words.txt".
    Returns:
        list: list with labels for every sample
    """
    
    _device = torch.device("cpu")
    if device is not None:
        _device = device

    with open(path_to_banned_words) as f:
        banned_words = f.readlines()
        banned_words = [word.strip() for word in banned_words]

    timemarks_for_target = dataset_preparation.get_annotation(path_to_data)
    files_features = dataset_preparation.annotation_to_features(
        timemarks_for_target, banned_words=banned_words
    )
    samples = dataset_preparation.get_samples(files_features)
    dataset = FeatureDataset(samples, 17, 7)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=4)
    prediction_model = LSTM_attention(221, 1024, 2, 3).to(_device)
    prediction_model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    prediction_model.eval()
    with torch.no_grad():
        predictions = []
        for elem, _ in tqdm(dataloader):
            batch_prediction = prediction_model(elem.to(_device))
            predictions.append([pred.argmax().to("cpu") for pred in batch_prediction])
    return predictions


if __name__ == "__main__":
    get_samples_predictions()
