from __future__ import annotations

import torch
from src.data import dataset_preparation
from src.features.feature_dataset import FeatureDataset
from src.models.prediction_model import LSTM_attention
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_samples_predictions(
    path_to_data: str, device: str = "cpu", weights: str = "./weights"
):
    """Inference for predictions for samples from the dataset

    Args:
        path_to_data (str): _description_
        device (str, optional): _description_. Defaults to "cpu".
        weights (str, optional): _description_. Defaults to "./weights".

    Returns:
        _type_: _description_
    """
    timemarks_for_target = dataset_preparation.get_annotation(path_to_data)
    files_features = dataset_preparation.annotation_to_features(timemarks_for_target)
    samples = dataset_preparation.get_samples(files_features)
    dataset = FeatureDataset(samples, 17, 7)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=4)
    prediction_model = LSTM_attention(221, 1024, 2, 3).to(device)
    prediction_model.load_state_dict(torch.load(weights))
    prediction_model.eval()
    with torch.no_grad():
        predictions = []
        for elem in tqdm(dataloader):
            batch_prediction = prediction_model(elem.to(device))
            predictions.append([pred.argmax().to("cpu") for pred in batch_prediction])
    return predictions
