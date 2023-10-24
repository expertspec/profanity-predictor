from __future__ import annotations

import json
from os import PathLike
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(
        self,
        data: Union[str, PathLike, List[Dict]],
        num_bins: int = 17,
        num_elems: int = 7,
    ):
        """Class for model's input fitting

        Args:
            data (Union[str, PathLike, List[Dict]]): path to list of features,
            or list which can be extracted with data/dataset_preparation.py
            num_bins (int, optional): number of MFCC bins. Defaults to 17.
            num_elems (int, optional): number of elements in sequence. Defaults to 7.
        """
        if isinstance(data, (str, PathLike)):
            with open(data) as f:
                self.data = json.load(f)
        else:
            self.data = data
        self.num_bins = num_bins
        self.num_elems = num_elems
        # pad element of silence
        self.pad = {
            "start": 0,
            "end": 0.42,
            "features": torch.rand(13, num_bins),
            "text": "",
            "mask": 0,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self._right_pad()
        features = self.get_features(idx)
        mask = self.get_mask(idx)
        return features, mask

    def _right_pad(self):
        for elem in range(len(self.data)):
            for i in range(len(self.data[elem])):
                # padding
                if type(self.data[elem][i]["features"]) != torch.Tensor:
                    self.data[elem][i]["features"] = self.pad
                else:
                    if self.data[elem][i]["features"].shape[1] < self.num_bins:
                        diff = self.num_bins - self.data[elem][i]["features"].shape[1]
                        self.data[elem][i]["features"] = torch.nn.functional.pad(
                            self.data[elem][i]["features"], (0, diff), "constant"
                        )
                    # truncation
                    elif self.data[elem][i]["features"].shape[1] > self.num_bins:
                        self.data[elem][i]["features"] = torch.narrow(
                            self.data[elem][i]["features"], 1, 0, self.num_bins
                        )
            if len(self.data[elem]) < self.num_elems:
                while len(self.data[elem]) < self.num_elems:
                    self.data[elem].append(self.pad)

    def get_features(self, idx):
        features = []
        for i in range(len(self.data[idx]))[:6]:
            features.append(self.data[idx][i]["features"])
        return torch.stack(features)

    def get_mask(self, idx):
        mask = []
        for i in range(len(self.data[idx]))[-1:]:
            mask.append(self.data[idx][i]["mask"])
        return torch.Tensor(mask)
