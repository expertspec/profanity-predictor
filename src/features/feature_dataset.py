from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):

    def __init__(self,
     data,
     num_bins,
     num_elems):
        
        self.data = data
        self.num_bins = num_bins
        self.num_elems = num_elems
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # text = self.get_texts(idx)
        features = self.get_features(idx)
        mask = self.get_mask(idx)
        return features, mask

    def _right_pad_if_necessary(self):
        for elem in range(len(self.data)):
            for i in range(len(self.data[elem])):
                # padding
                if type(self.data[elem][i]["features"]) != torch.Tensor:
                    self.data[elem][i]["features"] = pad
                else:
                    if self.data[elem][i]["features"].shape[1] < self.num_bins:
                        diff = self.num_bins - self.data[elem][i]["features"].shape[1]
                        self.data[elem][i]["features"]  = torch.nn.functional.pad(self.data[elem][i]["features"], (0, diff), "constant")
                    # truncation
                    elif self.data[elem][i]["features"].shape[1] > self.num_bins:
                        diff = self.num_bins - self.data[elem][i]["features"].shape[1]
                        self.data[elem][i]["features"]  = torch.narrow(self.data[elem][i]["features"], 1, 0, self.num_bins)
            if len(self.data[elem]) < self.num_elems:
                while len(self.data[elem]) < self.num_elems:
                    self.data[elem].append(pad)

    # def get_texts(self, idx):
    #     texts = []
    #     for i in range(len(self.data[idx])):
    #         texts.append(self.data[idx][i]["text"])
    #     return torch.Tensor(texts)
        
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