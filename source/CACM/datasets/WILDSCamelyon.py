import torch 
import torchvision
from CACM.datasets.base_dataset import MultipleDomainDataset
from wilds.datasets.wilds_dataset import WILDSDataset
from pathlib import Path
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from wilds.common.utils import subsample_idxs
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
import pandas as pd
import numpy as np
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None,
            train=False):
        self.name = metadata_name + "_" + str(metadata_value)
        
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)

        metadata_array = wilds_dataset.metadata_array
        id_val_indices = None
        if train:
            split_array = torch.tensor(wilds_dataset.split_array)
            id_val_indices = torch.where(split_array == 1)[0]
        else:
            id_val_indices = torch.tensor([])
        split_array = wilds_dataset.split_array

        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]
        if id_val_indices.numel() > 0:
            subset_indices = subset_indices[~torch.isin(subset_indices, id_val_indices)]
        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform
        self.metadata_array = wilds_dataset.metadata_array

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        a = self.metadata_array[self.indices[i]][:1] * 1.0
        return x, y, a

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 96, 96)
    def __init__(self, dataset, metadata_name):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.datasets = []
        train_data = dataset.get_subset('train')
        id_val = dataset.get_subset('id_val', transform=transform)
        ood_val = dataset.get_subset('val', transform=transform)
        ood_test = dataset.get_subset('test', transform=transform)
        for i, metadata_value in enumerate(
                self.metadata_values(train_data, metadata_name)):
            env_transform = transform
            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform,train=True)

            self.datasets.append(env_dataset)
        


        self.datasets.append(ood_val)
        self.datasets.append(ood_test)
        self.datasets.append(id_val)
        self.input_shape = (3, 96, 96,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4", "mix"]
    def __init__(self, root, download=True):
        dataset = Camelyon17Dataset(root_dir=root, download=download)
        super().__init__(
            dataset, "hospital")