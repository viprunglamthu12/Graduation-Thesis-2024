import os
import torch
import torchvision
import random
import warnings 
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity

import tensorflow_datasets as tfds

from dowhy.causal_prediction_selfcode.datasets.base_dataset import MultipleDomainDataset


class SmallNORB(VisionDataset):

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - four-legged animals", 
        "1 - human figures", 
        "2 - airplanes", 
        "3 - trucks", 
        "4 - cars"
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    @property
    def source_files(self) -> List:
        return [
            'dataset_info.json',
            'features.json',
            'label_category.labels.txt',
            'smallnorb-test.tfrecord-00000-of-00001',
            'smallnorb-train.tfrecord-00000-of-00001'
        ]
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, '2.0.0')

    def __init__(
        self, 
        root: str = None, 
        train: bool = True, 
        transforms: Callable[..., Any] | None = None, 
        transform: Callable[..., Any] | None = None, 
        target_transform: Callable[..., Any] | None = None,
        download: bool = True
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.train = train

        if download:
            self.download()

        self.data, self.data2, self.targets, self.lightings, self.azimuths = self._load_data()

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, filename))
            for filename in self.source_files
        )

    def download(self):

        if self._check_exists():
            return
        
        # os.makedirs(self.raw_folder, exist_ok=True)
        
        try:
            tfds.load('smallnorb', data_dir=self.root, download=True)
        except Exception as e:
            raise Exception(f'Exception while downloading:\n{e}')
        
    def _load_data(self):
        # Load the dataset using TensorFlow Datasets
        dataset, info = tfds.load('smallnorb', data_dir=self.root, download=False, split='train' if self.train else 'test', with_info=True)

        images = []
        image2s = []
        labels = []
        lightings = []
        azimuths = []
        for example in dataset:
            images.append(np.squeeze(example['image'].numpy()))
            image2s.append(np.squeeze(example['image2'].numpy()))
            labels.append(example['label_category'].numpy())
            lightings.append(example['label_lighting'].numpy())
            azimuths.append(example['label_azimuth'].numpy())

        # Convert lists to numpy arrays
        images = np.array(images)
        image2s = np.array(image2s)
        labels = np.array(labels)
        lightings = np.array(lightings)
        azimuths = np.array(azimuths)

        # return images, labels, lightings, azimuths

        images_tensor = torch.tensor(images, dtype=torch.uint8)
        image2s_tensor = torch.tensor(image2s, dtype=torch.uint8)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        lightings_tensor = torch.tensor(lightings, dtype=torch.long)
        azimuths_tensor = torch.tensor(azimuths, dtype=torch.long)

        return images_tensor, image2s_tensor,  labels_tensor, lightings_tensor, azimuths_tensor
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    

class SmallNORBCausalAttribute(MultipleDomainDataset):
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["+90%", "+95%", "-100%", "-100%"]
    INPUT_SHAPE = (5, 48, 48)

    def __init__(self, root, download=True) -> None:
        super().__init__()

        if root is None:
            raise ValueError("Data directory not specified!")
        
        original_dataset_tr = SmallNORB(root, train=True, download=download)
        
        original_images = original_dataset_tr.data
        original_image2s = original_dataset_tr.data2
        original_labels = original_dataset_tr.targets
        original_lightings = original_dataset_tr.lightings
        # original_azimuths = original_dataset_tr.azimuths

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_image2s = original_image2s[shuffle]
        original_labels = original_labels[shuffle]
        original_lightings = original_lightings[shuffle]
        # original_azimuths = original_azimuths[shuffle]

        self.datasets = []

        environments = (0.1, 0.05, 1)
        for i, env in enumerate(environments[:-1]):
            images = original_images[:20000][i::2]
            image2s = original_image2s[:20000][i::2]
            labels = original_labels[:20000][i::2]
            lightings = original_lightings[:20000][i::2]
            # self.datasets.append(self.lighting_dataset(images, image2s, labels, lightings, env))
            self.datasets.append(self.lighting_dataset_5_channels(images, labels, lightings, env))

        images = original_images[20000:]
        image2s = original_image2s[20000:]
        labels = original_labels[20000:]
        lightings = original_lightings[20000:]
        # self.datasets.append(self.lighting_dataset(images, image2s, labels, lightings, environment=environments[-1]))
        self.datasets.append(self.lighting_dataset_5_channels(images, labels, lightings, environments[-1]))

        # test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_image2s = original_dataset_te.data2
        original_labels = original_dataset_te.targets
        original_lightings = original_dataset_te.lightings
        # self.datasets.append(self.lighting_dataset(original_images, original_image2s, original_labels, original_lightings, environments[-1]))
        self.datasets.append(self.lighting_dataset_5_channels(original_images, original_labels, original_lightings, environments[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def lighting_dataset_5_channels(self, images, labels, _lightings, environment):

        # images = images.reshape((-1, 480, 480, ))[:, ::2, ::2]
        images = images[:, ::2, ::2]

        labels = self.add_noise(labels, 0.05)
        labels = labels.float()

        # _images, _labels, _lightings = self.lightings_selection(images, labels, lightings, environment)

        lightings = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))

        images = torch.stack([images, images, images, images, images], dim=1)

        images[torch.tensor(range(len(images))), ((1 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((2 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((3 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((4 + lightings) % 5).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        a = torch.unsqueeze(lightings, 1)

        return TensorDataset(x, y, a)
    
    def lighting_dataset(self, images, image2s, labels, lightings, environment):
        
        images = images[:, ::2, ::2]
        image2s = image2s[:, ::2, ::2]

        labels = self.add_noise(labels, 0.05)
        labels = labels.float()

        _images, _image2s, _labels, _lightings = self.lightings_selection(images, image2s, labels, lightings, environment)

        stacked_images = torch.stack([_images, _image2s], dim=1)

        x = stacked_images.float().div_(255.0)
        y = _labels.view(-1).long()
        a = torch.unsqueeze(_lightings, 1)

        return TensorDataset(x, y, a)

    def add_noise(self, labels: List, rate: float = 0.05):

        n_changes = int(len(labels) * rate)

        indices_to_change = random.sample(range(len(labels)), n_changes)

        for index in indices_to_change:
            labels[index] = random.choice([label for label in range(5) if label != labels[index]])

        return labels
    
    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        c = (a + b) % 5
        # for i in range(len(b)):
        #     if b[i] == 1:
        #         c[i] = (c[i] + random.randint(0, 3)) % 5
        return c

    def lightings_selection(self, images, image2s, labels, lightings, environment):

        _images = []
        _image2s = []
        _labels = []
        _lightings = []
        _not_hold_indices = []

        if environment != 1:
            for i in range(len(images)):
                if torch.equal(labels[i], lightings[i]):
                    _images.append(images[i])
                    _image2s.append(image2s[i])
                    _labels.append(labels[i])
                    _lightings.append(lightings[i])
                else:
                    _not_hold_indices.append(i)


            n_error_elements = int(environment*len(_images))

            for i in range(n_error_elements):
                if i < len(_not_hold_indices):
                    _images.append(images[_not_hold_indices[i]])
                    _image2s.append(image2s[_not_hold_indices[i]])
                    _labels.append(labels[_not_hold_indices[i]])
                    _lightings.append(lightings[_not_hold_indices[i]])
                else:
                    break
        else:
            for i in range(len(images)):
                if not torch.equal(labels[i], lightings[i]):
                    _images.append(images[i])
                    _image2s.append(image2s[i])
                    _labels.append(labels[i])
                    _lightings.append(lightings[i])

        return torch.tensor(np.array(_images), dtype=torch.uint8), torch.tensor(np.array(_image2s), dtype=torch.uint8), torch.LongTensor(np.array(_labels)), torch.LongTensor(np.array(_lightings))



class SmallNORBIndAttribute(MultipleDomainDataset):
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["0", "6", "17", "17"]
    INPUT_SHAPE = (1, 48, 48)

    def __init__(self, root, download=True):
        """Class for SmallNORBIndAttribute dataset.

        :param root: The directory where data can be found (or should be downloaded to, if it does not exist).
        :param download: Binary flag indicating whether data should be downloaded
        :returns: an instance of MultipleDomainDataset class

        """

        super().__init__()

        if root is None:
            raise ValueError("Data directory is not specified!")
        
        self.init_azimuth_selection(root, download)
        # self.init_random_selection(root, download)
    
    def init_random_selection(self, root, download):

        original_dataset_tr = SmallNORB(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets

        shuffle = torch.randperm(len(original_images))
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        azimuths = [0, 6, 17]
        for i, env in enumerate(azimuths[:-1]):
            images = original_images[:20000][i::2]
            labels = original_labels[:20000][i::2]
            self.datasets.append(self.azimuth_dataset(images, labels, i, azimuths[i]))
        images = original_images[20000:]
        labels = original_labels[20000:]
        self.datasets.append(self.azimuth_dataset(images, labels, len(azimuths) - 1, azimuths[-1]))

        # test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)
        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        self.datasets.append(self.azimuth_dataset(original_images, original_labels, len(azimuths) - 1, azimuths[-1]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def init_azimuth_selection(self, root, download):

        original_dataset_tr = SmallNORB(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets
        original_azimuths = original_dataset_tr.azimuths

        domain_1_indices = []
        domain_2_indices = []
        domain_3_indices = []

        self.datasets = []

        for i in range(len(original_images)):
            if original_azimuths[i] < 6:
                domain_1_indices.append(i)
            elif original_azimuths[i] >= 6 and original_azimuths[i] < 12:
                domain_2_indices.append(i)
            elif original_azimuths[i] >= 12:
                domain_3_indices.append(i)
        
        domain_1_images = torch.index_select(original_images, 0, torch.LongTensor(domain_1_indices))
        domain_2_images = torch.index_select(original_images, 0, torch.LongTensor(domain_2_indices))
        domain_3_images = torch.index_select(original_images, 0, torch.LongTensor(domain_3_indices))

        domain_1_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_1_indices))
        domain_2_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_2_indices))
        domain_3_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_3_indices))

        # domain_1_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_1_indices))
        # domain_2_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_2_indices))
        # domain_3_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_3_indices))

        azimuths = [0, 6, 17]
        self.datasets.append(self.azimuth_dataset(domain_1_images, domain_1_labels, 0, azimuths[0]))
        self.datasets.append(self.azimuth_dataset(domain_2_images, domain_2_labels, 1, azimuths[1]))
        self.datasets.append(self.azimuth_dataset(domain_3_images, domain_3_labels, 2, azimuths[2]))

        # Test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)

        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        original_azimuths = original_dataset_te.azimuths

        domain_4_indices = []

        for i in range(len(original_images)):
            if original_azimuths[i] >= 12:
                domain_4_indices.append(i)
        
        domain_4_images = torch.index_select(original_images, 0, torch.LongTensor(domain_4_indices))

        domain_4_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_4_indices))

        # domain_4_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_4_indices))

        self.datasets.append(self.azimuth_dataset(domain_4_images, domain_4_labels, 2, azimuths[2]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def azimuth_dataset(self, images, labels, env_id, angle):


        images = images.reshape((-1, 96, 96))[:, ::2, ::2]

        labels = self.add_noise(labels, 0.05)
        labels = labels.float()

        stacked_images = torch.stack([images], dim=1)

        x = stacked_images.float().div_(255.0)

        y = labels.view(-1).long()
        a = torch.full((y.shape[0],), env_id, dtype=torch.float32)
        a = torch.unsqueeze(a, 1)

        return TensorDataset(x, y, a)

    def add_noise(self, labels: List, rate: float = 0.05):

        n_changes = int(len(labels) * rate)

        indices_to_change = random.sample(range(len(labels)), n_changes)

        for index in indices_to_change:
            labels[index] = random.choice([label for label in range(5) if label != labels[index]])

        return labels
    

class SmallNORBCausalIndAttribute(MultipleDomainDataset):
    N_STEPS = 2001
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['+90%, 0', '+95%, 6', '-100%, 17', '-100%, 17']
    INPUT_SHAPE = (5, 48, 48)

    def __init__(self, root, download=True):

        super().__init__()

        if root is None:
            raise ValueError("Data directory is not specified!")
        
        original_dataset_tr = SmallNORB(root, train=True, download=download)

        original_images = original_dataset_tr.data
        original_labels = original_dataset_tr.targets
        original_azimuths = original_dataset_tr.azimuths

        domain_1_indices = []
        domain_2_indices = []
        domain_3_indices = []

        self.datasets = []

        for i in range(len(original_images)):
            if original_azimuths[i] < 6:
                domain_1_indices.append(i)
            elif original_azimuths[i] >= 6 and original_azimuths[i] < 12:
                domain_2_indices.append(i)
            elif original_azimuths[i] >= 12:
                domain_3_indices.append(i)
        
        domain_1_images = torch.index_select(original_images, 0, torch.LongTensor(domain_1_indices))
        domain_2_images = torch.index_select(original_images, 0, torch.LongTensor(domain_2_indices))
        domain_3_images = torch.index_select(original_images, 0, torch.LongTensor(domain_3_indices))

        domain_1_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_1_indices))
        domain_2_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_2_indices))
        domain_3_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_3_indices))

        # domain_1_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_1_indices))
        # domain_2_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_2_indices))
        # domain_3_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_3_indices))

        environments = [0.1, 0.05, 1]
        azimuths = [0, 6, 17]
        self.datasets.append(self.lighting_azimuth_dataset(domain_1_images, domain_1_labels, environments[0], 0, azimuths[0]))
        self.datasets.append(self.lighting_azimuth_dataset(domain_2_images, domain_2_labels, environments[1], 1, azimuths[1]))
        self.datasets.append(self.lighting_azimuth_dataset(domain_3_images, domain_3_labels, environments[2], 2, azimuths[2]))

        # Test environment
        original_dataset_te = SmallNORB(root, train=False, download=download)

        original_images = original_dataset_te.data
        original_labels = original_dataset_te.targets
        original_azimuths = original_dataset_te.azimuths

        domain_4_indices = []

        for i in range(len(original_images)):
            if original_azimuths[i] >= 12:
                domain_4_indices.append(i)
        
        domain_4_images = torch.index_select(original_images, 0, torch.LongTensor(domain_4_indices))

        domain_4_labels = torch.index_select(original_labels, 0, torch.LongTensor(domain_4_indices))

        # domain_4_azimuths = torch.index_select(original_azimuths, 0, torch.LongTensor(domain_4_indices))

        self.datasets.append(self.lighting_azimuth_dataset(domain_4_images, domain_4_labels, environments[2], 2, azimuths[2]))

        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 5

    def lighting_azimuth_dataset(self, images, labels, environment, env_id, azimuth):

        images = images.reshape((-1, 96, 96))[:, ::2, ::2]
        
        labels = self.add_noise(labels, 0.05).float()

        images, labels, lightings = self.lighting_dataset(images, labels, environment)
        
        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        azimuths = torch.full((y.shape[0], ), env_id, dtype=torch.float32)
        a = torch.stack((lightings, azimuths), 1)

        return TensorDataset(x, y, a)
    
    def lighting_dataset(self, images, labels, environment):

        lightings = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))

        images = torch.stack([images, images, images, images, images], dim=1)

        images[torch.tensor(range(len(images))), ((1 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((2 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((3 + lightings) % 5).long(), :, :] *= 0
        images[torch.tensor(range(len(images))), ((4 + lightings) % 5).long(), :, :] *= 0

        return images, labels, lightings
    
    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        c = (a + b) % 5
        return c

    def add_noise(self, labels: List, rate: float = 0.05):

        n_changes = int(len(labels) * rate)

        indices_to_change = random.sample(range(len(labels)), n_changes)

        for index in indices_to_change:
            labels[index] = random.choice([label for label in range(5) if label != labels[index]])

        return labels