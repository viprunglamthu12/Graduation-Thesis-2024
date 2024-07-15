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
categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]

def parse_timestamp(ts):
    return tuple(map(int, ts.split('T')[0].split('-')))

def is_after_year(ts, year):
    ts_year, ts_month, ts_day = parse_timestamp(ts)
    return ts_year >= year

class FMoWDataset(WILDSDataset):
    _dataset_name = 'fmow'
    _versions_dict = {
        '1.2': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/',
            'compressed_size': 53_893_324_800}
    }
    def __init__(self, version=None, root_dir='data', download=False, seed=111, use_ood_val=True):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self._split_dict = {'train': 0, 'id_val': 1, 'id_test': 2, 'val': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'id_val': 'ID Val', 'id_test': 'ID Test', 'val': 'OOD Val', 'test': 'OOD Test'}
        self._source_domain_splits = [0, 1, 2]

        self.oracle_training_set = False
        split_scheme = 'time_after_2016'
        self._split_scheme = split_scheme

        self.root = Path(self._data_dir)
        self.seed = int(seed)
        self._original_resolution = (224, 224)

        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}

        self.metadata = pd.read_csv(self.root / 'rgb_metadata.csv')
        country_codes_df = pd.read_csv(self.root / 'country_code_mapping.csv')
        countrycode_to_region = {k: v for k, v in zip(country_codes_df['alpha-3'], country_codes_df['region'])}
        regions = [countrycode_to_region.get(code, 'Other') for code in self.metadata['country_code'].to_list()]
        self.metadata['region'] = regions
        all_countries = self.metadata['country_code']

        self.num_chunks = 101
        self.chunk_size = len(self.metadata) // (self.num_chunks - 1)

        if self._split_scheme.startswith('time_after'):
            year = int(self._split_scheme.split('_')[2])
            #year_dt = datetime.datetime(year, 1, 1, tzinfo=pytz.UTC)
            self.test_ood_mask = np.asarray([is_after_year(ts, year) for ts in self.metadata['timestamp']])
            # use 3 years of the training set as validation
            #year_minus_3_dt = datetime.datetime(year-3, 1, 1, tzinfo=pytz.UTC)
            self.val_ood_mask = np.asarray([is_after_year(ts, year-3) for ts in self.metadata['timestamp']]) & ~self.test_ood_mask
            self.ood_mask = self.test_ood_mask | self.val_ood_mask
        else:
            raise ValueError(f"Not supported: self._split_scheme = {self._split_scheme}")
        
        self._split_array = -1 * np.ones(len(self.metadata))
        for split in self._split_dict.keys():
            idxs = np.arange(len(self.metadata))
            if split == 'test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[self.test_ood_mask & test_mask]
            elif split == 'val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[self.val_ood_mask & val_mask]
            elif split == 'id_test':
                test_mask = np.asarray(self.metadata['split'] == 'test')
                idxs = idxs[~self.ood_mask & test_mask]
            elif split == 'id_val':
                val_mask = np.asarray(self.metadata['split'] == 'val')
                idxs = idxs[~self.ood_mask & val_mask]
            else:
                split_mask = np.asarray(self.metadata['split'] == split)
                idxs = idxs[~self.ood_mask & split_mask]

            self._split_array[idxs] = self._split_dict[split]

        if not use_ood_val:
            self._split_dict = {'train': 0, 'val': 1, 'id_test': 2, 'ood_val': 3, 'test': 4}
            self._split_names = {'train': 'Train', 'val': 'ID Val', 'id_test': 'ID Test', 'ood_val': 'OOD Val', 'test': 'OOD Test'}

        # filter out sequestered images from full dataset
        seq_mask = np.asarray(self.metadata['split'] == 'seq')
        # take out the sequestered images
        self._split_array = self._split_array[~seq_mask]
        self.full_idxs = np.arange(len(self.metadata))[~seq_mask]

        self._y_array = np.asarray([self.category_to_idx[y] for y in list(self.metadata['category'])])
        self.metadata['y'] = self._y_array
        self._y_array = torch.from_numpy(self._y_array).long()[~seq_mask]
        self._y_size = 1
        self._n_classes = 62

        # convert region to idxs
        all_regions = list(self.metadata['region'].unique())
        region_to_region_idx = {region: i for i, region in enumerate(all_regions)}
        self._metadata_map = {'region': all_regions}
        region_idxs = [region_to_region_idx[region] for region in self.metadata['region'].tolist()]
        self.metadata['region'] = region_idxs

        # make a year column in metadata
        year_array = -1 * np.ones(len(self.metadata))

        def parse_timestamp_year(ts):
            return int(ts.split('-')[0])

        for i, ts in enumerate(self.metadata['timestamp']):
            ts_year = parse_timestamp_year(ts)
            if 2002 <= ts_year < 2018:
                year_array[i] = ts_year - 2002

        self.metadata['year'] = year_array

        self._metadata_map['year'] = list(range(2002, 2018))
        self._metadata_fields = ['region', 'year', 'y']
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].astype(int).to_numpy()).long()[~seq_mask]

        self._eval_groupers = {
            'year': CombinatorialGrouper(dataset=self, groupby_fields=['year']),
            'region': CombinatorialGrouper(dataset=self, groupby_fields=['region']),
        }
        
        self.input_shape = (3, 224, 224,)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        # Overall evaluation + evaluate by year
        all_results, all_results_str = self.standard_group_eval(
            metric,
            self._eval_groupers['year'],
            y_pred, y_true, metadata)
        # Evaluate by region and ignore the "Other" region
        region_grouper = self._eval_groupers['region']
        region_results = metric.compute_group_wise(
            y_pred,
            y_true,
            region_grouper.metadata_to_group(metadata),
            region_grouper.n_groups)
        all_results[f'{metric.name}_worst_year'] = all_results.pop(metric.worst_group_metric_field)
        region_metric_list = []
        for group_idx in range(region_grouper.n_groups):
            group_str = region_grouper.group_field_str(group_idx)
            group_metric = region_results[metric.group_metric_field(group_idx)]
            group_counts = region_results[metric.group_count_field(group_idx)]
            all_results[f'{metric.name}_{group_str}'] = group_metric
            all_results[f'count_{group_str}'] = group_counts
            if region_results[metric.group_count_field(group_idx)] == 0 or "Other" in group_str:
                continue
            all_results_str += (
                f'  {region_grouper.group_str(group_idx)}  '
                f"[n = {region_results[metric.group_count_field(group_idx)]:6.0f}]:\t"
                f"{metric.name} = {region_results[metric.group_metric_field(group_idx)]:5.3f}\n")
            region_metric_list.append(region_results[metric.group_metric_field(group_idx)])
        all_results[f'{metric.name}_worst_region'] = metric.worst(region_metric_list)
        all_results_str += f"Worst-group {metric.name}: {all_results[f'{metric.name}_worst_region']:.3f}\n"

        return all_results, all_results_str
    

    

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            split_mask,
            transform=None):
        self.name = split_mask
        split_dict = wilds_dataset.split_dict
        mask = split_dict.get(split_mask)

        split_array = wilds_dataset.split_array
        subset_indices = np.where(split_array == mask)[0]
        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform
        self.metadata_array = wilds_dataset.metadata_array
        self.metadata = wilds_dataset.metadata
        self.y_array = wilds_dataset.y_array[subset_indices]

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        a = self.metadata_array[self.indices[i]][:2] * 1.0
        return x, y, a

    def __len__(self):
        return len(self.indices)
    
    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)

class WILDSDatasets(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.datasets = []
        for i, split_mask in enumerate(
                self.metadata_values(dataset)):
            env_transform = transform

            env_dataset = WILDSEnvironment(dataset, split_mask, env_transform)
            self.datasets.append(env_dataset)
        
        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes
        
        
        self.metadata_fields = ['region', 'year', 'y']
        self.metadata_map = dataset.metadata_map
        self.metadata_array = dataset.metadata_array


    def metadata_values(self, wilds_dataset):
        return wilds_dataset.split_dict
    
class WILDSFMoW(WILDSDatasets):
    ENVIRONMENTS = ["2012-2013", "2013-2016", "2016-2018", "2002-2013", "2002-2013"]
    def __init__(self, root, download):
        dataset = FMoWDataset(root_dir=root, download=download)
        super().__init__(dataset)