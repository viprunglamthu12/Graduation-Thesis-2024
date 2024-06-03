from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset

import pandas as pd
import os
import torch


class DWCivilCommentsDataset(CivilCommentsDataset):

    def __init__(self, version=None, root_dir='data', download=True, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'all_data_with_identities.csv'),
            index_col=0)
        
        self._metadata_df = self._metadata_df.dropna(subset=['comment_text'])

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['toxicity'].values >= 0.5)
        self._y_size = 1
        self._n_classes = 2

        # Extract text
        self._text_array = list(self._metadata_df['comment_text'])

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        # metadata_df contains split names in strings, so convert them to ints
        for split in self.split_dict:
            split_indices = self._metadata_df['split'] == split
            self._metadata_df.loc[split_indices, 'split'] = self.split_dict[split]
        self._split_array = self._metadata_df['split'].values

        # Extract metadata
        self._identity_vars = [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white'
        ]
        self._auxiliary_vars = [
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'
        ]

        self._metadata_array = torch.cat(
            (
                torch.LongTensor((self._metadata_df.loc[:, self._identity_vars] >= 0.5).values),
                # torch.LongTensor((self._metadata_df.loc[:, self._auxiliary_vars] >= 0.5).values),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )

        # self._metadata_fields = self._identity_vars + self._auxiliary_vars + ['y']
        self._metadata_fields = self._identity_vars + ['y']

        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars]

        WILDSDataset.__init__(self, root_dir, download, split_scheme)