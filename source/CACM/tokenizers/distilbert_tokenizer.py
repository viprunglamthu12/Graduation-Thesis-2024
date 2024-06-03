import torch
from transformers import DistilBertTokenizerFast

class MyDistilBertTokenizer(DistilBertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if 'padding' not in kwargs:
            kwargs['padding'] = 'max_length'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 300
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'

        tokens = super().__call__(*args, **kwargs)

        x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)

        return x