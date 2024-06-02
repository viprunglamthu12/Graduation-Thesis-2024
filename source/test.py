import torch
import pytorch_lightning as pl
from CACM.datasets.MNIST import MNISTIndAttribute
from CACM.dataloaders.get_data_loader import get_loaders
from CACM.models.networks import MNIST_MLP, Classifier, MLP
from CACM.algorithms.CACM import CACM
from CACM.algorithms.CDANN import CDANN
data_dir = 'data'
dataset = MNISTIndAttribute(data_dir, download=True)

loaders = get_loaders(dataset, train_envs=[0, 1], batch_size=64, val_envs=[2], test_envs=[3])

featurizer = MNIST_MLP(dataset.input_shape)
classifier = Classifier(
    featurizer.n_outputs,
    dataset.num_classes)

model = torch.nn.Sequential(featurizer, classifier)

#algorithm = CACM(model, lr=1e-3, gamma=1e-2, attr_types=['causal'], lambda_causal=100.)
algorithm = CDANN(model, lr=1e-3)
trainer = pl.Trainer(devices = 1, max_epochs=1)

trainer.fit(algorithm, loaders['train_loaders'], loaders['val_loaders'])

if 'test_loaders' in loaders:
    a = trainer.test(dataloaders=loaders['test_loaders'], ckpt_path='best')