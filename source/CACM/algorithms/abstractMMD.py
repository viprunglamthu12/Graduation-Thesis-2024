import torch
from torch.nn import functional as F
from CACM.algorithms.base_algorithm import PredictionAlgorithm
from CACM.algorithms.utils import mmn_compute

class AbstractMMD(PredictionAlgorithm):
    def __init__(self, model,  sequence_classification=False, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 gaussian=True, gamma=1e-6, mmd_lambda=1):
        super(AbstractMMD, self).__init__(model, optimizer, lr, weight_decay, betas, momentum)
        if gaussian:
            self.kernel_type = "gaussian"
        else: 
            self.kernel_type = "mean_cov"

        self.gamma = gamma
        self.mmd_lambda = mmd_lambda
        self.sequence_classification = sequence_classification

    def mmd(self, x, y):
        return mmn_compute(x, y, self.kernel_type, self.gamma)

    def training_step(self, train_batch, batch_idx):
        objective = 0
        penalty = 0
        correct, total = 0, 0
        minibatches = train_batch 
        nmb = len(minibatches)

        if not self.sequence_classification:
            self.featurizer = self.model[0]
            self.classifier = self.model[1]



            features = [self.featurizer(xi) for xi, _, _ in minibatches]
            classifs = [self.classifier(fi) for fi in features]

            targets = [yi for _, yi, _ in minibatches]

            for i in range(nmb):
                objective += F.cross_entropy(classifs[i], targets[i])
                correct += (torch.argmax(classifs[i], dim=1)==targets[i]).float().sum().item()
                total += classifs[i].shape[0]
                for j in range(i+1, nmb):
                    penalty += self.mmd(features[i], features[j])
            
            objective /= nmb


        else:
            self.classifier = self.model

            features = train_batch[0]
            targets = train_batch[1]

            classifs = self.classifier(features)
            for i in range(nmb):
                objective += F.cross_entropy(classifs, targets)
                correct += (torch.argmax(classifs, dim=1)==targets).float().sum().item()
                total += classifs.shape[0]
                for j in range(i+1, nmb):
                    penalty += self.mmd(features[i], features[j])
            
            objective /= nmb

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)
            loss = objective + self.mmd_lambda * penalty
            
        acc = correct / total

        metrics =  {"train_acc": acc, "train_loss": loss, "penalty": penalty}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss        
