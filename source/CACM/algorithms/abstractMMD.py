import torch
from torch.nn import functional as F
from CACM.algorithms.base_algorithm import PredictionAlgorithm
from CACM.algorithms.utils import mmn_compute

class AbstractMMD(PredictionAlgorithm):
    def __init__(self, model,  sequence_classification=False, n_groups_per_batch=None, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 gaussian=True, gamma=1e-6, mmd_lambda=1):
        super(AbstractMMD, self).__init__(model, optimizer, lr, weight_decay, betas, momentum)

        if sequence_classification:
            if n_groups_per_batch == None:
                raise Exception('n_groups_per_batch must be specified if sequence_classification=True')

        if gaussian:
            self.kernel_type = "gaussian"
        else: 
            self.kernel_type = "mean_cov"

        self.gamma = gamma
        self.mmd_lambda = mmd_lambda
        self.sequence_classification = sequence_classification
        self.n_groups_per_batch = n_groups_per_batch
    def mmd(self, x, y):
        return mmn_compute(x, y, self.kernel_type, self.gamma)

    def training_step(self, train_batch, batch_idx):
        objective = 0
        penalty = 0
        correct, total = 0, 0
        

        if not self.sequence_classification:
            minibatches = train_batch 
            nmb = len(minibatches)
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

            nmb = self.n_groups_per_batch
            batch_size = len(train_batch[1])
            mb_size = int(batch_size/nmb)
            
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            features = []
            targets = []
            attributes = []
            classifs = []
            # classifs = torch.empty(0, device=device, dtype=train_batch[1].dtype)

            for i in range(0, batch_size, mb_size):
                mb_features = torch.empty((0, 300, 2), device=device, dtype=torch.long)
                mb_targets = torch.empty(0, device=device, dtype=train_batch[1].dtype)
                mb_attributes = torch.empty((0, 11), device=device, dtype=torch.long)
                for j in range(i, i+mb_size):
                    mb_features = torch.cat((mb_features, train_batch[0][j].unsqueeze(0)), dim=0)
                    mb_targets = torch.cat((mb_targets, train_batch[1][j].unsqueeze(0)), dim=0)
                    mb_attributes = torch.cat((mb_attributes, train_batch[2][j].unsqueeze(0)), dim=0)
                features.append(mb_features)
                targets.append(mb_targets)
                attributes.append(mb_attributes)
                # print('self.classif')
                # print(self.classifier(mb_features))
                classifs.append(self.classifier(mb_features))
                # classifs = torch.cat((classifs, self.classifier(mb_features)), dim=0)
            for i in range(nmb):
                objective += F.cross_entropy(classifs[i], targets[i])
                correct += (torch.argmax(classifs[i], dim=1) == targets[i]).float().sum().item()
                total += classifs[i].shape[0]
                for j in range(i+1, nmb):
                    penalty += self.mmd(features[i]*1.0, features[j]*1.0)
            objective /= nmb

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)
            loss = objective + self.mmd_lambda * penalty
        acc = correct / total
        metrics =  {"train_acc": acc, "train_loss": loss, "penalty": penalty}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss        
