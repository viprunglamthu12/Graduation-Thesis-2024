import torch
import torch.autograd as autograd
from torch.nn import functional as F
from algorithms.base_algorithm import PredictionAlgorithm
from models.networks import MLP

class AbstractDANN(PredictionAlgorithm):
    def __init__(self, model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 conditional=False, class_balance=False, d_steps_per_g=1, gradient_penalty=1e-2, DANN_lambda = 1e-1):
        super(AbstractDANN, self).__init__(model, optimizer, lr, weight_decay, betas, momentum)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance
        self.d_steps_per_g = d_steps_per_g
        self.gradient_penalty = gradient_penalty
        self.DANN_lambda = DANN_lambda
        self.featurizer = self.model[0]
        self.classifier = self.model[1]
        # Obey default values in DomainBed
        self.discriminator = MLP(self.featurizer.n_outputs, 2, 256, 3, 0.0)

        self.class_embeddings = torch.nn.Embedding(self.classifier.out_features, self.featurizer.n_outputs)
    def training_step(self, train_batch, batch_idx):
        

        device = "cuda" if train_batch[0][0].is_cuda else "cpu"
        self.update_count += 1
        minibatches = train_batch

        x = torch.cat([x for x, y, _ in train_batch])
        y = torch.cat([y for x, y, _ in train_batch])
        z = self.featurizer(x)

        if self.conditional:
            disc_input = z + self.class_embeddings(y)
        else: 
            disc_input = z
        
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y, a) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(y).sum(dim=0)
            weights = 1. / (y_counts[y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)


        input_grad = autograd.grad(
            F.cross_entropy(disc_out, disc_labels, reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.gradient_penalty * grad_penalty

        if (self.update_count.item() % (1+self.d_steps_per_g)) < self.d_steps_per_g:
            metrics = {'disc_loss': disc_loss.item()}
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
            return disc_loss
        else:
            all_preds = self.classifier(z)
            classifier_loss = F.cross_entropy(all_preds, y)
            gen_loss = (classifier_loss + self.DANN_lambda * -disc_loss)
            metrics = {'gen_loss': disc_loss.item()}
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
            return gen_loss



    def configure_optimizers(self):
        if self.optimizer == "Adam":
            if (self.update_count.item() % (1+self.d_steps_per_g)) < self.d_steps_per_g:
                optimizer = torch.optim.Adam(
                    (list(self.discriminator.parameters()) +
                list(self.class_embeddings.parameters())), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas
                )
            else:
                optimizer = torch.optim.Adam(
                    (list(self.featurizer.parameters()) +
                list(self.classifier.parameters())), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas
                )

        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum
            )

        return optimizer
