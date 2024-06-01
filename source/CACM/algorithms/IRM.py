import torch
from torch.nn import functional as F
from CACM.algorithms.base_algorithm import PredictionAlgorithm
import torch.autograd as autograd


class IRM(PredictionAlgorithm):
    def __init__(self, model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9,
                  irm_lambda = 1e2, irm_anneal_iters = 1e3):
        super().__init__(model, optimizer, lr, weight_decay, betas, momentum)
        self.register_buffer("update_count", torch.tensor([0]))
        self.irm_lambda = irm_lambda
        self.irm_anneal_iters = irm_anneal_iters


    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale  = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def training_step(self, train_batch, batch_idx):
        device = "cuda" if train_batch[0][0].is_cuda else "cpu"
        penalty_weight = (self.irm_lambda if self.update_count >= self.irm_anneal_iters else 1.0)
        nll = 0.
        penalty = 0.
        
        x = torch.cat([x for x, y, _ in train_batch])
        

        all_logits = self.model(x)
        all_logits_idx = 0
        for i, (x, y, a) in enumerate(train_batch):
            logits = all_logits[all_logits_idx:all_logits_idx+x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(train_batch)
        penalty /= len(train_batch)
        loss = nll +(penalty_weight * penalty)

        self.update_count += 1

        y = torch.cat([y for x, y, _ in train_batch])
        acc = (torch.argmax(all_logits, dim=1) == y).float().mean()

        metrics = {"train_acc": acc, "train_loss": loss, "nll": nll.item(), "penalty": penalty.item()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    