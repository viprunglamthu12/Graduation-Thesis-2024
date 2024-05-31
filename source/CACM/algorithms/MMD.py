from algorithms.abstractMMD import AbstractMMD

class MMD(AbstractMMD):
    def __init__(self, model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 gaussian=True, gamma=1e-6, mmd_lambda=1):
        super(MMD, self).__init__(model, optimizer, lr, weight_decay, betas, momentum, gaussian=True)