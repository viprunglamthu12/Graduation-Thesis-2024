from algorithms.abstractDANN import AbstractDANN

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 conditional=False, class_balance=False, d_steps_per_g=1, gradient_penalty=1e-2, DANN_lambda = 1e-1):
        super(DANN, self).__init__(model, optimizer="Adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), momentum=0.9, 
                 conditional=False, class_balance=False, d_steps_per_g=1, gradient_penalty=1e-2, DANN_lambda = 1e-1)
