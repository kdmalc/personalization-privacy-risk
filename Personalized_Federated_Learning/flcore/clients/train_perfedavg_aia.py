# This is clearly wrong... 
## Multiple params it doesn't even use...

import torch
import copy

def train_hf(self, epochs, delta=0.001):
    LOSS = 0
    self.model.train()

    for epoch in range(1, self.local_epochs + 1):  # local update 
        self.model.train()

        # Step 1: HF-MAML Update
        temp_model = copy.deepcopy(list(self.model.parameters()))

        # Calculate the Hessian-vector product approximation
        hf_approximation = []
        for param in self.model.parameters():
            param.requires_grad = True

        X, y = self.get_next_train_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        grad_params = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        for grad_p in grad_params:
            hf_approximation.append(grad_p)

        # Update the model parameters using the HF-MAML approximation
        for model_grad, hf_grad in zip(self.model.parameters(), hf_approximation):
            model_grad.data -= self.beta * hf_grad

        # Clone model to user model 
        self.clone_model_paramenter(self.model.parameters(), self.local_model)

    return LOSS