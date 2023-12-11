import torch
from torch.autograd import Variable

# Define the Elastic Weight Consolidation class
class EWC(object):
    def __init__(self, model, dataloader, fisher_multiplier):
        self.model = model
        self.dataloader = dataloader
        self.fisher_multiplier = fisher_multiplier
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.original_params = {n: p.clone().detach() for n, p in self.params.items()}
        self.fisher_information = self.calculate_fisher_information()

    def calculate_fisher_information(self):
        fisher_info = {n: torch.zeros_like(p) for n, p in self.params.items()}

        self.model.eval()
        for input_data, _ in self.dataloader:
            input_data = Variable(input_data, requires_grad=False)
            output = self.model(input_data)

            # Calculate gradients for each parameter
            self.model.zero_grad()
            output.sum().backward(create_graph=True)

            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                fisher_info[n] += (p.grad.detach() ** 2)

        fisher_info = {n: info / len(self.dataloader) for n, info in fisher_info.items()}
        return fisher_info

    def penalty(self):
        penalty = 0.0
        for n, p in self.params.items():
            fisher_info = self.fisher_information[n]
            original_params = self.original_params[n]
            penalty += (fisher_info * (p - original_params) ** 2).sum()

        return 0.5 * self.fisher_multiplier * penalty