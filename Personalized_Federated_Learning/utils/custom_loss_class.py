import torch

class CPHSLoss(torch.nn.modules.loss._Loss):
    # Default lambdas get overwritten in clientbase.py
    def __init__(self, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False, verbose=False, dt=1/60, normalize_V=False) -> None:
        super().__init__()
        self.lambdaF = lambdaF
        self.lambdaD = lambdaD
        self.lambdaE = lambdaE
        self.Nd = Nd
        self.Ne = Ne
        self.return_cost_func_comps = return_cost_func_comps
        # Don't use return_cost_func_comps since I don't think loss.item() will return a tuple, it only returns scalars AFAIK
        self.dt = dt
        self.normalize_V = normalize_V
        self.verbose = verbose

    def forward(self, outputs, targets):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        # Numerical integration of v_actual to get p_actual
        p_actual = torch.cumsum(outputs, dim=1)*self.dt
        V = (p_reference - p_actual)*self.dt
        Vplus = V[:,1:]

        # Uhh why can't I just return the t2 and t3 terms?
        ## Because it only passes back 1 term? Or I don't know how the gradients propagate when I pass multiple back? ...
        ## Could probably just try it ...
        ## Could also just have a second, specific getter function...

        # Performance
        return self.lambdaE*(torch.linalg.matrix_norm(outputs[:,:-1] - Vplus)**2)


class CPHSLoss_WithReg(torch.nn.modules.loss._Loss):
    def __init__(self, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False, verbose=False, dt=1/60) -> None:
        super().__init__()
        self.lambdaF = lambdaF
        self.lambdaD = lambdaD
        self.lambdaE = lambdaE
        self.Nd = Nd
        self.Ne = Ne
        self.return_cost_func_comps = return_cost_func_comps
        self.dt = dt
        self.verbose = verbose

    def forward(self, outputs, targets, model):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        p_actual = torch.cumsum(outputs, dim=1) * self.dt
        V = (p_reference - p_actual) * self.dt
        Vplus = V[:, 1:]

        # Performance loss
        performance_loss = self.lambdaE * (torch.linalg.matrix_norm(outputs[:, :-1] - Vplus) ** 2)

        # L2 regularization
        l2_reg_loss = 0
        for param in model.parameters():
            l2_reg_loss += torch.norm(param, p=2)
        l2_reg_loss *= self.lambdaD

        # Total loss
        total_loss = performance_loss + l2_reg_loss

        if self.return_cost_func_comps:
            return total_loss, performance_loss, l2_reg_loss
        else:
            return total_loss