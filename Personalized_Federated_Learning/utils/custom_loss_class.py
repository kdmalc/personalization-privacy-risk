import torch

class CPHSLoss(torch.nn.modules.loss._Loss):
    def __init__(self, F, D, V, learning_batch, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False, verbose=True) -> None:
        super().__init__()
        self.F = F
        self.D = D
        self.V = V
        self.learning_batch = learning_batch
        self.lambdaF = lambdaF
        self.lambdaD = lambdaD
        self.lambdaE = lambdaE
        self.Nd = Nd
        self.Ne = Ne
        self.return_cost_func_comps = return_cost_func_comps
        # Don't use return_cost_func_comps since I don't think loss.item() will return a tuple, it only returns scalaras AFAIK
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        Nt = self.learning_batch
        self.D = self.D.view(self.Nd, self.Ne)
        Vplus = self.V[:,1:]
        # Performance
        term1 = self.lambdaE*(torch.linalg.matrix_norm((torch.matmul(self.D, self.F) - Vplus))**2)
        # D Norm
        term2 = self.lambdaD*(torch.linalg.matrix_norm((self.D)**2))
        # F Norm
        term3 = self.lambdaF*(torch.linalg.matrix_norm((self.F)**2))
        
        if verbose:
            print(f"LambdaE*Error_Norm^2: {term1}")
            print(f"LambdaD*Decoder_Norm^2: {term2}")
            print(f"LambdaF*EMG_Norm^2: {term3}")
        
        if self.return_cost_func_comps:
            return (term1 + term2 + term3), term1, term2, term3
        else:
            return (term1 + term2 + term3)