import torch

class CPHSLoss(torch.nn.modules.loss._Loss):
    def __init__(self, F, D, V, learning_batch, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False, verbose=False, dt=1/60, normalize_V=False) -> None:
        super().__init__()
        self.F = F
        self.D = D.detach().clone()
        self.V = V
        self.learning_batch = learning_batch
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
        
    def forward(self, outputs, targets, my_model):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        p_actual = torch.cumsum(outputs, dim=1)*self.dt  # Numerical integration of v_actual to get p_actual
        self.V = (p_reference - p_actual)*self.dt
        if self.normalize_V:
            self.V = self.V/torch.linalg.norm(self.V, ord='fro')
            assert (torch.linalg.norm(self.V, ord='fro')<1.2) and (torch.linalg.norm(self.V, ord='fro')>0.8)
        
        Nt = self.learning_batch
        self.D = my_model.weight.detach().clone()
        self.D = self.D.view(self.Nd, self.Ne)
        Vplus = self.V[:,1:]
        
        # Performance
        term1 = self.lambdaE*(torch.linalg.matrix_norm(outputs[:,:-1] - Vplus)**2)
        # D Norm
        term2 = self.lambdaD*(torch.linalg.matrix_norm(my_model.weight)**2)
        # F Norm
        term3 = self.lambdaF
        #term3 = self.lambdaF*(torch.linalg.matrix_norm(inputs)**2)
        # I switched it so I can pass in the outputs instead of the inputs...
        # self.lambdaF presumably will be 0 for every trial

        if self.verbose:
            print(f"LambdaE*Error_Norm^2: {term1}")
            print(f"LambdaD*Decoder_Norm^2: {term2}")
            print(f"LambdaF*EMG_Norm^2: {term3}")
        
        if self.return_cost_func_comps:
            return (term1 + term2 + term3), term1, term2, term3
        else:
            return (term1 + term2 + term3)
        
    def update_FDV(F, D, V, learning_batch):
        self.F = F
        self.D = D.detach().clone()
        self.V = V
        self.learning_batch = learning_batch
        
    def calc_obj_loss(self) -> torch.Tensor:
        Nt = self.learning_batch
        self.D = self.D.view(self.Nd, self.Ne)
        Vplus = self.V[:,1:]
        # Performance
        term1 = self.lambdaE*(torch.linalg.matrix_norm((torch.matmul(self.D, self.F) - Vplus))**2)
        # D Norm
        term2 = self.lambdaD*(torch.linalg.matrix_norm(self.D)**2)
        # F Norm
        term3 = self.lambdaF#*(torch.linalg.matrix_norm(self.F)**2)
        
        if self.verbose:
            print(f"LambdaE*Error_Norm^2: {term1}")
            print(f"LambdaD*Decoder_Norm^2: {term2}")
            print(f"LambdaF*EMG_Norm^2: {term3}")
        
        if self.return_cost_func_comps:
            return (term1 + term2 + term3), term1, term2, term3
        else:
            return (term1 + term2 + term3)