import torch
#from math import isnan

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
        self.term1_error = 0
        self.term2_ld_decnorm = 0
        self.term3_lf_emgnorm = 0


    def forward(self, outputs, targets):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        # Numerical integration of v_actual to get p_actual
        p_actual = torch.cumsum(outputs, dim=1)*self.dt
        self.V = (p_reference - p_actual)*self.dt
        Vplus = self.V[:,1:]
        # Performance
        return self.lambdaE*(torch.linalg.matrix_norm(outputs[:,:-1] - Vplus)**2)
      
        
    def update_FDV(self, F, D, V):
        print("update_FDV")
        self.F = F  # This isn't used currently
        self.D = D  # Why was I using detach().clone() here on D?
        self.V = V  
        self.learning_batch = self.F.shape[0] # Idk if things need to be transposed or what... maybe it should be [1]
       