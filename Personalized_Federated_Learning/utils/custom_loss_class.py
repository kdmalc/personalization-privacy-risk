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
      
        
    #def update_FDV(self, F, D, V):
    #    print("update_FDV")
    #    self.F = F  # This isn't used currently
    #    self.D = D  # Why was I using detach().clone() here on D?
    #    self.V = V  
    #    self.learning_batch = self.F.shape[0] # Idk if things need to be transposed or what... maybe it should be [1]
       