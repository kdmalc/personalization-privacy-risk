import torch
from math import isnan

class CPHSLoss2(torch.nn.modules.loss._Loss):
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
        p_actual = torch.cumsum(outputs, dim=1)*self.dt  # Numerical integration of v_actual to get p_actual
        self.V = (p_reference - p_actual)*self.dt
        if self.normalize_V:
            self.V = self.V/torch.linalg.norm(self.V, ord='fro')
            assert (torch.linalg.norm(self.V, ord='fro')<1.2) and (torch.linalg.norm(self.V, ord='fro')>0.8)
        Vplus = self.V[:,1:]
        # Performance
        return self.lambdaE*(torch.linalg.matrix_norm(outputs[:,:-1] - Vplus)**2)

        
    def forward_prev_main(self, outputs, targets, my_model):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        p_actual = torch.cumsum(outputs, dim=1)*self.dt  # Numerical integration of v_actual to get p_actual
        self.V = (p_reference - p_actual)*self.dt
        if self.normalize_V:
            self.V = self.V/torch.linalg.norm(self.V, ord='fro')
            assert (torch.linalg.norm(self.V, ord='fro')<1.2) and (torch.linalg.norm(self.V, ord='fro')>0.8)
        
        #Nt = self.learning_batch
        # What is this doing...
        #self.D = my_model.weight.detach().clone()
        #self.D = self.D.view(self.Nd, self.Ne)
        #self.D = my_model.weight.view(self.Nd, self.Ne)
        Vplus = self.V[:,1:]
        
        # Performance
        self.term1_error = self.lambdaE*(torch.linalg.matrix_norm(outputs[:,:-1] - Vplus)**2)
        # D Norm
        self.term2_ld_decnorm = self.lambdaD*(torch.linalg.matrix_norm(my_model.weight)**2)
        # F Norm
        self.term3_lf_emgnorm = self.lambdaF*(torch.linalg.matrix_norm(self.F)**2)
        #term3 = self.lambdaF*(torch.linalg.matrix_norm(inputs)**2)
        # I switched it so I can pass in the outputs instead of the inputs...
        # self.lambdaF presumably will be 0 for every trial

        if self.verbose:
            print(f"ERROR: LambdaE*Error_Norm^2: {self.term1_error:0,.4f}")
            print(f"D: LambdaD*Decoder_Norm^2: {self.term2_ld_decnorm:0,.4f}")
            print(f"F: LambdaF*EMG_Norm^2: {self.term3_lf_emgnorm:0,.1f}")
        
        if isnan(self.term1_error) or isnan(self.term2_ld_decnorm) or isnan(self.term3_lf_emgnorm):
            print(f"Error term: {self.term1_error}")
            print(f"D term: {self.term2_ld_decnorm}")
            print(f"F term: {self.term3_lf_emgnorm}")
            raise ValueError("One of the cost function terms is NAN...")
        
        total_loss = self.term1_error + self.term2_ld_decnorm + self.term3_lf_emgnorm
        if self.return_cost_func_comps:
            return total_loss, self.term1_error, self.term2_ld_decnorm, self.term3_lf_emgnorm
        else:
            return total_loss
        
    def update_FDV(self, F, D, V, learning_batch):
        print("update_FDV")
        self.F = F
        # Why...
        self.D = D.detach().clone()
        self.V = V
        self.learning_batch = learning_batch
       