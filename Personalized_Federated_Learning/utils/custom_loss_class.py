import torch
import copy

class CPHSLoss(torch.nn.modules.loss._Loss):
    # Default lambdas get overwritten in clientbase.py
    def __init__(self, use_hit_bound=False, lambdaF=0, lambdaD=1e-3, lambdaE=1e-6, Nd=2, Ne=64, return_cost_func_comps=False, verbose=False, normalize_V=False) -> None:
        super().__init__()
        self.use_hit_bound = use_hit_bound
        self.lambdaF = lambdaF
        self.lambdaD = lambdaD
        self.lambdaE = lambdaE
        self.Nd = Nd
        self.Ne = Ne
        self.return_cost_func_comps = return_cost_func_comps
        # Don't use return_cost_func_comps since I don't think loss.item() will return a tuple, it only returns scalars AFAIK
        self.normalize_V = normalize_V
        self.verbose = verbose
        # Fixed from data collection
        self.dt = 1/60
        self.gain = 120
        self.ALMOST_ZERO_TOL = 0.01

    def forward(self, outputs, targets, current_round):
        outputs = torch.transpose(outputs, 0, 1)
        p_reference = torch.transpose(targets, 0, 1)
        # Numerical integration of v_actual to get p_actual
        p_actual = torch.cumsum(outputs, dim=1)*self.dt

        # Add the boundary conditions code here
        if self.use_hit_bound:
            # Maneeshika code
            p_ref_lim = copy.deepcopy(targets)
            # TODO: Do these vars need to be self? ... No sure about how continuity works across updates and such...
            if current_round<2:
                self.vel_est = torch.zeros_like((p_ref_lim))
                self.pos_est = torch.zeros_like((p_ref_lim))
                self.int_vel_est = torch.zeros_like((p_ref_lim))
                #self.vel_est[0] = self.w@s[:,0]  # Translated from: Ds_fixed@emg_tr[0]
                self.vel_est[0] = outputs[:,0]  # TODO: Is the shape/orientation correct?
                self.pos_est[0] = [0, 0]
            else:
                prev_vel_est = self.vel_est[-1]
                prev_pos_est = self.pos_est[-1]
                
                self.vel_est = torch.zeros_like((p_ref_lim))
                self.pos_est = torch.zeros_like((p_ref_lim))
                self.int_vel_est = torch.zeros_like((p_ref_lim))
                
                self.vel_est[0] = prev_vel_est
                self.pos_est[0] = prev_pos_est
            #for tt in range(1, s.shape[1]):
            for tt in range(1, outputs.shape[1]):
                # Note this does not keep track of actual updates, only the range of 1 to s.shape[1] (1202ish)
                #vel_plus = self.w@s[:,tt]  # Translated from: Ds_fixed@emg_tr[tt]
                vel_plus = outputs[:, tt] # TODO: May need to be transposed or something...
                p_plus = self.pos_est[tt-1, :] + (self.vel_est[tt-1, :]*self.dt)
                # These are just correctives, such that vel_plus can get bounded
                # x-coordinate
                if abs(p_plus[0]) > 36:  # 36 hardcoded from earlier works
                    p_plus[0] = self.pos_est[tt-1, 0]
                    vel_plus[0] = 0
                    self.hit_bound += 1 # update hit_bound counter
                if abs(p_plus[1]) > 24:  # 24 hardcoded from earlier works
                    p_plus[1] = self.pos_est[tt-1, 1]
                    vel_plus[1] = 0
                    self.hit_bound += 1 # update hit_bound counter
                if self.hit_bound > 200:  # 200 hardcoded from earlier works
                    p_plus[0] = 0
                    vel_plus[0] = 0
                    p_plus[1] = 0
                    vel_plus[1] = 0
                    self.hit_bound = 0
                # now update velocity and position
                self.vel_est[tt] = vel_plus
                self.pos_est[tt] = p_plus
                # calculate intended velocity
                #self.int_vel_est[tt] = calculate_intended_vels(p_ref_lim[tt], p_plus, 1/self.dt)
                intended_vector = (p_ref_lim[tt] - p_plus)/(1/self.dt)
                if torch.norm(intended_vector) <= self.ALMOST_ZERO_TOL:
                    intended_norm = torch.zeros((2,))
                else:
                    intended_norm = intended_vector * self.gain
                self.int_vel_est[tt] = intended_norm

            V = self.int_vel_est[:tt+1].T
            #print(f"V.shape: {self.V.shape}")
        else:
            # My original code, doesn't take into account the position reset...
            V = (p_reference - p_actual)*self.dt

        Vplus = V[:,1:]
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