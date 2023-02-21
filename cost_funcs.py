import numpy as np


def hessian_cost(F, alphaD, alphaE=1e-6):
    return ((2*alphaE*(F * F.T).T  # Idk if HV-V+ drops, I think it would stay.  Online calc says it drops lol
        + 2*alphaD ).flatten())


# set up gradient of cost:
# d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*alphaD*D
def gradient_cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, alphaE=1e-6, Nd=2, Ne=64):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder # TODO: we now have a timeseries component - consult Sam
    H: 2 x 2 state transition matrix
    
    alphaE is 1e-6 for all conditions
    ''' 
    
    Nt = learning_batch

    D = np.reshape(D,(Nd, Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    return ((2*(D@F + H@Vminus - Vplus)@F.T*(alphaE) #/ (Nd*Nt)
        + 2*alphaD*D ).flatten())  #/ (Nd*Ne)


# set up the cost function: 
# c_L2 = (||DF + HV - V+||_2)^2 + alphaD*(||D||_2)^2 + alphaF*(||F||_2)^2
def cost_l2(F, D, H, V, learning_batch, alphaF, alphaD, alphaE=1e-6, Nd=2, Ne=64):
    '''
    F: 64 channels x time EMG signals
    V: 2 x time target velocity
    D: 2 (x y vel) x 64 channels decoder
    H: 2 x 2 state transition matrix
    
    alphaE is 1e-6 for all conditions
    ''' 

    Nt = learning_batch
    D = np.reshape(D,(Nd,Ne))
    Vplus = V[:,1:]
    Vminus = V[:,:-1]

    #e = ( np.sum((D@F + H@Vminus - Vplus)**2)*(alphaE) #/ (Nd*Nt) 
    #        + alphaD*np.sum(D**2) #/ (Nd*Ne)
    #        + alphaF*np.sum(F**2) ) #/ (Ne*Nt) )
    
    #term1 = np.sum((D@F + H@Vminus - Vplus)**2)*(alphaE)
    #term2 = alphaD*np.sum(D**2) #/ (Nd*Ne)
    #term3 = alphaF*np.sum(F**2) #/ (Ne*Nt) )
    term1 = (np.linalg.norm((D@F + H@Vminus - Vplus))**2)*(alphaE)
    term2 = alphaD*(np.linalg.norm(D)**2)
    term3 = alphaF*(np.linalg.norm(F)**2)
    
    return (term1 + term2 + term3)


def estimate_decoder(F, H, V):
    return (V[:,1:]-H@V[:,:-1])@np.linalg.pinv(F)