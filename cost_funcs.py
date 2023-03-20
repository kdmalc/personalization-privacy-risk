import numpy as np

# np.kron(a, b) is the kronecker product
# The function assumes that the number of dimensions of a and b are the same, if necessary prepending the smallest with ones. If a.shape = (r0,r1,..,rN) and b.shape = (s0,s1,...,sN), the Kronecker product has shape (r0*s0, r1*s1, ..., rN*SN) --> Thus for 2D: output is (r0*s0, r1*s1) --> In our case, must have square output since F@F.T is square and identity is square
def hessian_cost_l2(F, alphaD, alphaE=1e-6):
    # Not sure about shape indices... should they all be 0? Or m, n, m like it is now?  
    # Dims work out in both cases...
    return 2*alphaE*np.kron((F@F.T), np.identity(F.shape[1])) + 2*alphaD*np.kron(np.identity(F.shape[0]), np.identity(F.shape[1]))


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
    return ((2*(D@F + H@Vminus - Vplus)@F.T*(alphaE) + 2*alphaD*D ).flatten())


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
    term1 = alphaE*(np.linalg.norm((D@F + H@Vminus - Vplus))**2)
    term2 = alphaD*(np.linalg.norm(D)**2)
    term3 = alphaF*(np.linalg.norm(F)**2)
    return (term1 + term2 + term3)


def estimate_decoder(F, H, V):
    return (V[:,1:]-H@V[:,:-1])@np.linalg.pinv(F)