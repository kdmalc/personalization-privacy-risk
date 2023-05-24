# PFLNIID

#from opacus import PrivacyEngine
# Yah I just didn't wanna bother installing another package
# I'm not using DP right now anyway so maybe I'll come back

#MAX_GRAD_NORM = 1.0
#DELTA = 1e-5

#def initialize_dp(model, optimizer, data_loader, dp_sigma):
#    privacy_engine = PrivacyEngine()
#    model, optimizer, data_loader = privacy_engine.make_private(
#        module=model,
#        optimizer=optimizer,
#        data_loader=data_loader,
#        noise_multiplier = dp_sigma, 
#        max_grad_norm = MAX_GRAD_NORM,
#    )
#
#    return model, optimizer, data_loader, privacy_engine


#def get_dp_params(privacy_engine):
#    return privacy_engine.get_epsilon(delta=DELTA), DELTA