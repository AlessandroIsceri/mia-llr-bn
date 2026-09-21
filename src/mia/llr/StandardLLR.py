from src.mia.llr.AbstractLLR import AbstractLLR
import numpy as np

class StandardLLR(AbstractLLR):
    
    def __init__(self, inference_engine_theta_r, inference_engine_theta_t):
        self.inference_engine_theta_r = inference_engine_theta_r
        self.inference_engine_theta_t = inference_engine_theta_t
    
    def _get_ll(self, instance: dict, inference_engine_theta):

        # Erase all evidences and apply addEvidence(key, value) for every pairs in instance
        inference_engine_theta.setEvidence(instance)

        # Compute P(instance | theta)
        ll = inference_engine_theta.evidenceProbability()
        if ll == 0:
            ll += 1e-10

        return np.log(ll)

    def _compute_llr(self, instance: dict):

        loglik_r = self._get_ll(instance, self.inference_engine_theta_r)
        loglik_t = self._get_ll(instance, self.inference_engine_theta_t)
        
        return loglik_r - loglik_t