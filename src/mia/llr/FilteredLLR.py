from src.mia.llr.NodewiseLLR import NodewiseLLR
import numpy as np

class FilteredLLR(NodewiseLLR):

    def __init__(self, bn_theta_r, bn_theta_t, inference_engine_theta_t, size_T_estimate):
        super().__init__(bn_theta_r, bn_theta_t)
        self.inference_engine_theta_t = inference_engine_theta_t
        self.size_T_estimate = size_T_estimate
        self.penalty_value = size_T_estimate
        
        # Precompute cardinality of each node
        self._cardinalities = {
            node: bn_theta_t.get_cpds(node).variable_card
            for node in bn_theta_t.nodes()
        }

    def _is_filtered(self, parents_evidence):
        
        # Do not filter root nodes (nodes without parents)
        if not parents_evidence:
            return False
        
        # Compute probability of observing the parent configuration
        self.inference_engine_theta_t.eraseAllEvidence()
        self.inference_engine_theta_t.setEvidence(parents_evidence)
        p_pi = self.inference_engine_theta_t.evidenceProbability()
        
        # Normalize by the uniform probability over all parent configurations
        n_parent_configs = np.prod([self._cardinalities[p] for p in parents_evidence])
        
        if p_pi * n_parent_configs * self.size_T_estimate <= 2:
            return True
        return False
    
    def _get_filter_penalty(self):
        self.penalty_value += 1
        return self.penalty_value