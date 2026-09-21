from src.mia.weighting.WeightingStrategy import WeightingStrategy
import numpy as np

class ParentsProbabilityWeightingStrategy(WeightingStrategy):

    OPERATIONS_MAP = {
        "1": lambda p: p,
        "/": lambda p: 1 / (p + 1e-10),
        "-": lambda p: 2 - p,
        "-log": lambda p: -np.log(p + 1e-10) + 1
    }

    def __init__(self, inference_engine_theta_t, operation):
        self.inference_engine_theta_t = inference_engine_theta_t
        
        self.operation = operation
        self.weight_function = self.OPERATIONS_MAP[operation]
        self.neutral_element = self.weight_function(
            1
        )

    def get_weight(self, _, parents_evidence):
        
        if len(parents_evidence) == 0:
            return self.neutral_element
        
        # Compute the probability of the parent configuration
        self.inference_engine_theta_t.eraseAllEvidence()
        self.inference_engine_theta_t.setEvidence(parents_evidence)
        p_pi = self.inference_engine_theta_t.evidenceProbability()
        return self.weight_function(p_pi)

    def get_operation(self):
        return self.operation