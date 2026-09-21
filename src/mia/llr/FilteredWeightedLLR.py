from src.mia.llr.FilteredLLR import FilteredLLR

class FilteredWeightedLLR(FilteredLLR):
    
    def __init__(self, bn_theta_r, bn_theta_t, inference_engine_theta_t, size_T_estimate, weighting_strategy):   
        super().__init__(bn_theta_r, bn_theta_t, inference_engine_theta_t, size_T_estimate)
        self.weighting_strategy = weighting_strategy
            
    def _compute_weight(self, xi, parents_evidence):
        return self.weighting_strategy.get_weight(xi, parents_evidence)