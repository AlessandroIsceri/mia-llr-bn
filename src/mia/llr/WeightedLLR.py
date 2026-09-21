from src.mia.llr.NodewiseLLR import NodewiseLLR

class WeightedLLR(NodewiseLLR):

    def __init__(self, bn_theta_r, bn_theta_t, weighting_strategy):
        super().__init__(bn_theta_r, bn_theta_t)
        self.weighting_strategy = weighting_strategy

    def _compute_weight(self, xi, parents_evidence):
        return self.weighting_strategy.get_weight(xi, parents_evidence)