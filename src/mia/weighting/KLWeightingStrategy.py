from src.mia.weighting.WeightingStrategy import WeightingStrategy
import numpy as np

class KLWeightingStrategy(WeightingStrategy):
    
    _cache: dict = {} 
     
    OPERATIONS_MAP = {
        "1": lambda kl: kl,
        "/": lambda kl: 1 / (kl + 1e-10),
        "exp": lambda kl: np.exp(-kl)
    }
    
    def __init__(self, bn_theta_r, bn_theta_t, operation):
        self.operation = operation
        self.weight_function = self.OPERATIONS_MAP[operation]
        
        # Precompute factors for each node to speed up KL evaluation
        self._factors_t = {
            node: bn_theta_t.get_cpds(node).to_factor()
            for node in bn_theta_t.nodes()
        }
        self._factors_r = {
            node: bn_theta_r.get_cpds(node).to_factor()
            for node in bn_theta_r.nodes()
        }
        
        # Precompute cardinalities for each node to speed up KL evaluation
        self._cardinalities = {
            node: len(factor.state_names[node])
            for node, factor in self._factors_t.items()
        }

    def get_weight(self, xi, parents_evidence):
        xi_name = list(xi.keys())[0]
        cache_key = (xi_name, tuple(sorted(parents_evidence.items())))
    
        if KLWeightingStrategy._cache is not None and cache_key in KLWeightingStrategy._cache:
            # Return the cached KL divergence after applying the selected transformation
            return self.weight_function(KLWeightingStrategy._cache[cache_key])

        cardinality = self._cardinalities[xi_name]
        factor_t = self._factors_t[xi_name]
        factor_r = self._factors_r[xi_name]

        # Get probability distributions for the given parent configuration
        probs_t = np.array([
            factor_t.get_value(**{**parents_evidence, xi_name: v})
            for v in range(cardinality)
        ])
        probs_r = np.array([
            factor_r.get_value(**{**parents_evidence, xi_name: v})
            for v in range(cardinality)
        ])

        epsilon = 1e-10
        
        # Compute the Kullback-Leibler divergence between the two distributions:
        distance = np.sum(probs_r * np.log((probs_r + epsilon) / (probs_t + epsilon)))
        KLWeightingStrategy._cache[cache_key] = distance
            
        return self.weight_function(distance)
    
    def get_operation(self):
        return self.operation