from src.mia.weighting.WeightingStrategy import WeightingStrategy
import numpy as np

class EntropyWeightingStrategy(WeightingStrategy):
    
    _cache: dict = {} 

    OPERATIONS_MAP = {
        "1": lambda e: e,
        "/": lambda e: 1 / (e + 1e-10)
    }

    def __init__(self, bn_theta_t, operation, normalize):
        self.normalize = normalize
        self.operation = operation
        self.weight_function = self.OPERATIONS_MAP[operation]
        
        # Precompute factors for each node to speed up entropy evaluation
        self._factors = {
            node: bn_theta_t.get_cpds(node).to_factor()
            for node in bn_theta_t.nodes()
        }
        
        # Precompute cardinalities for each node to speed up entropy evaluation
        self._cardinalities = {
            node: len(factor.state_names[node])
            for node, factor in self._factors.items()
        }

    def get_weight(self, xi, parents_evidence):
        xi_name = list(xi.keys())[0]

        # Create a unique cache key
        cache_key = (xi_name, frozenset(parents_evidence.items()))
        if cache_key in EntropyWeightingStrategy._cache:
            # Retrieve previously computed entropy and variable cardinality
            entropy, cardinality = EntropyWeightingStrategy._cache[cache_key]
        else:
            # Get the conditional probability distribution given the parent evidence
            factor = self._factors[xi_name]
            cardinality = self._cardinalities[xi_name]

            instance = parents_evidence.copy()
            probs = np.array([
                factor.get_value(**{**instance, xi_name: v})
                for v in range(cardinality)
            ])

            # Compute Shannon entropy of the probability distribution
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            EntropyWeightingStrategy._cache[cache_key] = (entropy, cardinality)

        if self.normalize:
            entropy = entropy / np.log(cardinality)

        return self.weight_function(entropy)
    
    def get_operation(self):
        return self.operation
    
    def get_normalize(self):
        return self.normalize