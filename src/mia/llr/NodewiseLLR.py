from src.mia.llr.AbstractLLR import AbstractLLR
import numpy as np

class NodewiseLLR(AbstractLLR):

    def __init__(self, bn_theta_r, bn_theta_t):
        
        # Precompute factors for each node to speed up nodewise llr evaluation
        self._factors_r = {
            node: bn_theta_r.get_cpds(node).to_factor()
            for node in bn_theta_r.nodes()
        }
        self._factors_t = {
            node: bn_theta_t.get_cpds(node).to_factor()
            for node in bn_theta_t.nodes()
        }
        
        # Store the parent list of every node to speed up nodewise llr evaluation
        self._parents = {
            node: list(bn_theta_t.predecessors(node))
            for node in bn_theta_t.nodes()
        }
        self._cache = {}

    def _compute_weight(self, xi, parents_evidence):
        return 1.0

    def _is_filtered(self, parents_evidence):
        return False
    
    def _get_filter_penalty(self):
        return None

    def _compute_node_llr(self, xi, parents_evidence):
        xi_val = list(xi.values())[0]
        xi_name = list(xi.keys())[0]

        instance = parents_evidence.copy()
        instance[xi_name] = xi_val

        # Cache key: local assignment
        cache_key = frozenset(instance.items())
        if cache_key in self._cache.keys():
            return self._cache[cache_key]

        # Compute P(x | parents) under both models
        p_r = self._factors_r[xi_name].get_value(**instance)
        p_t = self._factors_t[xi_name].get_value(**instance)

        # Log-likelihood ratio for single node
        llr = np.log(p_r) - np.log(p_t)
        
        # Cache update
        self._cache[cache_key] = llr
        return llr
    
    def _compute_llr(self, instance):
        summation = 0
        for node_name, node_value in instance.items():
            xi = {node_name: node_value}
            parents_evidence = {p: instance[p] for p in self._parents[node_name]}

            # Check whether this instance should be filtered
            filtered = self._is_filtered(parents_evidence)
            if filtered:
                # Apply filtering penalty and stop computation
                summation += self._get_filter_penalty()
                break

            # Compute node weight and weighted local LLR contribution
            weight = self._compute_weight(xi, parents_evidence)
            summation += self._compute_node_llr(xi, parents_evidence) * weight

        return summation