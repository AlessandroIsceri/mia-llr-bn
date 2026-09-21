import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")
from itertools import product as iterproduct
from pgmpy.inference import VariableElimination

class InferenceEngine:
    """
    Wrapper around pgmpy VariableElimination to provide a simplified API:
    - setEvidence / eraseAllEvidence
    - evidenceProbability
    """
 
    def __init__(self, bn):
        self._bn = bn
        
        # Current evidence state
        self._evidence = {}
        
        # Cache: sorted evidence tuple -> probability
        self._cache = {}
        
        # Precompute factors for each node to speed up joint probability evaluation
        self._factors = {
            node: bn.get_cpds(node).to_factor()
            for node in bn.nodes()
        }

        # Store the parent list of every node
        self._parents = {
            node: list(bn.predecessors(node))
            for node in bn.nodes()
        }

        # Initialize the Variable Elimination inference engine
        self._ve = VariableElimination(bn)
 
    def eraseAllEvidence(self):
        self._evidence = {}
 
    def setEvidence(self, evidence: dict):
        self._evidence = dict(evidence)
 
    def evidenceProbability(self) -> float:
        if not self._evidence:
            raise ValueError("Evidence is not set. Call setEvidence() before evidenceProbability().")

        key = tuple(sorted(self._evidence.items()))
        if key in self._cache.keys():
            return self._cache[key]

        all_vars = list(self._bn.nodes())
        evidence_vars = set(self._evidence)
        missing = [v for v in all_vars if v not in evidence_vars]

        if not missing:
            # Complete instance
            p = self._joint_prob(self._evidence)
        else:
            # Otherwise, marginalize over the unobserved variables using VE
            result = self._ve.query(
                variables=list(self._evidence.keys()),
                evidence={},
                show_progress=False,
            )
            p = float(result.get_value(**self._evidence))
        
        # Cache and return the computed probability
        self._cache[key] = p
        return p
 
    def _joint_prob(self, evidence) -> float:
        p = 1.0

        for node, factor in self._factors.items():
            args = {node: evidence[node]}

            # Add parent assignments required by the factor
            for par in self._parents[node]:
                args[par] = evidence[par]

            p *= factor.get_value(**args)

            # Early stopping if the probability becomes zero
            if p == 0.0:
                return 0.0

        return float(p)