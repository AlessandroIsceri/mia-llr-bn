from src.mia.weighting.WeightingStrategy import WeightingStrategy
import math

class ParentsCountsWeightingStrategy(WeightingStrategy):

    OPERATIONS_MAP = {
        "/": lambda count: count / (count + 1),
        "log": lambda count: math.log(1 + count),
    }

    def __init__(self, R, operation):
        self.R = R
        self.operation = operation
        self.weight_function = self.OPERATIONS_MAP[operation]
        self._cache = {}

    def get_weight(self, _, parents_evidence):
        cache_key = tuple(sorted(parents_evidence.items()))
        if cache_key not in self._cache:
            if len(parents_evidence) == 0:
                count = len(self.R)
            else:
                # Count how many samples satisfy the parent configuration
                query_string = " & ".join([f"({k} == {v})" for k, v in parents_evidence.items()])
                count = len(self.R.query(query_string))
            self._cache[cache_key] = self.weight_function(count)

        return self._cache[cache_key]
    
    def get_operation(self):
        return self.operation