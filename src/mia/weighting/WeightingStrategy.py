from abc import ABC, abstractmethod

class WeightingStrategy(ABC):
    
    @abstractmethod
    def get_weight(self, xi, parents_evidence):
        # Must return a scalar weight for node Xi given its parents context
        pass