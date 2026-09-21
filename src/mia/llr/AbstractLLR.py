from abc import ABC, abstractmethod
import numpy as np

class AbstractLLR(ABC):

    def estimate_distribution(self, data):
        
        # LLR for each instance
        llr_distribution = np.zeros(len(data))   
            
        for i, instance in enumerate(data.to_dict("records")):
            llr_distribution[i] = self._compute_llr(instance)
                        
        return llr_distribution
    
    @abstractmethod
    def _compute_llr(self, instance):
        # Must return scalar LLR for one instance
        pass