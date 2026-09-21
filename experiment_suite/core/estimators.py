import inspect

from src.mia.weighting.EntropyWeightingStrategy import EntropyWeightingStrategy
from src.mia.weighting.KLWeightingStrategy import KLWeightingStrategy
from src.mia.weighting.ParentsProbabilityWeightingStrategy import ParentsProbabilityWeightingStrategy

def run_estimator(cls, params, runtime, E, roc_metrics_i, llr_distributions):
    cur_estimator = build_class(cls, params, runtime)
    
    method_str = ""
    if "weighting_strategy" in params.keys():
        method_str += " - " + params["weighting_strategy"].__class__.__name__
        if isinstance(params["weighting_strategy"], EntropyWeightingStrategy):
            method_str += " - normalize = " + str(params["weighting_strategy"].get_normalize())
        method_str += " - operation = " + params["weighting_strategy"].get_operation()
    
    method_name = cls.__name__ + method_str
    llr_distributions[method_name] = {
        "llr_distr_E": cur_estimator.estimate_distribution(E)
    }
        
    roc_metrics_i[method_name] = {
        "tpr": [],
        "fpr": []
    }

def build_class(cls, params, runtime):
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    for name, param in sig.parameters.items():

        if name == "self":
            continue

        if name in params:
            kwargs[name] = params[name]
            continue

        # Runtime resource
        if name in runtime:
            kwargs[name] = runtime[name]
            continue

        # Missing mandatory param
        if param.default is inspect.Parameter.empty:
            raise ValueError(f"Missing required dependency: {name}")

    return cls(**kwargs)