import gc

import numpy as np

from pgmpy.estimators import BayesianEstimator

from src.mia.InferenceEngine import InferenceEngine
from experiment_suite.core.estimators import run_estimator
from src.mia.llr.FilteredLLR import FilteredLLR
from src.mia.llr.FilteredWeightedLLR import FilteredWeightedLLR
from src.mia.llr.StandardLLR import StandardLLR
from experiment_suite.core.metrics import compute_bn_dim, compute_theoretical_tpr, compute_tpr_fpr
from src.mia.llr.WeightedLLR import WeightedLLR
from src.mia.weighting.EntropyWeightingStrategy import EntropyWeightingStrategy
from src.mia.weighting.KLWeightingStrategy import KLWeightingStrategy
from src.mia.weighting.ParentsCountsWeightingStrategy import ParentsCountsWeightingStrategy
from src.mia.weighting.ParentsProbabilityWeightingStrategy import ParentsProbabilityWeightingStrategy

def run_experiment_wrapper(args):
    bayesian_network, P, exp_settings, prior_type, size_target_population_vec, seed = args
    rng = np.random.default_rng(seed)
    return run_experiment(bayesian_network, P, exp_settings, prior_type, size_target_population_vec, rng)

def run_experiment(bayesian_network, P, exp_settings, prior_type, size_target_population_vec, np_rng):
    """
    Single extraction pipeline:
    1. samples reference/target/evaluation populations
    2. estimates Bayesian network parameters
    3. computes LLR distributions for different methods
    4. evaluates empirical and theoretical ROC curves
    """

    # 1. Sample populations R, T, E
    R, T, E, ground_truth = extract_reference_target_eval_pop(
        P,
        exp_settings["reference_proportion"],
        exp_settings["target_proportion"],
        np_rng
    )

    equivalent_sample_size_R = np.sqrt(len(R))
    equivalent_sample_size_T = np.sqrt(len(T))
    
    # 2. Estimate bayesian network parameters and inference engines from R and T
    bn_theta_r = estimate_bn_parameters_from_data(bayesian_network, R, prior_type, equivalent_sample_size_R, np_rng)
    inference_engine_theta_r = InferenceEngine(bn_theta_r)

    bn_theta_t = estimate_bn_parameters_from_data(bayesian_network, T, prior_type, equivalent_sample_size_T, np_rng)
    inference_engine_theta_t = InferenceEngine(bn_theta_t)
    
    roc_results = []
    for size_target_population in size_target_population_vec:
        runtime = {
            "bn_theta_r": bn_theta_r,
            "bn_theta_t": bn_theta_t,
            "inference_engine_theta_r": inference_engine_theta_r,
            "inference_engine_theta_t": inference_engine_theta_t,
            "size_T_estimate": size_target_population
        }
        roc_metrics = compute_metrics(runtime, T, R, E, ground_truth, exp_settings)
        roc_results.append(roc_metrics)
    
    # Cleanup (free memory)
    del inference_engine_theta_r
    del inference_engine_theta_t
    del bn_theta_r
    del bn_theta_t
    del R, T, E
    
    gc.collect()
    return roc_results
    
def compute_metrics(runtime, T, R, E, ground_truth, exp_settings):
    # Stores LLR distributions per method (filled by run_estimator)
    llr_distributions = {}

    # ROC metrics container
    roc_metrics_i = {
        "theoretical_metrics": {
            "tpr": [],
            "fpr": [],
            "config": {}
        }
    }

    # Create all the estimators
    entropyWeightingStrategies = [
        EntropyWeightingStrategy(runtime["bn_theta_t"], operation=operation, normalize=normalize)
        for normalize in [True, False]
        for operation in ["1", "/"]
    ]

    klWeightingStrategies = [
        KLWeightingStrategy(runtime["bn_theta_r"], runtime["bn_theta_t"], operation=operation)
        for operation in ["1", "/", "exp"]
    ]

    parentsCountsWeightingStrategies = [
        ParentsCountsWeightingStrategy(R, operation=operation)
        for operation in ["/", "log"]
    ]

    parentsProbabilityWeightingStrategies = [
        ParentsProbabilityWeightingStrategy(runtime["inference_engine_theta_t"], operation=operation)
        for operation in ["1", "/", "-", "-log"]
    ]

    weighting_strategies = (
        entropyWeightingStrategies
        + klWeightingStrategies
        + parentsCountsWeightingStrategies
        + parentsProbabilityWeightingStrategies
    )

    classes = [
        (StandardLLR, {}),
        (FilteredLLR, {}),
    ]

    classes += [
        (WeightedLLR, {"weighting_strategy": ws})
        for ws in weighting_strategies
    ]

    classes += [
        (FilteredWeightedLLR, {"weighting_strategy": ws})
        for ws in weighting_strategies
    ]
    
    # 3. Run LLR estimators of each class
    for cls, params in classes:
        run_estimator(cls, params, runtime, E, roc_metrics_i, llr_distributions)
    
    # 4a. Theoretical ROC curve  
    errors_thresholds = np.logspace(exp_settings["start_error"], exp_settings["end_error"], exp_settings["step_error"])
    for error_threshold in errors_thresholds:
        theoretical_tpr = compute_theoretical_tpr(compute_bn_dim(runtime["bn_theta_r"]), len(T), error_threshold)
        roc_metrics_i["theoretical_metrics"]["tpr"].append(theoretical_tpr)
        roc_metrics_i["theoretical_metrics"]["fpr"].append(error_threshold)

    # 4b. Empirical ROC curve for each method
    for method_name, distr in llr_distributions.items():
        llr_E = distr["llr_distr_E"]
        
        # Quantile-based threshold sampling for ROC curve
        thresholds = np.percentile(llr_E, np.linspace(0, 100, 1000))
    
        for t in thresholds:        
            # Compute tpr and fpr for each threhsold
            tpr, fpr = compute_tpr_fpr(llr_E, ground_truth, t)
            roc_metrics_i[method_name]["tpr"].append(tpr)
            roc_metrics_i[method_name]["fpr"].append(fpr)
        
        # Sort ROC curve by FPR for monotonic plotting
        fprs = roc_metrics_i[method_name]["fpr"]
        tprs = roc_metrics_i[method_name]["tpr"]
        
        pairs = sorted(zip(fprs, tprs))
        fprs_sorted, tprs_sorted = zip(*pairs)
        
        roc_metrics_i[method_name]["fpr"] = list(fprs_sorted)
        roc_metrics_i[method_name]["tpr"] = list(tprs_sorted)
    
    return roc_metrics_i

def extract_reference_target_eval_pop(P, reference_proportion, target_proportion, np_rng):
    shuffled_idx = np_rng.permutation(P.index)

    reference_size = int(reference_proportion * len(P))
    pool_size = int(target_proportion * len(P))
    
    t_idx = shuffled_idx[:pool_size]
    r_idx = shuffled_idx[pool_size : pool_size + reference_size]
    
    # E = P \ R
    e_idx = np.concatenate([
        shuffled_idx[:pool_size],
        shuffled_idx[pool_size + reference_size:]
    ])

    T = P.loc[t_idx]
    R = P.loc[r_idx]
    E = P.loc[e_idx].copy()

    ground_truth = np.isin(e_idx, t_idx)
    return R, T, E, ground_truth

def build_pseudo_counts(bn_structure, equivalent_sample_size, prior_type, np_rng):
    pseudo_counts = {}
    noise_std = 0.05
    for node in bn_structure.nodes():
        cpd_orig = bn_structure.get_cpds(node)
        node_card = cpd_orig.variable_card
        parents_card = [bn_structure.get_cpds(p).variable_card
                         for p in bn_structure.get_parents(node)]
        n_parent_configs = int(np.prod(parents_card)) if parents_card else 1

        # Alpha assigned to each CPT row 
        alpha_per_row = equivalent_sample_size / n_parent_configs

        if prior_type == 'uniform':
            # Uniform over all states: equivalent to the standard BDeu prior
            alpha_per_cell = alpha_per_row / node_card
            counts = np.full((node_card, n_parent_configs), alpha_per_cell)

        elif prior_type == 'domain_knowledge':
            orig_probs = cpd_orig.values.reshape(node_card, n_parent_configs).copy()
            
            counts = np.zeros_like(orig_probs)
            for col in range(n_parent_configs):
                p = orig_probs[:, col]
                
                noise = np_rng.normal(0, noise_std, size=p.shape)
                perturbed = p + noise
                perturbed = np.maximum(perturbed, 1e-8)  # No negative values
                perturbed = perturbed / perturbed.sum()  # Normalize
                
                counts[:, col] = perturbed * alpha_per_row
            
            pseudo_counts[node] = counts

        else:
            raise ValueError(f"prior_type must be 'uniform' or 'domain_knowledge', received: '{prior_type}'")

        pseudo_counts[node] = counts

    return pseudo_counts


def estimate_bn_parameters_from_data(bn_structure, data,
                                     prior_type='uniform',
                                     equivalent_sample_size=5,
                                     np_rng=None):

    bn = bn_structure.copy()
    data = data.copy()

    # Ensure that all data values are integers
    for col in data.columns:
        data[col] = data[col].astype(int)

    # Retrieve state_names from the original network
    state_names = {}
    for node in bn_structure.nodes():
        cpd = bn_structure.get_cpds(node)
        if cpd.state_names and node in cpd.state_names:
            state_names[node] = cpd.state_names[node]
        else:
            state_names[node] = list(range(cpd.variable_card))

    # Build the custom pseudo-counts
    pseudo_counts = build_pseudo_counts(
        bn_structure, equivalent_sample_size, prior_type, np_rng
    )

    # Fit using a custom Dirichlet prior
    bn.fit(
        data,
        estimator=BayesianEstimator,
        state_names=state_names,
        prior_type='dirichlet',
        pseudo_counts=pseudo_counts,
    )

    assert bn.check_model()
    return bn