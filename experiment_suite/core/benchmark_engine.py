import os
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

from experiment_suite.core.experiment import run_experiment_wrapper
from experiment_suite.core.generation import generate_general_pop
from experiment_suite.core.metrics import average_roc_metrics, compute_bn_dim
from experiment_suite.core.sample_size import estimate_sample_size_for_coverage
from experiment_suite.core.utils import bn_to_json, compute_cardinality_dict

def run_benchmark_experiments(config, bayesian_network):    
    # Get experiment settings
    exp_settings = config["exp_settings"]
    n_sample_extractions = exp_settings["n_sample_extractions"]
    
    # Set random seed for reproducibility
    random_seed = exp_settings["random_state"]
    np.random.seed(random_seed)
                 
    cardinality_dict = compute_cardinality_dict(bayesian_network)
    bn_complexity = compute_bn_dim(bayesian_network)    
    
    COVERAGE_TARGETS = [0.1, 0.25, 0.5, 0.75]
    size_target_population_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]

    size_general_population_dict = estimate_sample_size_for_coverage(bayesian_network, COVERAGE_TARGETS, exp_settings["target_proportion"])
    
    experiment_results = {
        "config": {
            "bn": compute_bn_config(bayesian_network),
            "exp_settings": exp_settings
        },
        "bns": [{
            "bn": bn_to_json(bayesian_network, cardinality_dict),
            "bn_complexity": bn_complexity,
            "bn_i": 0
        }],
        "general_population_sizes": [
            {
                "target_coverage": target,
                "bn_sizes": []
            }
            for target in COVERAGE_TARGETS
        ],
        "results": [
            {
                "prior_type": prior_type,
                "coverage_results": [
                    {
                        "target_coverage": target,
                        "t_size_estimates": [
                            {
                                "target_size_multiplier": m,
                                "bn_results": []
                            }
                            for m in size_target_population_multipliers
                        ]
                    }
                    for target in COVERAGE_TARGETS
                ]
            }
            for prior_type in exp_settings["prior_type_vec"]
        ]
    }
    
    # Index maps for lookup into the results list structure
    prior_type_index = {pt: i for i, pt in enumerate(exp_settings["prior_type_vec"])}
    target_index = {t: i for i, t in enumerate(COVERAGE_TARGETS)}
    multiplier_index = {m: i for i, m in enumerate(size_target_population_multipliers)}
    
    for target in COVERAGE_TARGETS:
        size_general_population = size_general_population_dict[target]
            
        P = generate_general_pop(bayesian_network, size_general_population, random_seed)
        print("Unique ratio: ", str(len(P.drop_duplicates()) / len(P)))
        
        t_idx = target_index[target]
        experiment_results["general_population_sizes"][t_idx]["bn_sizes"].append({
            "bn_i": 0,
            "size": size_general_population
        })
        
        size_target_population_true = int(len(P) * exp_settings["target_proportion"])
        size_target_population_vec = [int(size_target_population_true * m) for m in size_target_population_multipliers]
        
        for prior_type in exp_settings["prior_type_vec"]:
            pt_idx = prior_type_index[prior_type]
            
            # Perform the experiment n_sample_extractions times on the generated bn and population
            with Pool(processes=os.cpu_count(), maxtasksperchild=1) as pool:
                roc_metrics = list(
                    tqdm(
                        pool.imap(
                            run_experiment_wrapper,
                            [
                                (bayesian_network, P, exp_settings, prior_type, size_target_population_vec, random_seed + seed)
                                for seed in range(n_sample_extractions)
                            ]
                        ),
                        total=n_sample_extractions,
                        desc=f"prior_type={prior_type}, target={target}"
                    )
                )
            
            for m in size_target_population_multipliers:
                m_idx = multiplier_index[m]
                
                # Average the obtained results
                roc_metrics_for_t = [roc_metrics[seed][m_idx] for seed in range(n_sample_extractions)]
                avg_metrics = average_roc_metrics(roc_metrics_for_t, n_sample_extractions, exp_settings)
                
                experiment_results["results"][pt_idx]["coverage_results"][t_idx]["t_size_estimates"][m_idx]["bn_results"].append({
                    "bn_i": 0,
                    "avg_metrics": avg_metrics
                })
        
    return experiment_results
   
def compute_bn_config(bn):
    n_nodes = len(bn.nodes())
    n_arcs = len(bn.edges())
    ratio_arc = n_arcs / n_nodes if n_nodes > 0 else 0

    cardinalities = [cpd.variable_card for cpd in bn.get_cpds()]
    n_modmin = min(cardinalities)
    n_modmax = max(cardinalities)

    return {
        "n_nodes": n_nodes,
        "ratio_arc": ratio_arc,
        "n_modmin": n_modmin,
        "n_modmax": n_modmax
    }