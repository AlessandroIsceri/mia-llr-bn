import os
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

from experiment_suite.core.experiment import run_experiment_wrapper
from experiment_suite.core.generation import generate_general_pop, generate_random_bn
from experiment_suite.core.metrics import average_roc_metrics, compute_bn_dim
from experiment_suite.core.sample_size import estimate_sample_size_for_coverage
from experiment_suite.core.utils import bn_to_json

def run_synthetic_experiments(config):
    # Get experiment settings
    exp_settings = config["exp_settings"]
    n_bn_extractions = exp_settings["n_bn_extractions"]
    n_sample_extractions = exp_settings["n_sample_extractions"]
    
    COVERAGE_TARGETS = [0.1, 0.25, 0.5, 0.75]
    
    size_target_population_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    # Set random seed for reproducibility
    random_seed = exp_settings["random_state"]
    np_rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)
    
    experiment_results = {
        "config": config,
        "bns": [],
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
    
    generated_bayesian_networks = []
    cardinality_dicts = []
    bn_complexities = []
    per_bn_size_general_population_vec = []
    for bn_i in range(n_bn_extractions):
        
        # Extract n_bn_extractions bns using the current configuration
        bayesian_network, cardinality_dict = generate_random_bn(
            config["bn"]["n_nodes"],
            config["bn"]["ratio_arc"],
            config["bn"]["n_modmin"],
            config["bn"]["n_modmax"],
            np_rng
        )
        
        generated_bayesian_networks.append(bayesian_network)
        cardinality_dicts.append(cardinality_dict)
        bn_complexity = compute_bn_dim(bayesian_network)
        bn_complexities.append(bn_complexity)
        
        # Compute per-bn sample sizes for each coverage target
        per_bn_size_general_population_vec.append(estimate_sample_size_for_coverage(bayesian_network, COVERAGE_TARGETS, exp_settings["target_proportion"]))
        experiment_results["bns"].append({
                "bn": bn_to_json(bayesian_network, cardinality_dict),
                "bn_complexity": bn_complexity,
                "bn_i": bn_i 
            })
    
    
    # Compute the median population size for each coverage target across
    # all generated BNs. BNs whose required population size exceeds 5x
    # the median for any target are regenerated.
    median_per_target = {
        target: np.median([s[target] for s in per_bn_size_general_population_vec])
        for target in COVERAGE_TARGETS
    }
    
    for bn_i in range(n_bn_extractions):
        while any(
            per_bn_size_general_population_vec[bn_i][target] > 5 * median_per_target[target]
            for target in COVERAGE_TARGETS
        ):
            print(f"BN {bn_i} rejected, regenerating...")
                        
            bayesian_network, cardinality_dict = generate_random_bn(config["bn"]["n_nodes"],
                config["bn"]["ratio_arc"],
                config["bn"]["n_modmin"],
                config["bn"]["n_modmax"],
                np_rng
            )
            
            size_dict = estimate_sample_size_for_coverage(bayesian_network, COVERAGE_TARGETS, exp_settings["target_proportion"])
            bn_complexity = compute_bn_dim(bayesian_network)
            
            # Replace all references to this BN consistently
            generated_bayesian_networks[bn_i] = bayesian_network
            cardinality_dicts[bn_i] = cardinality_dict
            per_bn_size_general_population_vec[bn_i] = size_dict
            bn_complexities[bn_i] = bn_complexity
            
            experiment_results["bns"][bn_i] = {
                "bn": bn_to_json(bayesian_network, cardinality_dict),
                "bn_complexity": bn_complexity,
                "bn_i": bn_i
            }
            
            # Recompute medians after replacement
            median_per_target = {
                target: np.median([s[target] for s in per_bn_size_general_population_vec])
                for target in COVERAGE_TARGETS
            }
    
    for target in COVERAGE_TARGETS:
        t_idx = target_index[target]
        
        for bn_i, (bayesian_network, size_general_population_vec) in enumerate(zip(
            generated_bayesian_networks, 
            per_bn_size_general_population_vec)):
                
            size_general_population = size_general_population_vec[target]
            P = generate_general_pop(bayesian_network, size_general_population, random_seed)
            print("Unique ratio: ", str(len(P.drop_duplicates()) / len(P)))
            
            size_target_population_true = int(len(P) * exp_settings["target_proportion"])
            size_target_population_vec = [int(size_target_population_true * m) for m in size_target_population_multipliers]
            
            experiment_results["general_population_sizes"][t_idx]["bn_sizes"].append({
                "bn_i": bn_i,
                "size": size_general_population
            })
            
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
                            desc=f"config={config['bn']} prior={prior_type} bn {bn_i+1}/{n_bn_extractions} target={target}"
                        )
                    )

                for m in size_target_population_multipliers:
                    m_idx = multiplier_index[m]
                    
                    # Average the obtained results
                    roc_metrics_for_t = [roc_metrics[seed][m_idx] for seed in range(n_sample_extractions)]
                    avg_metrics = average_roc_metrics(roc_metrics_for_t, n_sample_extractions, exp_settings)
                    
                    experiment_results["results"][pt_idx]["coverage_results"][t_idx]["t_size_estimates"][m_idx]["bn_results"].append({
                        "bn_i": bn_i,
                        "avg_metrics": avg_metrics
                    })
    
    return experiment_results