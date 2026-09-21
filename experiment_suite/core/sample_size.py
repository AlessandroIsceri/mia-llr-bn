from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from pgmpy.sampling import BayesianModelSampling

def _mean_coverage(sampler, node_info, total_params, size_T, n_simulations=100, min_hits=3):
    
    coverages = []
    
    for _ in range(n_simulations):
        samples = sampler.forward_sample(size=size_T, show_progress=False)
        
        covered = 0
        for n in node_info:
            if not n['parents']:
                # Root nodes always contribute all their free parameters
                covered += n['free_params_per_config']
            else:
                # Count only parent configurations observed at least min_hits times
                parent_counts = samples.groupby(n['parents']).size()
                unique = np.sum(parent_counts >= min_hits)
                covered += unique * n['free_params_per_config']
        coverages.append(covered / total_params)
    
    return float(np.mean(coverages))

def _search_single_target(args):
    bn, node_info, total_params, target, n_simulations = args
    min_hits = 3
    sampler = BayesianModelSampling(bn)
    
    lo = min_hits
    hi = total_params
    
    # Increase the search interval until the target coverage is reachable
    n_doublings = 0
    MAX_DOUBLINGS = 10
    while _mean_coverage(sampler, node_info, total_params, hi, n_simulations, min_hits) < target:
        hi *= 2
        n_doublings += 1
        if n_doublings > MAX_DOUBLINGS:
            # BN is pathological: signal with inf so it gets rejected
            return target, {'sample_size_T': float('inf'), 'actual_coverage': -1}

    best_t = lo
    best_coverage = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        coverage = _mean_coverage(sampler, node_info, total_params, mid, n_simulations, min_hits)
        if coverage >= target:
            best_t = mid
            best_coverage = coverage
            
            # Stop early if the estimate is sufficiently close to the target
            if abs(coverage - target) < 0.01:
                break
            hi = mid - 1
        else:
            lo = mid + 1

    return target, {'sample_size_T': best_t, 'actual_coverage': best_coverage}

def estimate_sample_size_for_coverage(bn, coverage_targets, target_proportion, n_simulations=100):
    
    node_info = []
    total_params = 0
    # Precompute structural information
    for node in bn.nodes():
        cpd = bn.get_cpds(node)
        parents = bn.get_parents(node)
        parents_card = [bn.get_cpds(p).variable_card for p in parents]
        n_parent_configs = int(np.prod(parents_card)) if parents_card else 1
        free_params = (cpd.variable_card - 1) * n_parent_configs
        total_params += free_params
        node_info.append({
            'node': node,
            'parents': parents,
            'n_parent_configs': n_parent_configs,
            'free_params_per_config': int(cpd.variable_card - 1),
        })
        
    results = {}
    
    # Estimate the required sample size for each coverage target in parallel
    with Pool(processes=len(coverage_targets)) as pool:
        target_results = list(tqdm(
            pool.imap(
                _search_single_target,
                [(bn, node_info, total_params, target, n_simulations) for target in sorted(coverage_targets)]
            ),
            total=len(coverage_targets),
            desc="Estimating sample sizes for coverage targets"
        ))
    results = dict(target_results)
    
    if any(v['sample_size_T'] == float('inf') for v in results.values()):
        return {target: float('inf') for target in coverage_targets}
    
    sampler = BayesianModelSampling(bn)
    
    # Reduce sample sizes that overshoot the requested coverage by a large margin
    idxs_to_be_decreased = []
    for i, coverage_target in enumerate(coverage_targets):
        if results[coverage_targets[i]]['actual_coverage'] > coverage_target + 0.08:
            idxs_to_be_decreased.append(i)
    
    for i in idxs_to_be_decreased:
        best_t = results[coverage_targets[i]]['sample_size_T'] - 1
        cov = _mean_coverage(sampler=sampler, node_info=node_info, total_params=total_params,
                            size_T=best_t, n_simulations=n_simulations, min_hits=best_t)
        results[coverage_targets[i]] = {
            'sample_size_T': best_t,
            f'actual_coverage - *min_hits={best_t}': cov
        }
        
    # Ensure sample sizes increase monotonically with the coverage target    
    for i in range(len(coverage_targets) - 1):
        if results[coverage_targets[i]]['sample_size_T'] >= results[coverage_targets[i + 1]]['sample_size_T']:
            best_t = results[coverage_targets[i + 1]]['sample_size_T'] - 1
            cov = _mean_coverage(sampler=sampler, node_info=node_info, total_params=total_params,
                                size_T=best_t, n_simulations=n_simulations, min_hits=best_t)
            results[coverage_targets[i]] = {
                'sample_size_T': best_t,
                f'actual_coverage - *min_hits={best_t}': cov
            }
    
    # Avoid assigning the same minimum sample size to multiple coverage targets
    n_equals_min_hits = 0
    for i in range(len(coverage_targets)):
        if results[coverage_targets[i]]['sample_size_T'] == 3:
            n_equals_min_hits += 1
    
    for i in range(n_equals_min_hits - 1):
        best_t = i + 2
        cov = _mean_coverage(sampler=sampler, node_info=node_info, total_params=total_params,
                            size_T=best_t, n_simulations=n_simulations, min_hits=best_t)
        results[coverage_targets[i]] = {
            'sample_size_T': best_t,
            f'actual_coverage - *min_hits={best_t}': cov
        }

    return {target: int(results[target]['sample_size_T'] / target_proportion) for target in coverage_targets}