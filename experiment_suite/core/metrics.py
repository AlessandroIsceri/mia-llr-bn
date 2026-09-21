import math
import numpy as np
from scipy.stats import norm

def compute_bn_dim(bn):
    dim = 0
    for node in bn.nodes():
        # Get the CPD of the current node
        cpd = bn.get_cpds(node)
        card_node = cpd.variable_card
        
        # Get involved variables and their cardinality
        cpd_variables = cpd.variables 
        cpd_cardinality = cpd.cardinality
        
        # Compute the number of parent configurations
        n_parent_configs = 1
        for cpd_var, cpd_card in zip(cpd_variables, cpd_cardinality):
            if cpd_var == node:
                continue
            n_parent_configs = n_parent_configs * cpd_card
        
        # Update the sum 
        dim += (card_node - 1) * n_parent_configs
    return int(dim)

def compute_theoretical_tpr(bayesian_network_dim, training_set_size, error) -> float:

    # Compute bound
    bound = math.sqrt(bayesian_network_dim / training_set_size)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = norm.ppf(1 - error).item()
    z_one_minus_beta = bound - z_alpha
    beta = norm.cdf(z_one_minus_beta).item()

    return beta

def compute_tpr_fpr(llr_e, ground_truth, t):
    y_pred = llr_e < t
    tpr = sum(ground_truth & y_pred) / sum(ground_truth)
    
    negatives = (ground_truth == 0)
    fpr = np.sum(y_pred & negatives) / np.sum(negatives)
    return tpr, fpr

def average_roc_metrics(results, N, exp_settings):
    """
    Average ROC metrics across multiple extractions.

    For theoretical metrics:
        - values are identical across runs -> no averaging needed

    For empirical methods:
        - interpolate all ROC curves on a common FPR grid
        - compute mean TPR across runs
    """
    avg_roc_metrics = {}
    method_names = list(results[0].keys())

    for method_name in method_names:
        avg_roc_metrics[method_name] = {} 
        
        if method_name == "theoretical_metrics":
            # Theoretical metrics do not depend on the extraction of R, T, E
            avg_roc_metrics[method_name]["fpr"] = results[0][method_name]["fpr"]
            avg_roc_metrics[method_name]["tpr"] = results[0][method_name]["tpr"]
            continue
        
        # For all the other methods, get tpr and fpr
        fpr_curves = [results[i][method_name]["fpr"] for i in range(N)]
        tpr_curves = [results[i][method_name]["tpr"] for i in range(N)]
        
        # Common FPR grid for interpolation (log-spaced)
        common_fpr = np.logspace(exp_settings["start_error"], exp_settings["end_error"], exp_settings["step_error"])
        interpolated_tprs = []
        
        # Interpolate each ROC curve onto common FPR grid
        for fpr, tpr in zip(fpr_curves, tpr_curves):
            fpr_arr = np.array(fpr)
            tpr_arr = np.array(tpr)
            sort_idx = np.argsort(fpr_arr)
            
            interp_tpr = np.interp(common_fpr, fpr_arr[sort_idx], tpr_arr[sort_idx])
            interpolated_tprs.append(interp_tpr)
        
        # Average interpolated ROC curves
        avg_roc_metrics[method_name]["tpr"] = np.mean(interpolated_tprs, axis=0).tolist()
        avg_roc_metrics[method_name]["fpr"] = common_fpr.tolist()

    return avg_roc_metrics