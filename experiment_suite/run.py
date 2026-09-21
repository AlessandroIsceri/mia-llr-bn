import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from experiment_suite.core.benchmark_engine import run_benchmark_experiments

from .core.synthetic_engine import run_synthetic_experiments
import json
import argparse
from multiprocessing import freeze_support
from pathlib import Path
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
 
def remap_bn_to_int(bn):
    state_maps = {}
    
    for cpd in bn.get_cpds():
        var = cpd.variable
        state_maps[var] = {state: i for i, state in enumerate(cpd.state_names[var])}
    
    new_cpds = []
    for cpd in bn.get_cpds():
        var = cpd.variable
        parents = cpd.variables[1:]
        
        new_cpd = TabularCPD(
            variable=var,
            variable_card=cpd.variable_card,
            values=cpd.get_values(),
            evidence=parents if parents else None,
            evidence_card=[len(cpd.state_names[p]) for p in parents] if parents else None,
            state_names={v: list(range(len(state_maps[v]))) for v in cpd.state_names}
        )
        new_cpds.append(new_cpd)
    
    bn.remove_cpds(*bn.get_cpds())
    bn.add_cpds(*new_cpds)
    bn.check_model()
    
    return state_maps

if __name__ == "__main__":
    freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    
    args = parser.parse_args()
    exp_dir = Path(args.exp)
        
    config_path = exp_dir / "config.json"
    output_path = exp_dir / "experiment_results.json"
    
    with open(config_path) as f:
        config = json.load(f)
        
    bif_files = list(exp_dir.glob("*.bif"))

    # Run main experiment pipeline
    if bif_files:
        bayesian_network = DiscreteBayesianNetwork.load(str(bif_files[0]), filetype="bif")
        remap_bn_to_int(bayesian_network)
        experiment_results = run_benchmark_experiments(
            config,
            bayesian_network
        )     
    else:
        experiment_results = run_synthetic_experiments(config)

    # Persist results to disk
    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=4)