def bn_to_json(bn, cardinality_dict):
    nodes = list(bn.nodes())

    edges = []
    for u, v in bn.edges():
        edges.append({"from": u, "to": v})

    data = {
        "nodes": [
            {
                "name": node,
                "cardinality": int(cardinality_dict.get(node, -1))
            }
            for node in nodes
        ],
        "edges": edges
    }
    return data

def compute_cardinality_dict(bayesian_network):
    cardinality_dict = {}

    for node in bayesian_network.nodes():
        cpd = bayesian_network.get_cpds(node)
        cardinality_dict[node] = cpd.variable_card

    return cardinality_dict