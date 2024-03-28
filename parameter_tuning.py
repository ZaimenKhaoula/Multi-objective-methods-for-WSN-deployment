from irace import irace
from pymoo.indicators.hv import HV
import launch_algos
import numpy as np
import init_env
import NSGAII as nsga
import MOGWO
from MOEAD import MOEAD
import MOPSO
import SPEAII
import os


def compute_hypervolume(front, nb_targets, nb_zones):
    new_front = []
    nadir=[nb_targets, nb_zones]
    for p in front:
        new_front.append([(nb_targets - p.coverage), p.cost])
    hv_calculator = HV(ref_point=np.array(nadir))
    print("************* front **************")
    print(nadir)
    print(np.array(new_front))
    return hv_calculator(np.array(new_front))




# Fixed parameter
mutation_prob = 0.1

# Parameter configuration space
configuration = {
    "pop_size": [3,4]
}

crossover_prob=1
mutation_prob =0

# Objective function
def evaluate(experiment, scenario):
    results = []



    #Instance 1
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = launch_algos.init_env.init_archi1()
    # Some configurations produced a warning, but the values are within the limits. That seems a bug in scipy. TODO: Report the bug to scipy.
    print(f'{experiment["configuration"]}')
    front= nsga.evolve(3,  list_target_points, communication_graph, nb_zones, coordinates, 2,1 , mutation_prob, **experiment["configuration"])
    hv=compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    # Instance 9
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = launch_algos.init_env.init_archi9()
    front = nsga.evolve(3, list_target_points, communication_graph, nb_zones, coordinates, 2, 1, mutation_prob, **experiment["configuration"])
    hv = compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    # Instance 4
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = launch_algos.init_env.init_archi7()
    front = nsga.evolve(3, list_target_points, communication_graph, nb_zones, coordinates, 2, 1, mutation_prob,  **experiment["configuration"])
    hv = compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    print("hypervolume    ", np.mean(results))
    # Return hypervolume mean
    return dict(cost=np.mean(results))

scenario = dict(
    instances = np.arange(5),
    maxExperiments = 96,
    logFile = ""
    )

parameters_table = '''
pop_size      "" i (3, 6)
crossover     "" r (0.8, 0.9)
'''



default_values = '''
initial_temp restart_temp_ratio visit accept no_local_search
5230         3               2.62   -5.0   ""
'''

# Tune parameters
#tuner = irace.IRAce(configuration, evaluate, max_evals=2)
tuner = irace(scenario, parameters_table,  evaluate)
#tuner.set_initial_from_str(default_values)
best_confs = tuner.run()
# Pandas DataFrame
print("best config *************************************")
print(best_confs)
print("ok")