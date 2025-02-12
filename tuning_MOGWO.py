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

epoch= 100

def compute_hypervolume(front, nb_targets, nb_zones):
    new_front = []
    nadir = [nb_targets, nb_zones]
    for p in front:
        new_front.append([(nb_targets - p.coverage), p.cost_value])
    hv_calculator = HV(ref_point=np.array(nadir))
    return hv_calculator(np.array(new_front))



# Objective function
def target_runner(experiment, scenario):
    results = []

    # Instance 1
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = init_env.init_archi1()
    # Some configurations produced a warning, but the values are within the limits. That seems a bug in scipy. TODO: Report the bug to scipy.
    print(f'{experiment["configuration"]}')
    front = MOGWO.evolve(epoch, nb_zones, communication_graph, coordinates, list_target_points, nb_targets, **experiment["configuration"])
    hv = compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    # Instance 9
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = init_env.init_archi9()
    front = MOGWO.evolve(epoch, nb_zones, communication_graph, coordinates, list_target_points, nb_targets,  **experiment["configuration"])
    hv = compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    # Instance 7
    nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = init_env.init_archi7()
    front = MOGWO.evolve(epoch, nb_zones, communication_graph, coordinates, list_target_points, nb_targets, **experiment["configuration"])
    hv = compute_hypervolume(front, nb_targets, nb_zones)
    results.append(hv)

    print("hypervolume    ", np.mean(results))
    # Return hypervolume mean
    return dict(cost= (np.mean(results) * -1))


scenario = dict(
    instances=np.arange(1),
    maxExperiments=96,
    logFile=""
)


parameters_table = '''
NoGrid "" i (30, 50)
nRep  "" i (40, 50)
pop_size  "" i (70, 150)
'''

default_values = '''
initial_temp restart_temp_ratio visit accept no_local_search
5230         3               2.62   -5.0   ""
'''

# Tune parameters
# tuner = irace.IRAce(configuration, evaluate, max_evals=2)
tuner = irace(scenario, parameters_table, target_runner)
# tuner.set_initial_from_str(default_values)
best_confs = tuner.run()
# Pandas DataFrame
print("best config *************************************")
print(best_confs)
print("ok")