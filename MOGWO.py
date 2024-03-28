import numpy as np
import Optimizer
from copy import deepcopy

class wolf:
    def __init__(self, nb_zones):
        self.position = [0] * nb_zones
        self.IsDominated = None
        self.gridIndex = []
        self.gridSubIndex = []
        self.cost = []
        self.coverage = None
        self.cost_value = None
        self.covered_targets = []



def calculate_cost(solution):
    return np.count_nonzero(solution)



def evolve(max_eval, nb_zones, communication_graph, coordinates, list_target_points, nb_targets, NoGrid, nRep, pop_size, coverage_graph):
    nb_current_evaluation = 0
    epoch = int(max_eval/pop_size) + 1
    pop_size = int(pop_size)
    nRep = int(nRep)
    NoGrid = int(NoGrid)
    #target_covering_zones = Optimizer.covering_zones_for_each_target(coordinates, list_target_points)
    wolves = []
    for i in range(pop_size):
        wolves.append(wolf(nb_zones))
        wolves[i].position = Optimizer.create_random_pos(nb_zones, communication_graph, coordinates)
        nb_current_evaluation = Optimizer.costfunction(wolves[i], nb_targets, list_target_points, coverage_graph,  nb_current_evaluation)
        wolves[i].IsDominated = False
    Optimizer.DetermineDomination(wolves)
    Repos = [deepcopy(item) for item in wolves if item.IsDominated == False]
    grid = Optimizer.CreateGrid(Repos, NoGrid, alpha=0.1)
    for r in range(len(Repos)):
        Repos[r] = Optimizer.FindGridIndex(Repos[r], grid)
    current_epoch =0
    while nb_current_evaluation < max_eval:
        a = 2 - 2 * current_epoch / (epoch - 1)
        current_epoch = current_epoch +1
        list_best = []
        c, b, alp = Optimizer.choose_leaders(Repos)
        list_best.append(deepcopy(alp))
        list_best.append(deepcopy(b))
        list_best.append(deepcopy(c))
        for idx in range(0, pop_size):
            A1, A2, A3 = a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1)
            C1, C2, C3 = 2 * np.random.uniform(), 2 * np.random.uniform(), 2 * np.random.uniform()
            X1 = np.abs(list_best[0].position - A1 * np.abs(C1 * list_best[0].position - wolves[idx].position))
            X2 = np.abs(list_best[1].position - A2 * np.abs(C2 * list_best[1].position - wolves[idx].position))
            X3 = np.abs(list_best[2].position - A3 * np.abs(C3 * list_best[2].position - wolves[idx].position))
            wolves[idx].position = (X1 + X2 + X3) / 3.0
            wolves[idx].position =  Optimizer.amend_position(wolves[idx].position, communication_graph, coordinates)
            nb_current_evaluation = Optimizer.costfunction(wolves[idx], nb_targets, list_target_points, coverage_graph,nb_current_evaluation)
        Repos.extend(wolves)
        Optimizer.DetermineDomination(Repos)
        Repos = [deepcopy(item) for item in Repos if item.IsDominated == False]
        for r in range(len(Repos)):
            Repos[r] = Optimizer.FindGridIndex(Repos[r], grid)
        if len(Repos) > nRep:
            extra = len(Repos) - nRep
            for e in range(extra):
                Repos = Optimizer.deleteOneRepositoryMember(Repos)
            grid = Optimizer.CreateGrid(Repos, NoGrid, alpha=0.1)
        #new_front_mogwo = Optimizer.coverage_cost_front_swarm(Repos)
        #Optimizer.save_results('mogwo_new_archi8', str(new_front_mogwo))
    return Repos