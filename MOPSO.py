import numpy as np
import Optimizer
from copy import deepcopy

class Particle:
    def __init__(self, nb_zones):
        self.position = [0] * nb_zones
        self.cost = []
        self.coverage = None
        self.cost_value = None
        self.velocity = []
        self.best_position = []
        self.best_cost = []
        self.IsDominated = None
        self.gridIndex = None
        self.gridSubIndex = []

def evolve(max_eval, nb_zones, communication_graph, coordinates, nb_targets, list_target_points, pop_size, NoGrid, c1, c2, nRep):
    nb_current_evaluation = 0
    NoGrid = int(NoGrid)
    nRep = int(nRep)
    epoch = int(max_eval/ pop_size)+1
    w_max = 0.9
    w_min = 0.4
    beta = 1
    Particles =[]
    target_covering_zones = Optimizer.covering_zones_for_each_target(coordinates, list_target_points)
    for i in range(pop_size):
        Particles.append(Particle(nb_zones))
        Particles[i].position = Optimizer.create_random_pos(nb_zones, communication_graph, coordinates)
        Particles[i].velocity = np.zeros(nb_zones)
        nb_current_evaluation = Optimizer.costfunction(Particles[i], nb_targets, list_target_points, target_covering_zones, nb_current_evaluation)
        Particles[i].best_position = Particles[i].position
        Particles[i].best_cost = Particles[i].cost
        Particles[i].IsDominated = False
    Optimizer.DetermineDomination(Particles)
    Repos = [deepcopy(item) for item in Particles if item.IsDominated == False]
    grid = Optimizer.CreateGrid(Repos, NoGrid, alpha=0.1)
    for r in range(len(Repos)):
        Repos[r] = Optimizer.FindGridIndex(Repos[r], grid)
    current_iter=0
    while nb_current_evaluation < max_eval:
        leader = Optimizer.SelectLeader(Repos, beta)
        w = w_max - (w_max - w_min) * (current_iter / epoch)
        current_iter = current_iter + 1
        for i in range(pop_size):
            Particles[i].velocity = w * Particles[i].velocity + c1 * np.random.uniform() * np.subtract(Particles[i].best_position, Particles[i].position) \
                                    + c2 * np.random.uniform() * np.subtract(leader.best_position, Particles[i].position)
            Particles[i].position = Particles[i].position + Particles[i].velocity
            for j in range(len(Particles[i].position)):
                if Particles[i].position[j]<0:
                    Particles[i].position[j]=0
                if Particles[i].position[j]>1:
                    Particles[i].position[j]=1
            Particles[i].position = Optimizer.amend_position(Particles[i].position, communication_graph, coordinates)
            nb_current_evaluation = Optimizer.costfunction(Particles[i], nb_targets, list_target_points, target_covering_zones, nb_current_evaluation)
            if Optimizer.Dominates(Particles[i].cost, Particles[i].best_cost):
                Particles[i].best_position = deepcopy(Particles[i].position)
                Particles[i].best_cost = deepcopy(Particles[i].cost)
        Repos.extend(Particles)
        Optimizer.DetermineDomination(Repos)
        Repos = [deepcopy(item) for item in Repos if item.IsDominated == False]
        for r in range(len(Repos)):
            Repos[r] = Optimizer.FindGridIndex(Repos[r], grid)
        if len(Repos) > nRep:
            extra = len(Repos) - nRep
            for e in range(extra):
                Repos = Optimizer.deleteOneRepositoryMember(Repos)
            grid = Optimizer.CreateGrid(Repos, NoGrid, alpha=0.1)
        #new_front_mopso = Optimizer.coverage_cost_front_swarm(Repos)
        #Optimizer.save_results('mopso_450_archi10', str(new_front_mopso))

    return Repos