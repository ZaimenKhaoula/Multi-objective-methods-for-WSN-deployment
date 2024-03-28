import time
from Individual import Population, Individual
import Optimizer
from Levenshtein import distance as levenshtein_distance
import numpy as np
import Optimizer
from copy import deepcopy
import random


class wolf:
    def __init__(self, nb_zones):
        self.position = [0] * nb_zones
        self.IsDominated = None
        self.gridIndex = []
        self.gridSubIndex = []
        self.cost = []
        self.coverage = None
        self.cost_value = None
        self.covered_targets =[]



def calculate_cost(solution):
    return np.count_nonzero(solution)



def evolve(max_eval, nb_zones, communication_graph, coordinates, list_target_points, nb_targets, NoGrid, nRep, pop_size_nsga, pop_size_mogwo, num_of_tour_particips, tournament_prob, mutation_param, crossover,  coverage_graph ):
    nb_current_evaluation=0
    nRep = int(nRep)
    epoch = int(max_eval / pop_size_mogwo)
    NoGrid = int(NoGrid)
    #target_covering_zones = Optimizer.covering_zones_for_each_target(coordinates, list_target_points)
    wolves = []
    #t_start=time.time()
    for i in range(pop_size_mogwo):
        wolves.append(wolf(nb_zones))
        wolves[i].position = Optimizer.create_random_pos(nb_zones, communication_graph, coordinates)
        nb_current_evaluation = Optimizer.costfunction(wolves[i], nb_targets, list_target_points, coverage_graph , nb_current_evaluation)
        wolves[i].IsDominated = False
    Optimizer.DetermineDomination(wolves)
    Repos = [deepcopy(item) for item in wolves if item.IsDominated == False]
    grid = Optimizer.CreateGrid(Repos, NoGrid, alpha=0.1)
    for r in range(len(Repos)):
        Repos[r] = Optimizer.FindGridIndex(Repos[r], grid)
    current_epoch = 0
    while nb_current_evaluation < int(max_eval/4):
    #for current_epoch in range(int(epoch /3)):
        a = 2 - 2 * current_epoch / (epoch - 1)
        current_epoch = current_epoch + 1
        list_best = []
        c, b, alp = Optimizer.choose_leaders(Repos)
        list_best.append(deepcopy(alp))
        list_best.append(deepcopy(b))
        list_best.append(deepcopy(c))
        for idx in range(0, pop_size_mogwo):
            A1, A2, A3 = a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1), a * (2 * np.random.uniform() - 1)
            C1, C2, C3 = 2 * np.random.uniform(), 2 * np.random.uniform(), 2 * np.random.uniform()
            X1 = np.abs(list_best[0].position - A1 * np.abs(C1 * list_best[0].position - wolves[idx].position))
            X2 = np.abs(list_best[1].position - A2 * np.abs(C2 * list_best[1].position - wolves[idx].position))
            X3 = np.abs(list_best[2].position - A3 * np.abs(C3 * list_best[2].position - wolves[idx].position))
            wolves[idx].position = (X1 + X2 + X3) / 3.0
            wolves[idx].position =  Optimizer.amend_position(wolves[idx].position, communication_graph, coordinates)
            nb_current_evaluation = Optimizer.costfunction(wolves[idx], nb_targets, list_target_points, coverage_graph, nb_current_evaluation)
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
        #Optimizer.save_results('new-local-archi4-450', str(new_front_mogwo))
    ############################################  NSGA-II   ##################################################################
    #print("time to execute gwo is ", time.time()-t_start)
    size = pop_size_nsga - len(Repos)
    population = Optimizer.create_population(size, nb_zones, communication_graph, coordinates, list_target_points, coverage_graph)
    nb_current_evaluation = nb_current_evaluation + pop_size_nsga
    for r in Repos:
        indiv = Individual(nb_zones)
        indiv.deployment = r.position
        indiv.coverage = r.coverage
        indiv.cost = np.count_nonzero(indiv.deployment)
        indiv.covered_targets = r.covered_targets
        population.append(indiv)
    #print("nb evaluation before starting nsga-ii ", nb_current_evaluation)
    #t_start = time.time()
    while nb_current_evaluation < max_eval:
    #for current_epoch in range(int((2*epoch)/3)):
        Optimizer.fast_nondominated_sort(population)
        #new_front_nsga = Optimizer.coverage_cost_front(population.fronts[0])
        #Optimizer.save_results('new-local-archi8-450', str(new_front_nsga))
        for front in population.fronts:
            Optimizer.calculate_crowding_distance(front)
        children = Optimizer.create_children_basic(population, coordinates, communication_graph, list_target_points,
                                                   num_of_tour_particips, tournament_prob, crossover,
                                                   mutation_param, coverage_graph)
        nb_current_evaluation = nb_current_evaluation + pop_size_nsga
        population.population.extend(children)
        Optimizer.fast_nondominated_sort(population)

        new_population = Population()
        front_num = 0
        while len(new_population.population) + len(population.fronts[front_num]) <= pop_size_nsga:
            new_population.extend(population.fronts[front_num])
            front_num += 1
        Optimizer.calculate_crowding_distance(population.fronts[front_num])
        population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        new_population.extend(population.fronts[front_num][0: (pop_size_nsga - len(new_population))])
        #cpt=0
        for indiv in new_population:
            r = random.random()
            if r < 0.5:
                #cpt=cpt+1
                Optimizer.local_search(indiv, 0.1, communication_graph, list_target_points, coverage_graph)
                #print("from main algo indiv.coverage is ", indiv.coverage)
                nb_current_evaluation = nb_current_evaluation + 1
        population = new_population
        #print("nb indiv that has executed local search are ", cpt)
        #print("current nb evaluation ", nb_current_evaluation)
    Optimizer.fast_nondominated_sort(population)

    #print("time to execute nsga is ", time.time() - t_start)
    return population.fronts[0]


