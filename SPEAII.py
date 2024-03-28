import time
from Individual import Population
import Optimizer
from Levenshtein import distance as levenshtein_distance
from copy import deepcopy


def evolve(max_evaluations, list_target_points, graph, nb_zones, coordinates, nb_targets, num_of_tour_particips, tournament_prob, crossover_param, mutation_param,pop_size, archive_size):
    nb_current_evaluation = 0
    pop_size = int(pop_size)
    archive_size = int(archive_size)
    target_covering_zones = Optimizer.covering_zones_for_each_target(coordinates, list_target_points)
    population = Optimizer.create_population_EA(pop_size, nb_zones, graph, coordinates, list_target_points, target_covering_zones)
    archive = []

    while nb_current_evaluation < max_evaluations:
        Optimizer.compute_fitness_SPEA_II(population, nb_targets, archive)
        nb_current_evaluation = nb_current_evaluation + pop_size + len(archive)
        archive = Optimizer.c(archive, population, archive_size,nb_targets)
        #new_front_spea = Optimizer.coverage_cost_front(archive)
        #Optimizer.save_results('spea-archi9_450', str(new_front_spea))
        population = Optimizer.create_children_SPEA_II(pop_size,coordinates, graph, list_target_points ,num_of_tour_particips, tournament_prob, crossover_param, mutation_param, target_covering_zones, archive)

    # select parents from archive
    #Optimizer.compute_fitness_SPEA_II(population, nb_targets, archive)
    #population.extend(archive)
    #archive = [deepcopy(item) for item in population if item.fitness < 1]
    #new_front_spea = Optimizer.coverage_cost_front(archive)
    #Optimizer.save_results('spea-archi9_450', str(new_front_spea))

    return archive