import time
from Individual import Population
import Optimizer




def evolve(max_eval, list_target_points, communication_graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob, mutation_param, pop_size, crossover, coverage_graph):
    nb_current_evaluation = 0
    pop_size = int(pop_size)
    #target_covering_zones = Optimizer.covering_zones_for_each_target(coordinates, list_target_points)
    population = Optimizer.create_population(pop_size, nb_zones, communication_graph, coordinates, list_target_points, coverage_graph)
    nb_current_evaluation = nb_current_evaluation + pop_size
    while nb_current_evaluation < max_eval:
        Optimizer.fast_nondominated_sort(population)
        #new_front_nsga = Optimizer.coverage_cost_front(population.fronts[0])
        #Optimizer.save_results('nsga_new_archi8', str(new_front_nsga))
        for front in population.fronts:
            Optimizer.calculate_crowding_distance(front)
        children = Optimizer.create_children_basic(population, coordinates, communication_graph, list_target_points,
                                                   num_of_tour_particips, tournament_prob, crossover,
                                                   mutation_param, coverage_graph)
        nb_current_evaluation = nb_current_evaluation + pop_size
        population.population.extend(children)
        Optimizer.fast_nondominated_sort(population)
        new_population = Population()
        front_num = 0
        while len(new_population.population) + len(population.fronts[front_num]) <= pop_size:
            new_population.extend(population.fronts[front_num])
            front_num += 1
        Optimizer.calculate_crowding_distance(population.fronts[front_num])
        population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        new_population.extend(population.fronts[front_num][0: (pop_size - len(new_population))])
        population = new_population

    Optimizer.fast_nondominated_sort(population)
    #new_front_nsga = Optimizer.coverage_cost_front(population.fronts[0])
    #Optimizer.save_results('nsga-archi10_450', str(new_front_nsga))
    return population.fronts[0]