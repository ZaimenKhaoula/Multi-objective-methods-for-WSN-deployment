from pymoo.indicators.hv import HV
import numpy as np
import Optimizer
import init_env
import NSGAII as nsga
import MOGWO
from MOEAD import MOEAD
import MOPSO
import SPEAII
import hybrid_mogwo_nsga
import hybridWithLocalSearch
import os
import networkx as nx
import Connectivity_repair_SP as sp

num_of_tour_particips= 2
tournament_prob =1
crossover_param = 1
epoch=450


nb_run = 9
######################## Algorithms parameters computed by irace #############################

######### NSGA-II #############
mutation_param_nsga = 0.0890
pop_size_nsga =  133
crossover_nsga = 0.8742

######### SPEA-II #############

crossover_param_spea = 0.8620
mutation_param_spea = 0.0589
pop_size_spea = 139
#archive_size_spea = 40
archive_size_spea = 139

######### MOEA/D #############
pop_size_moea = 101
mutation_param_moea = 0.0254
T_ = 20


######### MOPSO #############
pop_size_pso = 94
NoGrid_pso = 48
c1 = 1.9089
c2 = 1.9260
nRep_pso = 49


######### MOGWO #############

NoGrid_mogow = 46
nRep_mogwo = 49
pop_size_mogwo = 96



def compute_hypervolume(front, nb_targets, nb_zones):
    new_front = []
    nadir = [nb_targets, nb_zones]
    for p in front:
        new_front.append([(nb_targets - p.coverage), p.cost])
    hv_calculator = HV(ref_point=np.array(nadir))
    return hv_calculator(np.array(new_front))


def coverage_cost_front(front):
    new_front = []
    for p in front:
        new_front.append([p.coverage, p.cost])
    return new_front


def coverage_cost_front_swarm(front):
    new_front = []
    for p in front:
        new_front.append([p.coverage, p.cost_value])
    return new_front

# Objective function
i=0
nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = init_env.init_archi8()
print("nb nodes ",len(nx.nodes(communication_graph)))
print("nb zones ",nb_zones )
max_evaluations_archi8 =100*0.25*nb_zones
disjoint_sets = sp.distinct_connected_components(communication_graph)
if len(disjoint_sets) > 1:
    print("archi 8 communication graph not connected nb disjoint sets is ", len(disjoint_sets))
epoch = 450
#CONSTANTES
Rc = 6
Rs = 4
Ru = 2.5
sensitivity = -94
print("generating coverage graph")
coverage_graph= sp.generate_coverage_graph(coordinates, Rs, Ru)
print("finish generating coverage graph")
nb_run=2

for i in range(nb_run):
    results = []
    # Instance 1
    print("nb run", i)
    print("local")
    front = hybridWithLocalSearch.evolve(max_evaluations_archi8, nb_zones, communication_graph, coordinates, list_target_points, nb_targets,
                                     NoGrid_mogow, nRep_mogwo, pop_size_nsga, pop_size_mogwo, num_of_tour_particips,
                                     tournament_prob, mutation_param_nsga, crossover_nsga, coverage_graph)

    new_front_local = Optimizer.coverage_cost_front(front)
    Optimizer.save_results('new-local-archi8-9', str(new_front_local))

    #########################  HNSGA  #############################
    print("hybrid")
    #front = hybrid_mogwo_nsga.evolve(max_evaluations_archi8, nb_zones, communication_graph, coordinates, list_target_points, nb_targets,
                                     #NoGrid_mogow, nRep_mogwo, pop_size_nsga, pop_size_mogwo, num_of_tour_particips,
                                     #tournament_prob, mutation_param_nsga, crossover_nsga, coverage_graph)

    #new_front_local = Optimizer.coverage_cost_front(front)
    #Optimizer.save_results('new-hybrid-archi8-2', str(new_front_local))



    #########################  MOGWO  #############################
    print("mogwo")

    #front_mogwo = MOGWO.evolve(max_evaluations_archi8, nb_zones, communication_graph, coordinates, list_target_points,nb_targets, NoGrid_mogow, nRep_mogwo, pop_size_mogwo, coverage_graph)
    #new_front_mogwo = Optimizer.coverage_cost_front_swarm(front_mogwo)
    #Optimizer.save_results('new-mogwo-archi8-2', str(new_front_mogwo))



    #########################  NSGA-II  #############################
    print("nsga-ii")

    #front_nsga = nsga.evolve(max_evaluations_archi8, list_target_points, communication_graph, nb_zones, coordinates,num_of_tour_particips, tournament_prob, mutation_param_nsga, pop_size_nsga, crossover_nsga, coverage_graph)
    #new_front_nsga = Optimizer.coverage_cost_front(front_nsga)
    #Optimizer.save_results('new-nsga-archi8-2', str(new_front_nsga))



    #########################  MOGWO  #############################
    #print("mogwo")

    #front_mogwo = MOGWO.evolve(max_evaluations_archi8, nb_zones, communication_graph, coordinates, list_target_points,
                               #nb_targets, NoGrid_mogow, nRep_mogwo, pop_size_mogwo)
    #new_front_mogwo = Optimizer.coverage_cost_front_swarm(front_mogwo)
    #Optimizer.save_results('mogwo_archi8-1', str(new_front_mogwo))

    #########################  NSGA-II  #############################
    #print("nsga-ii")

    #front_nsga = nsga.evolve(max_evaluations_archi8, list_target_points, communication_graph, nb_zones, coordinates,num_of_tour_particips, tournament_prob, mutation_param_nsga, pop_size_nsga, crossover_nsga)
    #new_front_nsga = Optimizer.coverage_cost_front(front_nsga)
    #Optimizer.save_results('nsga_archi8-1', str(new_front_nsga))

    #########################  SPEA-II  #############################
    #print("spea-ii")

    #front_spea = SPEAII.evolve(max_evaluations_archi8, list_target_points, communication_graph, nb_zones, coordinates,nb_targets, num_of_tour_particips,
                               #tournament_prob, crossover_param_spea, mutation_param_spea, pop_size_spea, archive_size_spea)
    #new_front_spea = Optimizer.coverage_cost_front(front_spea)
    #Optimizer.save_results('spea_archi8-1', str(new_front_spea))

    #########################  MOPSO #############################
    #print("mopso")

    #front_pso = MOPSO.evolve(max_evaluations_archi8, nb_zones, communication_graph, coordinates, nb_targets,list_target_points, pop_size_pso, NoGrid_pso, c1, c2, nRep_pso)
    #new_front_mopso = Optimizer.coverage_cost_front_swarm(front_pso)
    #Optimizer.save_results('mopso_archi8-1', str(new_front_mopso))