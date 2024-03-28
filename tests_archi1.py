from pymoo.indicators.hv import HV
import numpy as np
import Optimizer
import NSGAII as nsga
import MOGWO
import init_env
from MOEAD import MOEAD
import MOPSO
import SPEAII
import hybrid_mogwo_nsga
import hybridWithLocalSearch
import os
import Connectivity_repair_SP as sp


num_of_tour_particips= 2
tournament_prob =1
crossover_param = 1


nb_run = 1


######################## Algorithms parameters computed by irace #############################

######### NSGA-II #############
mutation_param_nsga = 0.0890
pop_size_nsga =  133
crossover_nsga = 0.8742

######### SPEA-II #############

crossover_param_spea = 0.8620
mutation_param_spea = 0.0589
pop_size_spea = 139
archive_size_spea = 139
#archive_size_spea= 40 pop size 139

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
nb_zones, nb_targets, list_target_points, communication_graph, coordinates, obstacles = init_env.init_archi1()

max_evaluations_archi1 =100*2*nb_zones

epoch =450
#CONSTANTES
Rc = 6
Rs = 4
Ru = 2.5
sensitivity = -94

print("generating coverage graph")
coverage_graph= sp.generate_coverage_graph(coordinates, Rs, Ru)
print("finish generating coverage graph")


for i in range(nb_run):
    results = []
    front = hybrid_mogwo_nsga.evolve(max_evaluations_archi1, nb_zones, communication_graph, coordinates, list_target_points, nb_targets,
                                     NoGrid_mogow, nRep_mogwo,  pop_size_nsga, pop_size_mogwo, num_of_tour_particips,
                                     tournament_prob, mutation_param_nsga, crossover_nsga, coverage_graph)


    #new_front_local = Optimizer.coverage_cost_front(front)
    #print(new_front_local)

    #########################  MOGWO  #############################
    #print("executing MOGWO")

    #front_mogwo = MOGWO.evolve(max_evaluations_archi1, nb_zones, communication_graph, coordinates, list_target_points, nb_targets, NoGrid_mogow, nRep_mogwo, pop_size_mogwo, coverage_graph)
    #new_front_mogwo = Optimizer.coverage_cost_front_swarm(front_mogwo)
    #Optimizer.save_results('mogwo_archi1', str(new_front_mogwo))
    #print(new_front_mogwo)
    #########################  NSGA-II  #############################


    #print("executing NSGA-II")

    front_nsga = nsga.evolve(max_evaluations_archi1, list_target_points, communication_graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob,  mutation_param_nsga, pop_size_nsga, crossover_nsga, coverage_graph)
    new_front_nsga = Optimizer.coverage_cost_front(front_nsga)
    #Optimizer.save_results('nsga_archi1', str(new_front_nsga))
    print(new_front_nsga)

    #########################  SPEA-II  #############################
    #print("executing SPEA-II")

    #front_spea = SPEAII.evolve(max_evaluations_archi1, list_target_points, communication_graph, nb_zones, coordinates, nb_targets, num_of_tour_particips,
                               #tournament_prob, crossover_param_spea, mutation_param_spea ,pop_size_spea, archive_size_spea)
    #new_front_spea = Optimizer.coverage_cost_front(front_spea)
    #Optimizer.save_results('spea_archi1', str(new_front_spea))


    #########################  MOPSO #############################
    #print("executing MOPSO")
    #front_pso = MOPSO.evolve(max_evaluations_archi1, nb_zones, communication_graph, coordinates, nb_targets, list_target_points, pop_size_pso, NoGrid_pso, c1, c2, nRep_pso)
    #new_front_mopso = Optimizer.coverage_cost_front_swarm(front_pso)
    #Optimizer.save_results('mopso_archi1', str(new_front_mopso))