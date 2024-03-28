import init_env
import NSGAII as nsga
import MOGWO
from MOEAD import MOEAD
from Levenshtein import distance as levenshtein_distance
import MOPSO
import SPEAII
import NSGAII
#CONSTANTES
pop_size = 8
epoch= 5


#VARIABLES COMMUNES
nb_zones=0
nb_targets = 0
list_target_points = []
communication_graph = None
coordinates=[]
obstacles=[]

#VARIABLES NSGA-II

num_of_tour_particips= 2
tournament_prob =1
crossover_param = 1
mutation_param = 0.2

#VARIABLES MOEA/D
num_of_tour_particips= 2
tournament_prob =1
crossover_param = 1
mutation_param = 0.2
T=2

#VARIABLES SPEA-II
archive_size = 10

#VARIABLES MOPSO
nVar = 5  # number of decision vars
varMin = 0
varMax = 1
nRep = 5  # size of repository
w = 0.5  # inertia wieght
c1 = 2  # personal learning coefficient
c2 = 2  # global learning coefficient
wdamping = 0.99

#VARIABLES MOGWO
c=2


# ################ constriction coefficients
# phi1 = 2.05
# phi2 = 2.05
# phi = phi1+phi2
# chi = 2/(phi - 2 + np.sqrt(phi**2 - 4*phi))
# w = chi # inertia wieght
# c1 = chi*phi1 # personal learning coefficient
# c2 = chi*phi2 # global learning coefficient
# wdamping = 1
# #################

beta = 1  # leader selection pressure
gamma = 1  # deletion selection pressure
NoGrid = 5
alpha = 0.1  # nerkhe tavarrom grid
archive_size=3


#VARIABLES MOGWO









######################################### Archi 1 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi1()
#nsga.evolve(epoch, pop_size,  list_target_points, communication_graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)
#moea = MOEAD(epoch, pop_size,  list_target_points, communication_graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob, crossover_param, mutation_param, T)
#rep = SPEAII.evolve(epoch, pop_size,  list_target_points, communication_graph, nb_zones, coordinates, archive_size, nb_targets, num_of_tour_particips, tournament_prob, crossover_param, mutation_param)


"""
rep = NSGAII.evolve(epoch, list_target_points, communication_graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob,mutation_param, pop_size, crossover_param)
new_front=[]
for p in rep:
    new_front.append([(nb_targets - p.coverage), p.cost])
print(new_front)
for j in range (len(rep)):
    for i in range (len(rep)):
        if j != i and levenshtein_distance(rep[j].deployment, rep[i].deployment) == 0:
            print("distance is 0 and and it is not the same objects")

print("ok")
"""
#MOGWO.evolve(epoch, pop_size, nb_zones, communication_graph, coordinates, list_target_points, nb_targets,c, beta,NoGrid, nRep, gamma)
#MOPSO.evolve(epoch, pop_size, NoGrid, beta, w, c1, c2, nRep, gamma, c, nb_zones, communication_graph, coordinates, nb_targets, list_target_points)



######################################### Archi 2 ##################################################################################

#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi2()


######################################### Archi 3 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi3()


######################################### Archi 4 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi4()


######################################### Archi 5 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi5()


######################################### Archi 6 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi6()


######################################### Archi 7 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi7()


######################################### Archi 8 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi8()


######################################### Archi 9 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi9()



######################################### Archi 10 ##################################################################################
#nb_zones, nb_targets, list_target_points, communication_graph,  coordinates, obstacles = init_env.init_archi10()