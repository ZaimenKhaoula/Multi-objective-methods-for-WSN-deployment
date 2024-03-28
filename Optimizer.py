from Individual import Individual, Population
from math import sqrt
import random
import numpy as np
import math
import Connectivity_repair_SP as sp
import PropagationModel as pm
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt


#CONSTANTES
Rc = 6
Rs = 4
Ru = 2.5
sensitivity = -94



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




def save_results(fichier, result):
    with open(fichier, 'a') as f:
        # Write a new line to the file
        f.write(result)
        f.write('\n')




def create_deployment_graph(solution, graph):
    index_of_deployed_sensors = [i for i in range(len(solution)) if solution[i] == 1]
    deployment_graph = graph.subgraph(index_of_deployed_sensors).copy()
    return deployment_graph


def covering_zones_for_each_target(coordinates, list_target_points):
    result=[]
    for target in list_target_points:
        lst=[]
        for i in range(len(coordinates)):
            if pm.Elfes_model(coordinates[i][0],coordinates[i][1],coordinates[target][0], coordinates[target][1], Rs, Ru):
                lst.append(i)
        result.append(lst)

    return result

def amend_position(solution, graph, coordinates):
    solution = sigmoid_transformation(solution)
    deployment_graph = create_deployment_graph(solution , graph)
    disjoint_sets = sp.distinct_connected_components(deployment_graph)
    if len(disjoint_sets) > 1:
        solution = sp.connectivity_repair_heuristic(disjoint_sets, solution,  graph, coordinates)
    return np.array(solution)

def covered(target_point, solution, coverage_graph):
    for i in range(len(solution)):
        if solution[i] == 1 and coverage_graph.has_edge(i, target_point):
            return True
    return False

def calculate_cost(solution):
    return np.count_nonzero(solution)




def calculate_coverage(deployment, covered_targets, list_target_points, coverage_graph):
    coverage = 0
    covered_targets.clear()
    for target in list_target_points:
        if covered(target, deployment, coverage_graph):
            coverage = coverage + 1
            covered_targets.append(target)
    return coverage



def costfunction(particle, nb_targets, list_target_points, coverage_graph, current_evaluation):
    current_evaluation = current_evaluation + 1
    coverage = calculate_coverage(particle.position, particle.covered_targets, list_target_points, coverage_graph)
    cost = calculate_cost(particle.position)
    particle.coverage = coverage
    particle.cost_value = cost
    particle.cost.clear()
    particle.cost.append((nb_targets - coverage))
    particle.cost.append(cost)
    return current_evaluation



class GridDim:
    LowerBounds = []
    UpperBounds = []


def FindGridIndex(particle, grid):
    nObj = 2
    NGrid = len(grid[0].LowerBounds)

    particle.gridSubIndex = np.zeros((1, nObj))[0]

    for j in range(nObj):
        index_in_Dim = len([item for item in grid[j].UpperBounds if particle.cost[j] > item])
        particle.gridSubIndex[j] = index_in_Dim


    particle.gridIndex = particle.gridSubIndex[0]

    for j in range(1, nObj):
        #particle.gridIndex = particle.gridIndex
        particle.gridIndex = NGrid * particle.gridIndex
        particle.gridIndex = particle.gridIndex + particle.gridSubIndex[j]

    return particle





def CreateGrid(Rep, ngrid, alpha):
    costs = [item.cost for item in Rep]
    nobj = 2
    Cmin = np.min(costs, axis=0)
    Cmax = np.max(costs, axis=0)
    deltaC = Cmax - Cmin
    Cmin = Cmin - alpha * deltaC
    Cmax = Cmax + alpha * deltaC
    grid = [GridDim() for p in range(nobj)]
    for i in range(nobj):
        dimValues = np.linspace(Cmin[i], Cmax[i], ngrid + 1).tolist()
        grid[i].LowerBounds = [-float('inf')] + dimValues
        grid[i].UpperBounds = dimValues + [float('inf')]

    return grid



def CreateGrid_ancien(pop, nGrid, alpha, nobj):
    costs = [item.cost for item in pop]
    Cmin = np.min(costs, axis=0)
    Cmax = np.max(costs, axis=0)
    deltaC = Cmax - Cmin
    Cmin = Cmin - alpha * deltaC
    Cmax = Cmax + alpha * deltaC
    grid = [GridDim() for p in range(nobj)]
    for i in range(nobj):
        dimValues = np.linspace(Cmin[i], Cmax[i], nGrid + 1).tolist()
        grid[i].LowerBounds = [-float('inf')] + dimValues
        grid[i].UpperBounds = dimValues + [float('inf')]

    return grid




def roulettewheelSelection(p):
    r = random.random()
    cumsum = np.cumsum(p)
    y = (cumsum < r)
    x = [i for i in y if i == True]
    return len(x)


def roulette_SPEA(pop):
    N = np.zeros(len(pop))
    for k in range(len(pop)):
        N[k] = pop[k].fitness
    # selection probablity
    #p = [math.exp(gamma * item) for item in N]
    p = np.array(N) / sum(N)

    # select cell index
    return roulettewheelSelection(p)


def SelectLeader(rep, beta=1):
    # Get occupied cells and member counts
    occ_cell_index, occ_cell_member_count = GetOccupiedCells(rep)

    # Calculate selection probabilities
    p = np.power(occ_cell_member_count, -beta)
    p = p / np.sum(p)

    # Select a cell index using RouletteWheelSelection
    #roulette = occ_cell_index[np.random.choice(len(occ_cell_index), p=p)]
    selected_cell_index = occ_cell_index[roulettewheelSelection(p)]

    # Assuming GridIndex is a list of grid indices
    GridIndices = [r.gridIndex for r in rep]

    # Find indices of selected_cell_index in GridIndices
    selected_cell_members = [i for i, index in enumerate(GridIndices) if index == selected_cell_index]

    n = len(selected_cell_members)

    # Select a random member index
    selected_member_index = np.random.randint(0, n)

    # Get the selected member
    h = selected_cell_members[selected_member_index]

    # Return the selected leader
    rep_h = rep[h]

    return rep_h





# Assuming Archive is a list of solutions with Position attribute
# and SelectLeader is a function to select a leader from a list of solutions
def choose_leaders(Archive):
    # Initialize variables for the leaders
    Delta = None
    Beta = None
    Alpha = None
    beta =4
    # Choose the alpha, beta, and delta grey wolves
    Delta = SelectLeader(Archive, beta)
    Beta = SelectLeader(Archive, beta)
    Alpha = SelectLeader(Archive, beta)
    rep2 = []

    # If there are more than one solution in the Archive,
    # find the second least crowded hypercube to choose another leader from.
    if len(Archive) > 1:
        counter = 0
        rep2 = []
        for newi in range(len(Archive)):
            if np.sum(Delta.position != Archive[newi].position) != 0:
                counter += 1
                rep2.append(Archive[newi])

        # Choose a leader (Beta) from rep2
        if len(rep2) > 0:
            Beta = SelectLeader(rep2, beta)

    # If there are more than two solutions in rep2,
    # find the third least crowded hypercube to choose Alpha leader from.
    if len(rep2) > 2:
        counter = 0
        rep3 = []
        for newi in range(len(rep2)):
            if np.sum(Beta.position != rep2[newi].position) != 0:
                counter += 1
                rep3.append(rep2[newi])

        # Choose a leader (Alpha) from rep3
        if len(rep3) > 0:
            Alpha = SelectLeader(rep3, beta)

    return Delta, Beta, Alpha



def GetOccupiedCells(pop):
    # Extract GridIndices from the pop object, assuming it's a list of objects with a GridIndex attribute
    GridIndices = [item.gridIndex for item in pop]
    # Find unique occ_cell_index values
    occ_cell_index = np.unique(GridIndices)

    # Initialize occ_cell_member_count with zeros
    occ_cell_member_count = np.zeros(len(occ_cell_index))

    m = len(occ_cell_index)

    # Count members for each occ_cell_index
    for k in range(m):
        occ_cell_member_count[k] = np.sum(GridIndices == occ_cell_index[k])

    return occ_cell_index, occ_cell_member_count


def deleteOneRepositoryMember(rep):
    gridindices = [item.gridIndex for item in rep]
    OCells = np.unique(gridindices)  # ocupied cells
    N = np.zeros(len(OCells))
    for k in range(len(OCells)):
        N[k] = gridindices.count(OCells[k])

    # Calculate p using NumPy element-wise exponentiation
    gamma = 2  # Replace with the desired value of gamma
    p = np.power(N, gamma)

    # Normalize p
    p = p / np.sum(p)

    # Select a cell index using RouletteWheelSelection
    selected_cell_index = OCells[np.random.choice(len(OCells), p=p)]

    # Assuming GridIndices is a list of grid indices
    GridIndices = [r.gridIndex for r in rep]

    # Find indices of selected_cell_index in GridIndices
    selected_cell_members = [i for i, index in enumerate(GridIndices) if index == selected_cell_index]

    n = len(selected_cell_members)

    # Select a random member index
    selected_member_index = np.random.randint(0, n)

    # Get the selected member index
    j = selected_cell_members[selected_member_index]

    # Remove the selected member from the rep list
    rep.pop(j)

    return  rep



def deleteOneRepositoryMember_ancien(rep):
    gamma = 2
    gridindices = [item.gridIndex for item in rep]
    OCells = np.unique(gridindices)  # ocupied cells
    N = np.zeros(len(OCells))
    for k in range(len(OCells)):
        N[k] = gridindices.count(OCells[k])
    # selection probablity
    #p = [math.exp(gamma * item) for item in N]
    p = np.power(np.array(N), gamma)
    p = p / sum(p)
    # select cell index

    sci = roulettewheelSelection(p)
    """
    while N[sci] == 0:
        sci = roulettewheelSelection(p)
    """

    SelectedCell = OCells[sci]
    """
    gindex=0
    for i in range(len(p)):
        if p[i] > p[gindex]:
            gindex = i
    SelectedCell = OCells[gindex]
    """
    # selected Cell members

    """
    selectedCellmembers = [item for item in gridindices if item == SelectedCell]

    
    selectedmemberindex = np.random.randint(0, len(selectedCellmembers))
    # selectedmember = selectedCellmembers[selectedmemberindex]
    # delete memeber
    # rep[selectedmemberindex] = []
    rep = np.delete(rep, selectedmemberindex)
    """
    selectedCellmembers = [item for item in rep if item.gridIndex == SelectedCell]
    ind = np.random.randint(0, len(selectedCellmembers))
    selectedmember = selectedCellmembers[ind]
    rep1=[]
    find = False
    for item in rep:
        if item.cost_value != selectedmember.cost_value or item.coverage != selectedmember.coverage:
            rep1.append(deepcopy(item))

        else:
            if find:
                rep1.append(deepcopy(item))
            find =True
    return rep1




def SelectLeaderGWO(rep, pop, nb_targets, nb_zones, beta):
    # if length rep is less or equal to 3 then return it as it is
    if len(rep) == 3:
        return rep[0], rep[1], rep[2]
    if len(rep) == 2:
        pop.sort(key=lambda x: (nb_targets - x.coverage)/ nb_targets + x.cost_value / nb_zones)
        return rep[0], rep[1], pop[0]

    if len(rep) == 1:
        pop.sort(key=lambda x: (nb_targets - x.coverage) / nb_targets + x.cost_value / nb_zones)
        return rep[0], pop[0], pop[1]

    if len(rep) == 0:
        pop.sort(key=lambda x: (nb_targets - x.coverage) / nb_targets + x.cost_value / nb_zones)
        return pop[0], pop[1], pop[2]
    if len(rep) > 3:
        rep1 = deepcopy(rep)
        leaders =[]

        while len(leaders) != 3:
            gridindices = [item.gridIndex for item in rep1]
            OCells = np.unique(gridindices)  # ocupied cells
            N = np.zeros(len(OCells))
            for k in range(len(OCells)):
                N[k] = gridindices.count(OCells[k])
            # selection probablity
            p = [math.pow(item, -beta) for item in N]
            p = np.array(p) / sum(p)
            # select cell index
            sci = roulettewheelSelection(p)
            SelectedCell = OCells[sci]
            # selected Cell members
            selectedCellmembers = [item for item in rep1 if item.gridIndex == SelectedCell]
            selectedmemberindex = np.random.randint(0, len(selectedCellmembers))
            leaders.append(deepcopy(selectedCellmembers[selectedmemberindex]))
            selected= False
            maintained=[]
            for item in rep1:
                if selected == True:
                    maintained.append(item)
                else :
                    if item.gridIndex == SelectedCell and item.coverage == selectedCellmembers[selectedmemberindex].coverage and item.cost_value == selectedCellmembers[selectedmemberindex].cost_value:
                        selected = True
                    else:
                        maintained.append(item)


            rep1 = maintained
        return leaders[0], leaders[1], leaders[2]





def SelectLeader_ancien(rep, beta):
    #print("len rep", len(rep))
    gridindices = [item.gridIndex for item in rep]
    OCells = np.unique(gridindices)  # ocupied cells
    N = np.zeros(len(OCells))
    for k in range(len(OCells)):
        N[k] = gridindices.count(OCells[k])
    # selection probablity
    p = [math.pow(item, -beta) for item in N]
    p = np.array(p) / sum(p)
    # select cell index
    sci = roulettewheelSelection(p)
    SelectedCell = OCells[sci]
    # selected Cell members
    selectedCellmembers = [item for item in rep if item.gridIndex == SelectedCell]
    selectedmemberindex = np.random.randint(0, len(selectedCellmembers))

    return  selectedCellmembers[selectedmemberindex]


def create_random_pos(nb_zones, communication_graph, coordinates):
    solution = [0, ] * nb_zones
    for i in range(nb_zones):
        if random.uniform(0, 1) > 0.75:
            solution[i] = 1
    deployment_graph = create_deployment_graph(solution, communication_graph)
    disjoint_sets = sp.distinct_connected_components(deployment_graph)
    if len(disjoint_sets) > 1:
        sp.connectivity_repair_heuristic(disjoint_sets, solution,  communication_graph, coordinates)
    return np.array(solution)



def sig(x):
 return 1/(1 + np.exp(-10*(x-0.5)))


def sigmoid_transformation(position):
    pos=[]
    for i in range(len(position)):
        if math.isclose(position[i], 0):
            pos.append(0)
        else:
            if math.isclose(position[i], 1):
                pos.append(1)
            else:
                r = random.random()
                s = sig(position[i])
                if s < r:
                    pos.append(1)
                else:
                    pos.append(0)
    if np.count_nonzero(pos)==0:
        pos[random.randint(0,len(position)-1)]=1

    return pos





def create_children_basic(population, coordinates, graph, list_target_points ,num_of_tour_particips, tournament_prob,
                          crossover_param, mutation_param, coverage_graph):
    children = []
    population = population.population
    while len(children) < len(population):
        parent1 = tournament(population, num_of_tour_particips, tournament_prob)
        parent2 = parent1
        while parent1 == parent2:
            parent2 = tournament(population, num_of_tour_particips, tournament_prob)
        if random.uniform(0,1) < crossover_param:
            child1, child2 = uniform_crossover(parent1, parent2)
            if random.uniform(0, 1) < mutation_param:
                child1 = inversion_mutation(child1)
                child2 = inversion_mutation(child2)
            child1_dep = create_deployment_graph(child1.deployment, graph)
            child2_dep = create_deployment_graph(child2.deployment, graph)
            disjoint_sets = sp.distinct_connected_components(child1_dep)
            if len(disjoint_sets) > 1:
                sp.connectivity_repair_heuristic(disjoint_sets, child1.deployment,  graph, coordinates)
            disjoint_sets = sp.distinct_connected_components(child2_dep)
            if len(disjoint_sets) > 1:
                sp.connectivity_repair_heuristic(disjoint_sets, child2.deployment, graph, coordinates)
            child1.coverage = calculate_coverage(child1.deployment, child1.covered_targets, list_target_points, coverage_graph)
            child2.coverage = calculate_coverage(child2.deployment, child2.covered_targets,  list_target_points, coverage_graph)
            child1.cost = calculate_cost(child1.deployment)
            child2.cost = calculate_cost(child2.deployment)
            children.append(child1)
            children.append(child2)
    return children


def mutation(indiv):
    ind= random.randint(0,len(indiv.deployment))
    if indiv.deployment[ind]==1:
        indiv.deployment[ind] =0
    else:
        indiv.deployment[ind] = 1


def tournament(population,num_of_tour_particips, tournament_prob):
    participants = random.sample(population, num_of_tour_particips)
    best = None
    for participant in participants:
        if best is None or (crowding_operator(participant, best) == 1 and choose_with_prob(tournament_prob)):
            best = participant

    return best

def choose_with_prob(prob):
    if random.random() <= prob:
        return True
    return False


def crowding_operator(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):

        return 1
    else:
        return -1



#Create initial population, graph represents communicatio graph between zones


def inversion_mutation(indiv):
    # Select a random subset of the chromosome
    start = random.randint(0, len(indiv.deployment) - 1)
    end = random.randint(start, len(indiv.deployment) - 1)
    subset = indiv.deployment[start:end+1]
    # Reverse the order of the values withifn the subset
    subset = [1 - bit for bit in subset]
    # Replace the original subset with the reversed subset
    indiv.deployment[start:end+1] = subset
    return indiv



def create_population(pop_size, nb_zones, graph, coordinates, list_target_points, coverage_graph):
    population = Population()
    for i in range(pop_size):
        indiv = Individual(nb_zones)
        indiv.deployment = create_random_pos(nb_zones, graph, coordinates)
        indiv.cost = calculate_cost(indiv.deployment)
        indiv.coverage = calculate_coverage(indiv.deployment, indiv.covered_targets,  list_target_points, coverage_graph)
        population.append(indiv)

    return population



def Dominates(x, y):
    x = np.array(x)
    y = np.array(y)
    x_dominate_y = all(x <= y) and any(x < y)
    return x_dominate_y





def DetermineDomination(pop):
    pop_len = len(pop)
    for i in range(pop_len):
        pop[i].IsDominated = False
    for i in range(pop_len):
        for j in range(i+1, pop_len):
            if Dominates(pop[i].cost, pop[j].cost):
                pop[j].IsDominated = True
            if Dominates(pop[j].cost, pop[i].cost):
                pop[i].IsDominated = True



def create_population_EA(pop_size, nb_zones,graph, coordinates, list_target_points, target_covering_zones):

    population = []
    for i in range(pop_size):
        indiv = Individual(nb_zones)
        indiv.deployment = create_random_pos(nb_zones, graph, coordinates)
        indiv.cost = calculate_cost(indiv.deployment)
        indiv.coverage = calculate_coverage(indiv.deployment, indiv.covered_targets, list_target_points, target_covering_zones)
        population.append(indiv)

    return population



def euclidean_distance(x0, y0, x1, y1):
    return math.sqrt(pow((x0-x1),2)+ pow((y0-y1),2))





#we do an assignment of the concatenation of deployment schemes of parent1 and parent2 without a deepcopy
def uniform_crossover(parent_1, parent_2):
    child_1 = Individual(parent_1.nb_zones)
    child_2 = Individual(parent_1.nb_zones)

    for i in range(len(parent_1.deployment)):
        if random.random() <= 0.5:
            child_1.deployment[i]= parent_1.deployment[i]
            child_2.deployment[i] = parent_2.deployment[i]
        else:
            child_1.deployment[i] = parent_2.deployment[i]
            child_2.deployment[i] = parent_1.deployment[i]
    return child_1, child_2


def fast_nondominated_sort(population):
    population.fronts = [[]]
    for individual in population.population:
        individual.dominated_solutions = []
        individual.domination_count = 0

    for individual in population.population:
        for other_individual in population:
            if other_individual != individual and individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1
        if individual.domination_count == 0:
            individual.rank = 0
            population.fronts[0].append(individual)
    i = 0
    while len(population.fronts[i]) > 0:
        temp = []
        for individual in population.fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)

        i = i + 1
        population.fronts.append(temp)


def calculate_crowding_distance(front):
    if len(front) > 0:
        solutions_num = len(front)
        front.sort(key=lambda individual: individual.coverage)
        front[0].crowding_distance = 10 ** 9
        front[solutions_num - 1].crowding_distance = 10 ** 9
        m_values = [individual.coverage for individual in front]
        scale = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        for i in range(1, solutions_num - 1):
            front[i].crowding_distance = 0
            front[i].crowding_distance += (front[i + 1].coverage - front[i - 1].coverage) / scale
        front.sort(key=lambda individual: individual.cost)
        front[0].crowding_distance = 10 ** 9
        front[solutions_num - 1].crowding_distance = 10 ** 9
        m_values = [individual.cost for individual in front]
        scale = max(m_values) - min(m_values)
        if scale == 0:
            scale = 1
        for i in range(1, solutions_num - 1):
            front[i].crowding_distance += (front[i + 1].cost - front[i - 1].cost) / scale


# compute strength + raw+ density of individuals
def compute_fitness_SPEA_II(population, nb_targets, archive):
    k = int(sqrt(len(population)+len(archive)))
    for i in range(len(population)):
        distance=[]
        population[i].strength=0
        population[i].dominated_solutions.clear()
        population[i].domination_count = 0
        population[i].fitneness = None
        population[i].dominators.clear()
        for j in range(len(population)):
            if i!= j:
                distance.append(euclidean_distance(population[i].cost, (nb_targets - population[i].coverage), population[j].cost,
                                       (nb_targets - population[j].coverage)))
                if population[i].dominates(population[j]):
                    population[i].strength = population[i].strength + 1

        for j in range(len(archive)):
            distance.append(euclidean_distance(population[i].cost, (nb_targets - population[i].coverage),
                                               archive[j].cost, (nb_targets - archive[j].coverage)))
            if population[i].dominates(archive[j]):
                population[i].strength = population[i].strength + 1

        distance.sort()
        population[i].density = (1 / (2 + distance[k]))

    for i in range(len(archive)):
        distance=[]
        archive[i].strength=0
        archive[i].dominated_solutions.clear()
        archive[i].domination_count = 0
        archive[i].fitneness = None
        archive[i].dominators.clear()
        for j in range(len(population)):
            distance.append(euclidean_distance(archive[i].cost, (nb_targets - archive[i].coverage), population[j].cost, (nb_targets - population[j].coverage)))
            if archive[i].dominates(population[j]):
                archive[i].strength = archive[i].strength +1
        for j in range(len(archive)):
            if i != j:
                distance.append(euclidean_distance(archive[i].cost, (nb_targets - archive[i].coverage),
                                                   archive[j].cost, (nb_targets - archive[j].coverage)))
                if archive[i].dominates(archive[j]):
                    archive[i].strength = archive[i].strength + 1

        distance.sort()
        archive[i].density = (1 / (2 + distance[k]))

    for i in range(len(population)):
        raw = 0
        for j in range(len(population)):
            if i!= j and population[j].dominates(population[i]):
                raw = raw + population[j].strength
        for j in range(len(archive)):
            if archive[j].dominates(population[i]):
                raw = raw + archive[j].strength
        population[i].fitness = raw + population[i].density

    for i in range(len(archive)):
        raw = 0
        for j in range(len(population)):
            if population[j].dominates(archive[i]):
                raw = raw + population[j].strength
        for j in range(len(archive)):
            if i!= j and archive[j].dominates(archive[i]):
                raw = raw + archive[j].strength
        archive[i].fitness = raw + archive[i].density



def none_dominated_solutions(population):
    non_dominated_sol = []
    for individual in population.population:
        if individual.domination_count == 0:
            individual.rank = 0
            non_dominated_sol.append(individual)

    return non_dominated_sol





def create_children_SPEA_II(pop_size,coordinates, graph, list_target_points,num_of_tour_particips, tournament_prob, crossover_param, mutation_param, target_covering_zones, archive):
    children = []
    population = archive
    while len(children) < pop_size:
        parent1 = roulette_SPEA(population)
        parent2 = parent1
        while parent1 == parent2:
            parent2 = roulette_SPEA(population)
        if random.uniform(0,1) < crossover_param:
            child1, child2 = uniform_crossover(population[parent1], population[parent2])
            if random.uniform(0, 1) < mutation_param:
                child1 = inversion_mutation(child1)
                child2 = inversion_mutation(child2)
            child1_dep = create_deployment_graph(child1.deployment, graph)
            child2_dep = create_deployment_graph(child2.deployment, graph)
            disjoint_sets = sp.distinct_connected_components(child1_dep)
            if len(disjoint_sets) > 1:
                sp.connectivity_repair_heuristic(disjoint_sets, child1.deployment,  graph, coordinates)
            disjoint_sets = sp.distinct_connected_components(child2_dep)
            if len(disjoint_sets) > 1:
                sp.connectivity_repair_heuristic(disjoint_sets, child2.deployment, graph, coordinates)
            child1.coverage = calculate_coverage(child1.deployment, child1.covered_targets, list_target_points, target_covering_zones)
            child2.coverage = calculate_coverage(child2.deployment, child2.covered_targets,  list_target_points, target_covering_zones)
            child1.cost = calculate_cost(child1.deployment)
            child2.cost = calculate_cost(child2.deployment)
            children.append(child1)
            children.append(child2)
    return children



def archive_update_SPEA_II(archive, population, archive_size, nb_targets):
    union_pop_archive=[]
    union_pop_archive.extend(population)
    union_pop_archive.extend(archive)
    new_archive=[]
    for indiv in union_pop_archive:
        if indiv.fitness < 1:
            new_archive.append(deepcopy(indiv))

    if len(new_archive) < archive_size:
        union_pop_archive.sort(key=lambda x: x.fitness)
        i=len(new_archive)
        while len(new_archive) < archive_size:
            new_archive.append(deepcopy((union_pop_archive[i])))
            i=i+1
    if len(new_archive) > archive_size:
        while len(new_archive) > archive_size:
            distance = float('inf')
            remove = None
            for i in range(len(new_archive)):
                dist=0
                for j in range(len(new_archive)):
                    if i !=j:
                        dist = dist + euclidean_distance(new_archive[i].cost/len(new_archive[i].deployment), (nb_targets - new_archive[i].coverage)/nb_targets, new_archive[j].cost/len(new_archive[j].deployment), (nb_targets - new_archive[j].coverage)/nb_targets)
                if dist < distance:
                    distance = dist
                    remove = i
            new_archive.pop(remove)

    return new_archive


def targets_covered_by_a_sensor(sensor, target_covering_zones):
    list_covered_targets = []
    for target in range(len(target_covering_zones)):
        if sensor in target_covering_zones[target]:
            list_covered_targets.append(target)
    return list_covered_targets

def new_dominated_sensor(sensor, deployment, coverage_graph, list_target_points):
    neigbors = list(coverage_graph.neighbors(sensor))
    targets= [t for t in list_target_points if t in neigbors]
    for target in targets:
        dominated = False
        for i in range(len(deployment)):
            if deployment[i] == 1 and i != sensor and coverage_graph.has_edge(target, i):
                dominated = True
            if dominated:
                break
        if dominated == False:
            return False
    return True



def dominated_sensor(sensor, deployment,list_target_points, target_covering_zones ):
    targets = targets_covered_by_a_sensor(sensor, target_covering_zones)
    for target in targets:
        dominated = False
        for i in range(len(deployment)):
            if deployment[i] == 1 and i != sensor and i in target_covering_zones[target]:
                dominated = True
            if dominated:
                break
        if dominated == False:
            return False
    return True

import time

def local_search(indiv, rate, communication_graph,list_target_points, coverage_graph):
    iter = int(rate * len(indiv.deployment))
    #print("cost before ", indiv.cost)
    #print("coverage before ", indiv.coverage)
    #print("enter local search ")
    #t1=time.time()
    #tcoverage = 0
    deployment_graph = create_deployment_graph(indiv.deployment, communication_graph)
    for i in range(iter):
        cut_vertices = list(nx.articulation_points(deployment_graph))
        all_nodes = list(nx.nodes(deployment_graph))
        if len(all_nodes) > 0:
            sensor_to_move = random.choice(all_nodes)
            neighbring_sensors = list(deployment_graph.neighbors(sensor_to_move))
            if len(neighbring_sensors) > 0:
                #tk=time.time()
                neighbring_coverage = list(coverage_graph.neighbors(sensor_to_move))
                covered_targets_by_sensor = list(set(neighbring_coverage) & set(indiv.covered_targets))
                targets_covered_only_by_sensor = []
                for target in covered_targets_by_sensor:
                    find=False
                    for i in range(len(indiv.deployment)):
                        if indiv.deployment[i] == 1 and i != sensor_to_move and coverage_graph.has_edge(target, i):
                            find= True
                            break
                    if find == False:
                        targets_covered_only_by_sensor.append(target)

                #tk=time.time()-tk
                if sensor_to_move in cut_vertices:
                    intersect=[]
                    for neighbor in neighbring_sensors:
                        c_zones = [item for item in list(communication_graph.neighbors(neighbor)) if
                                   indiv.deployment[item] == 0]
                        intersect.append(c_zones)
                    candidate_zones = list(set(intersect[0]) & set(intersect[1]))
                    if len(intersect) > 2:
                        for i in range(2, len(intersect)):
                            candidate_zones = list(set(candidate_zones) & set(intersect[i]))
                else:
                    candidate_zones = []
                    for neighbor in neighbring_sensors:
                        c_zones = [item for item in list(communication_graph.neighbors(neighbor)) if
                                   indiv.deployment[item] == 0]
                        candidate_zones.extend(c_zones)
                if len(candidate_zones) > 0:
                    random.shuffle(candidate_zones)
                    #tcoverage =  tk
                    for pos in candidate_zones:
                        #ta=time.time()
                        neighbring_coverage = list(coverage_graph.neighbors(pos))
                        lst= list(set(neighbring_coverage) & set(list_target_points))
                        covered_targets_by_pos = [target for target in lst if target not in indiv.covered_targets]
                        #print("len covered targets ", len(indiv.covered_targets))
                        #tcoverage= tcoverage + (time.time()-ta)
                        #co1=calculate_coverage(indiv.deployment, indiv.covered_targets, list_target_points, coverage_graph)
                        if len(covered_targets_by_pos) > len(targets_covered_only_by_sensor):
                            #print("len targets of pos ", len(covered_targets_by_pos))
                            #print("len targets sensor to move ", len(targets_covered_only_by_sensor))
                            indiv.deployment[sensor_to_move] = 0
                            indiv.deployment[pos] = 1
                            deployment_graph = create_deployment_graph(indiv.deployment, communication_graph)
                            indiv.covered_targets.extend(covered_targets_by_pos)
                            indiv.covered_targets=[target for target in indiv.covered_targets if target not in targets_covered_only_by_sensor ]
                            #cov=calculate_coverage(indiv.deployment, indiv.covered_targets, list_target_points, coverage_graph)
                            #print("coverage before moving ", co1)
                            #print("coverage after moving ", cov)
                            break
        else:
            break

    #print("time to compute coverage of all movements is ", tcoverage)
    #print("time to execute coverage part ", time.time()-t1)
    #t2=time.time()
    #co1 = calculate_coverage(indiv.deployment, indiv.covered_targets, list_target_points, coverage_graph)
    deployment_graph = create_deployment_graph(indiv.deployment, communication_graph)
    cut_vertices = list(nx.articulation_points(deployment_graph))
    for j in range(len(indiv.deployment)):
        if indiv.deployment[j] == 1:
            if j not in cut_vertices and new_dominated_sensor(j, indiv.deployment , coverage_graph, list_target_points):
                    indiv.deployment[j] = 0
                    deployment_graph.remove_node(j)
                    cut_vertices = list(nx.articulation_points(deployment_graph))

    indiv.cost = calculate_cost(indiv.deployment)
    #print("time to execute cost part ", time.time() - t2)
    indiv.coverage= calculate_coverage(indiv.deployment, indiv.covered_targets, list_target_points, coverage_graph)
    #print("cost after ", indiv.cost)
    #print("coverage after ", indiv.coverage)

    #print("time to execute local search ", time.time() - t1)





