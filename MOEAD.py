import random
import math
import Optimizer
from Individual  import Individual
import Connectivity_repair_SP as sp
from copy import deepcopy


class MOEAD():

    def __init__(self,epoch, list_target_points, graph, nb_zones, coordinates, num_of_tour_particips, tournament_prob, crossover_param):
        # Z vector (ideal point)
        self.z_ = []  # of type floats
        # Lambda vectors (Weight vectors)
        self.lambda_ = []  # of type list of floats, i.e. [][], e.g. Vector of vectors of floats
        # Neighbourhood size
        # Neighbourhood
        self.neighbourhood_ = []  # of type int, i.e. [][], e.g. Vector of vectors of integers
        self.paretoFront = []
        self.nb_objs = 2
        self.pop = []
        self.epoch = epoch
        self.list_target_points = list_target_points
        self.graph = graph
        self.nb_zones = nb_zones
        self.coordinates = coordinates
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param


    def evolve(self, pop_size, mutation_param, T_):
        T_ = int(T_)
        pop_size = int(pop_size)
        evaluations_ = 0
        # 2-D list of size populationSize * T_
        self.neighbourhood_ = [[None] * T_] * (pop_size)
        # List of size number of objectives. Contains best fitness.
        self.z_ = self.nb_objs * [100000000000]
        # 2-D list  Of size populationSize_ x Number of objectives. Used for weight vectors.
        self.lambda_ = [[None] * self.nb_objs] * pop_size
        target_covering_zones = Optimizer.covering_zones_for_each_target(self.coordinates, self.list_target_points)
        # STEP 1. Initialization
        self.pop = Optimizer.create_population_EA(pop_size,  self.nb_zones, self.graph, self.coordinates, self.list_target_points, target_covering_zones)
        self.initUniformWeight(pop_size)
        self.initNeighbourhood(pop_size, T_)

        for i in range(pop_size):
            self.updateReference(self.pop[i])

        while evaluations_ < self.epoch:
            for i in range(pop_size):
                child= self.reproduction(i, target_covering_zones, mutation_param, T_)
                self.updateReference(child)
                self.updateProblem(child, i, T_)
            evaluations_ += 1
            new_front_moea = Optimizer.coverage_cost_front(self.paretoFront)
            Optimizer.save_results('MOEAD-archi9', str(new_front_moea))
        return self.paretoFront

    def crossover(self, parent_1, parent_2):
         child_1 = Individual(parent_1.nb_zones)
         for i in range(len(parent_1.deployment)):
             if random.random() <= 0.5:
                 child_1.deployment[i] = parent_1.deployment[i]
             else:
                 child_1.deployment[i] = parent_2.deployment[i]

         return child_1


    def reproduction(self,i, target_covering_zones, mutation_param, T_):
        p=[]
        self.matingSelection(p, i, T_)
        child = self.crossover(self.pop[p[0]], self.pop[p[1]])
        if random.uniform(0, 1) < mutation_param:
            child = Optimizer.inversion_mutation(child)
        child1_dep = Optimizer.create_deployment_graph(child.deployment, self.graph)
        disjoint_sets = sp.distinct_connected_components(child1_dep)
        if len(disjoint_sets) > 1:
            sp.connectivity_repair_heuristic(disjoint_sets, child.deployment, self.graph, self.coordinates)
        child.coverage = Optimizer.calculate_coverage(child.deployment, self.list_target_points, target_covering_zones)
        child.cost = Optimizer.calculate_cost(child.deployment)
        return child

    """
    " initUniformWeight
    """
    def initUniformWeight(self, pop_size):
        for n in range(pop_size):
            a = 1.0 * float(n) / (pop_size - 1)
            self.lambda_[n][0] = a
            self.lambda_[n][1] = 1 - a

    """ 
    " initNeighbourhood
    """

    def initNeighbourhood(self, pop_size, T_):
        x = [None] * pop_size  # Of type float
        idx = [None] * pop_size  # Of type int

        for i in range(pop_size):
            for j in range(pop_size):
                x[j] = self.distVector(self.lambda_[i], self.lambda_[j])
                idx[j] = j

            self.minFastSort(x, idx, pop_size, T_)
            self.neighbourhood_[i][0:T_] = idx[0:T_]  # System.arraycopy(idx, 0, neighbourhood_[i], 0, T_)


    """
    " matingSelection
    """

    def matingSelection(self, vector, cid, T_):
        # vector : the set of indexes of selected mating parents
        # cid    : the id o current subproblem
        # size   : the number of selected mating parents
        # type   : 1 - neighborhood; otherwise - whole population
        """
        Selects 'size' distinct parents,
        either from the neighbourhood (type=1) or the populaton (type=2).
        """

        r = random.randint(0, T_ -1)
        p = self.neighbourhood_[cid][r]
        vector.append(p)
        r1 = random.randint(0, T_ - 1)
        while r1 == r:
            r1 = random.randint(0, T_ - 1)
        p = self.neighbourhood_[cid][r1]
        vector.append(p)




    """
    " updateReference
    " @param individual
    """

    def updateReference(self, individual):
        if (len(self.list_target_points) - individual.coverage) < self.z_[0]:
            self.z_[0] = (len(self.list_target_points) - individual.coverage)
        if individual.cost < self.z_[1]:
            self.z_[1] = individual.cost
    """
    " updateProblem
    " @param individual
    " @param id
    " @param type
    """

    def updateProblem(self, individual, id_, T_):
        """
        individual : A new candidate individual
        id : index of the subproblem
        type : update solutions in neighbourhood (type = 1) or whole population otherwise.
        """

        for i in range(T_):
            k = self.neighbourhood_[id_][i]
            f1 = self.fitnessFunction(self.pop[k], self.lambda_[k])
            f2 = self.fitnessFunction(individual, self.lambda_[k])
            if f2 < f1:
                self.pop[k] = deepcopy(individual)
        self.update_pareto_front(individual)


    def update_pareto_front(self, child):

        if len(self.paretoFront) > 0:
            self.paretoFront = [item for item in self.paretoFront if child.dominates(item)==False]
        dominated = False
        i = 0
        while i < len(self.paretoFront) and not dominated:
            if self.paretoFront[i].dominates(child):
                dominated = True
            i += 1
        if not dominated:
            self.paretoFront.append(child)





    """
    " fitnessFunction
    " @param individual
    " @param lambda_
    """

    def fitnessFunction(self, individual, lambda_):

        diff1 = abs((len(self.list_target_points) - individual.coverage) - self.z_[0])
        diff2 = abs(individual.cost - self.z_[1])
        if lambda_[0] == 0:
            feval1 = 0.0001 * diff1
        else:
            feval1 = diff1 * lambda_[0]
        if lambda_[1] == 0:
            feval2 = 0.0001 * diff2
        else:
            feval2 = diff2 * lambda_[1]

        return max(feval2, feval1)


    #######################################################################
    # Ported from the Utils.java class
    #######################################################################
    def distVector(self, vector1, vector2):
        dim = len(vector1)
        sum_ = 0
        for n in range(dim):
            sum_ += ((vector1[n] - vector2[n]) * (vector1[n] - vector2[n]))
        return math.sqrt(sum_)


    def minFastSort(self, x,  idx, n, m):

        """
        x   : list of floats
        idx : list of integers (each an index)
        n   : integer pop_size
        m   : integer neighbourhood size
        """

        for i in range(m):
            for j in range(i + 1, n):
                if x[i] > x[j]:
                    temp = x[i]
                    x[i] = x[j]
                    x[j] = temp
                    id_ = idx[i]
                    idx[i] = idx[j]
                    idx[j] = id_

