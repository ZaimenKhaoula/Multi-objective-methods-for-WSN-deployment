import numpy as np
import networkx as nx
import Connectivity_repair_SP as sp
import random
from scipy.spatial import distance
import time
import math

class Individual:

    def __init__(self,nb_zones):
        self.deployment = [0] * nb_zones
        self.nb_zones = nb_zones
        self.rank=None
        self.coverage = None
        self.cost = None
        self.crowding_distance=0
        self.domination_count = 0
        self.dominated_solutions = []
        self.dominators = []
        self.strength= None # number of solutions the indiv dominates
        self.raw = None # the sum of the strength of  all its dominators that should be minimized
        #additional density information is incorporated to discriminate between individuals having identical raw fitness values.
        self.density = None
        # the individual fitness in SPEA-II is the sum of density and raw
        self.fitness = None
        self.covered_targets=[]


    def dominates(self, indiv):
        and_condition = True
        or_condition = False
        and_condition = and_condition and self.cost <= indiv.cost
        or_condition = or_condition or self.cost < indiv.cost
        and_condition = and_condition and self.coverage >= indiv.coverage
        or_condition = or_condition or self.coverage > indiv.coverage
        return and_condition and or_condition



    def none_dominated_point(self, point):
        for target_point in self.list_covered_targets:
            s = [i + target_point for i in self.mask_coordinates_Rs]
            if point in s:
                cpt = sum(self.deployment[i] for i in s if i >= 0 and i < self.nb_zones)
                if cpt > 1:
                    return True
        return False



    def generate_random_solution(self):
        self.deployment= np.random.randint(2, size=self.nb_zones)

    def distance(self,ax,bx):
        return math.sqrt(pow(ax[0]-bx[0], 2)+ pow(ax[1]-bx[1], 2))




class Population:

    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)