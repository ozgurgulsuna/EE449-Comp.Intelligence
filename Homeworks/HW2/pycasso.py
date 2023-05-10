## Pycasso, an evolutionary algorithm for the automatic generation of paintings.
#      ________
#    /          \  __     __    _____       _        _____     _____    ___
#   |            | \ \   / /  /       \   /   \    / ____|   / ____|  /     \
#   |            |  \ \_/ /  /    ____|  /  ^  \  | (____   | (____  |       |
#   |    _______/    \   /   |   |      |  /_\  |  \____ \   \____ \ |       |
#   |  /              | |    |   |      | /   \ |  ____| |   ____| |  \___/
#   |/                |_|     \____ \   |/     \| |______|  |______|
#                                   \)
# Evolutainary algorithm for the automatic generation of paintings. 

import cv2
import numpy as np
import random
import math
import sys
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from datetime import datetime

from multiprocessing import Pool

# Global variables
num_inds = 100   # Individual Number
num_genes = 1000 # Gene Number
num_generations = 1000 # Generation Number

tm_size = 100 # Template Size
frac_elites = 0.1 # Fraction of elites
frac_parents = 0.1 # Fraction of parents
mutation_prob = 0.1 # Mutation probability
mutation_type = 0 # Mutation type


# Class for individual	
class Individual:
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness

# Class for gene
class Gene:
    def __init__(self, x, y, r, g, b, a):
        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b
        self.a = a

# Class for template
class Template:
    def __init__(self, x, y, r, g, b, a):
        self.x = x
        self.y = y
        self.r = r
        self.g = g
        self.b = b
        self.a = a

# Class for image
class Image:
    def __init__(self, width, height, genes):
        self.width = width
        self.height = height
        self.genes = genes

# Class for fitness
class Fitness:
    def __init__(self, image, template):
        self.image = image
        self.template = template

# Population initialization
def init_population():
    population = []
    for i in range(num_inds):
        genes = []
        for j in range(num_genes):
            x = random.randint(0, tm_size)
            y = random.randint(0, tm_size)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = random.random(0,1)
            genes.append(Gene(x, y, r, g, b, a))
        population.append(Individual(genes, 0))
    return population

# TODO: Drawing order for circles






































#       _____                _____            __  __  __
#      /\    \              /\    \          /\ \/ / /\ \
#     /::\    \            /::\____\        /::\  / /  \ \
#    /::::\    \          /:::/    /       /:/\:\/ /    \ \
#   /::::::\    \        /:::/    /       /:/  \:\_\____\ \
#  /:::/\:::\    \      /:::/    /       /:/    \:\/___/ \
# /:::/  \:::\    \    /:::/    /       /:/     \:\__\   _
# \::/    \:::\    \  /:::/    /       /:/      \/__/  /\ \
#  \/____/ \:::\    \/:::/    /        \/_____/      /::\ \
#           \:::\____\/:::/    /                      /:/\:\ \
#            \::/    /\::/    /                      /:/__\:\ \
#             \/____/  \/____/                       \:\   \:\ \
#                                                      \:\   \:\ \
#                                                       \:\   \:\_\
#                                                        \:\__\::/
#                                                         \/__\:/ 
