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
# import math
# import sys
# import os

# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont

# from datetime import datetime

# from multiprocessing import Pool

# Global variables
source_image_path = "images/"
source_image_name = "cafe_terrace_at_night.png"
source_image = cv2.imread(source_image_path + source_image_name)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




num_inds = 100   # Individual Number
num_genes = 1000 # Gene Number
num_generations = 1000 # Generation Number

tm_size = 100 # Tournament size
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
    def __init__(self, x, y, s, r, g, b, a):
        self.x = x
        self.y = y
        self.s = s
        self.r = r
        self.g = g
        self.b = b
        self.a = a

# Class for template
class Template:
    def __init__(self, x, y, s, r, g, b, a):
        self.x = x
        self.y = y
        self.s = s
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

# Individual Evaluation
def evaluate_individual(individual):
    image = np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8)
    image.fill(255)
    for gene in individual.genes:
        cv2.circle(image, gene.x, gene.y, gene.s, gene.r, gene.g, gene.b, gene.a)
    fitness = 0
    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            fitness += abs(source_image[i][j][0] - image[i][j][0])
            fitness += abs(source_image[i][j][1] - image[i][j][1])
            fitness += abs(source_image[i][j][2] - image[i][j][2])
    individual.fitness = fitness
    return individual

# TODO: Drawing order for circles


Gene.x = 0
Gene.y = 0
Gene.s = 100
Gene.r = 0
Gene.g = 0
Gene.b = 0
Gene.a = 0.05


image = np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8)
image.fill(255)
overlay = image.copy()
cv2.circle(overlay, (Gene.x, Gene.y), Gene.s, (Gene.r, Gene.g, Gene.b), -1)
image = cv2.addWeighted(overlay, Gene.a, image, 1 - Gene.a, 0)





# image = Image(source_image.shape[0], source_image.shape[1], np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()






























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
