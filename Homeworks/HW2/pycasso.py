## Pycasso, an evolutionary algorithm for the automatic generation of paintings.
#      ________
#    /          \  __     __    _____       _        _____     _____    ___
#   |            | \ \   / /  /       \   /   \    / ____|   / ____|  /     \
#   |            |  \ \_/ /  /    ____|  /  ^  \  | (____   | (____  |       |
#   |    _______/    \   /   |   |      |  /_\  |  \____ \   \____ \ |       |
#   |  /              | |    |   |      | /   \ |  ____| |   ____| |  \___/
#   |/                |_|     \____ \   |/     \| |______|  |______|
#                                   \)
#
# https://www.petercollingridge.co.uk/blog/evolving-images/


import cv2
import numpy as np
import random
import math
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
h = source_image.shape[0]
w = source_image.shape[1]
s_max = int(math.sqrt(h**2+w**2))   # Maximum circle size, diagonal of the image, radius
h_margin = 1*h               # Horizontal margin
w_margin = 1*w               # Vertical margin
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




num_inds = 10   # Individual Number
num_genes = 10 # Gene Number
num_generations = 10 # Generation Number

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

# Class for image
class Image:
    def __init__(self, width, height, genes):
        self.width = width
        self.height = height
        self.genes = genes

# Population initialization
def init_population():
    population = []
    for i in range(num_inds):
        genes = []
        for j in range(num_genes):
            x = random.randint(-h_margin, h+h_margin)
            y = random.randint(-w_margin, w+w_margin)
            s = random.randint(0, s_max)
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = random.uniform(0,1)
            genes.append(Gene(x, y, s, r, g, b, a))
        population.append(Individual(genes, 0))
    print("Population initialized, checking collisions...")

    # Check for collisions
    for individual in population:
        for gene in individual.genes:
            while not check_circle(gene):
                gene.x = random.randint(-h_margin, h+h_margin)
                gene.y = random.randint(-w_margin, w+w_margin)
                gene.s = random.randint(0, s_max)
    print("Collisions checked, Sorting population...")

    # Sort population by size
    for individual in population:
        for gene in individual.genes:
            try:
                sorted(gene, key=lambda x: x.s)
            except:
                # print("Error sorting population")
                pass
    print("Population sorted, returning...")

    # for individual in population:
    #     for gene in individual.genes:
    #         print(gene.s)
    
    return population

# Check if a circle is inside the image
# https://stackoverflow.com/questions/75231142/collision-detection-between-circle-and-rectangle

## First method is crude, do dot capture all cases. (bounding box)
# def check_circle(gene):
#     if gene.x + gene.s > 0 and gene.x - gene.s < h and gene.y + gene.s > 0 and gene.y - gene.s < w:
#         return True
#     else:
#         return False

# Second method is more accurate, but slower. (distance)
def check_circle(gene):
    # nearest point on rectangle to circle
    nearest_x = max(0, min(gene.x, h))
    nearest_y = max(0, min(gene.y, w))

    # distance from circle center to nearest point
    dx = nearest_x - gene.x 
    dy = nearest_y - gene.y 

    # if distance is less than circle radius, there is a collision
    return (dx**2 + dy**2) < (gene.s**2)



# Individual Evaluation
def evaluate_individual(individual):
    image = np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8)
    image.fill(255)
    for gene in individual.genes:
        overlay = image.copy()
        cv2.circle(overlay, (gene.x, gene.y), gene.s, (gene.r, gene.g, gene.b), -1)
        image = cv2.addWeighted(overlay, gene.a, image, 1 - gene.a, 0)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    fitness = 0

    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            fitness += (source_image[i][j][0] - image[i][j][0])^2
            fitness += (source_image[i][j][1] - image[i][j][1])^2
            fitness += (source_image[i][j][2] - image[i][j][2])^2
    individual.fitness = -1*fitness
    return individual

# TODO: Drawing order for circles

# Tournament selection
def tournament_selection(population):
    tournament = []
    for i in range(tm_size):
        tournament.append(random.choice(population))
    best = tournament[0]
    for i in range(tm_size):
        if tournament[i].fitness > best.fitness:
            best = tournament[i]
    return best

# Elitism
def elitism(population):
    elites = []
    for i in range(int(frac_elites*num_inds)):
        best = population[0]
        for j in range(num_inds):
            if population[j].fitness > best.fitness:
                best = population[j]
        elites.append(best)
        population.remove(best)
    return elites

# Parent selection
def parent_selection(population):
    parents = []
    for i in range(int(frac_parents*num_inds)):
        parents.append(tournament_selection(population))
    return parents

# Crossover
def crossover(parents):
    children = []
    for i in range(int(frac_parents*num_inds)):
        child_genes = []
        for j in range(num_genes):
            if random.random() < 0.5:
                child_genes.append(parents[i].genes[j])
            else:
                child_genes.append(parents[i+1].genes[j])
        children.append(Individual(child_genes, 0))
    return children

# Mutation
def mutation(children):
    for child in children:
        for gene in child.genes:
            if random.random() < mutation_prob:
                if mutation_type == 0:
                    gene.x = random.randint(0, tm_size)
                    gene.y = random.randint(0, tm_size)
                elif mutation_type == 1:
                    gene.s = random.randint(0, tm_size)
                elif mutation_type == 2:
                    gene.r = random.randint(0, 255)
                    gene.g = random.randint(0, 255)
                    gene.b = random.randint(0, 255)
                elif mutation_type == 3:
                    gene.a = random.random(0,1)
    return children

# Main
def main():
    population = init_population()
    return population
    # for i in range(num_generations):
    #     for individual in population:
    #         evaluate_individual(individual)
    #     elites = elitism(population)
    #     parents = parent_selection(population)
    #     children = crossover(parents)
    #     children = mutation(children)
    #     population = elites + children
    # best = population[0]
    # for individual in population:
    #     if individual.fitness > best.fitness:
    #         best = individual
    # return best

# Run
best_case = main()
# print(best_case[0].fitness)
# print(best_case.fitness)










# image = Image(source_image.shape[0], source_image.shape[1], np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8))
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#





























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
