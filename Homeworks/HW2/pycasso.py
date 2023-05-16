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
import time
import matplotlib.pyplot as plt

# import sys
# import os

# from PIL import Image
# from PIL import ImageDraw
# from PIL import ImageFont

# from datetime import datetime

# from multiprocessing import Pool

# Global variables
source_image_path = "images/"
source_image_name = "test.png"
source_image = cv2.imread(source_image_path + source_image_name)
test = cv2.imread("images/test.png")
test_image = cv2.imread("images/test_son.png")
h = source_image.shape[0]
w = source_image.shape[1]
#s_max = int(math.sqrt(h**2+w**2))   # Maximum circle size, diagonal of the image, radius
s_max = 0.3*min(h,w)
h_margin = 1*h               # Horizontal margin
w_margin = 1*w               # Vertical margin 

# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##----------------------------------------------------------------------------##
# Information Display
print_info = True




num_inds = 10 #20  # Individual Number
num_genes = 10 #50 # Gene Number
num_generations = 10000 # Generation Number

tm_size = 5 # Tournament size
frac_elites = 0.2 # Fraction of elites
frac_parents = 0.3 # Fraction of parents
mutation_prob = 0.2  # Mutation probability
mutation_type = 1 # Mutation type


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
        self.genes = genes.astype('int64')

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
        population.append(Individual(genes, 1))
    if print_info == True:
        print("Population initialized, checking collisions...")

    # Check for collisions
    for individual in population:
        for gene in individual.genes:
            while not check_circle(gene):
                gene.x = random.randint(-h_margin, h+h_margin)
                gene.y = random.randint(-w_margin, w+w_margin)
                gene.s = random.randint(0, s_max)
    if print_info == True:
        print("Collisions checked, Sorting population...")

    # Sort population by size
    # for individual in population:
    #     individual.genes.sort(key=lambda x: x.s, reverse=True)
    
    for individual in population:
        individual = evaluate_individual(individual)    
    
    return population

# Check if a circle is inside the image
# https://stackoverflow.com/questions/75231142/collision-detection-between-circle-and-rectangle

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

    # Sort genes by size
    individual.genes.sort(key=lambda x: x.s, reverse=True)

    for gene in individual.genes:
        overlay = image.copy()
        cv2.circle(overlay, (gene.x, gene.y), gene.s, (gene.r, gene.g, gene.b), -1)
        image = cv2.addWeighted(overlay, gene.a, image, 1 - gene.a, 0)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    fitness = 0
    # test_fitness = np.sum((test_image.astype(np.int64)-test.astype(np.int64))**2)
    # print("test fitness: ", test_fitness)
    fitness = np.sum(np.square(image.astype(np.int64)-source_image.astype(np.int64)))

    # fitness = 0
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(3):
    #             # fitness += abs(int(image[i][j][k]) - int(source_image[i][j][k]))
    #             fitness += (int(image[i][j][k]) - int(source_image[i][j][k]))**2

    individual.fitness = -1*fitness.copy()

    return individual

# Tournament selection
def tournament_selection(population):
    tournament = []
    candidates = population.copy()
    
    # for high number of individuals, tournament size is limited to the number of individuals
    size = tm_size
    if tm_size > len(candidates):
        size = len(candidates)

    # main tournament loop
    for i in range(size) :
        tournament.append(random.choice(candidates))
        candidates.remove(tournament[i])
        # print("kisi",i, ":", tournament[i].fitness)
    best = tournament[0]
    for individual in tournament:
        if individual.fitness > best.fitness:
            best = individual
    # print("best: ", best.fitness)
    return best

    # for i in range(tm_size):
    #     tournament.append(random.choice(population))
    # best = tournament[0]
    # for i in range(tm_size):
    #     if tournament[i].fitness > best.fitness:
    #         best = tournament[i]
    # return best

# Elitism
def elitism(population):
    elites = []
    for i in range(math.ceil(frac_elites*num_inds)):
        best = population[0]
        for individual in population:
            if individual.fitness > best.fitness:
                best = individual
        elites.append(best)
        population.remove(best)
    return elites


def natural_selection(population):
    best = []
    for i in range(math.ceil(num_inds - math.ceil(frac_elites*num_inds)- math.ceil(frac_parents*num_inds)*2)):
        best.append(tournament_selection(population))
        population.remove(best[i])
    return best

def parent_selection(population):
    parents = []
    for i in range(math.ceil(frac_parents*num_inds)*2):
        parents.append(tournament_selection(population))
        population.remove(parents[i])
    return parents
# for <num_inds> = 5, <frac_parents> = 0.15 then <num_parents> = 0.75, which is rounded to 1 couple (1*2).


# Crossover
# Parent selection
# <num parents> number of individuals will be used for crossover. The parents are chosen among the best
# individuals which do not advance to the next generation directly. Two parents will create two children.
# Exchange of each gene is calculated individually with equal probability. The probabilities of child 1
# having genei of parent 1 or parent 2 have equal probability, that is 0.5; child 2 gets the genei
# from the other parent which is not chosen for child 1, where 0 ⩽ i < <num genes>.
def crossover(parents):
    children = []
    for i in range(math.ceil(frac_parents*num_inds)):
        child1 = []
        child2 = []
        for j in range(num_genes):
            if random.random() < 0.5:
                child1.append(parents[i].genes[j])
                child2.append(parents[i+1].genes[j])
            else:
                child1.append(parents[i+1].genes[j])
                child2.append(parents[i].genes[j])
        children.append(Individual(child1, 2))
        children.append(Individual(child2, 2))
    return children



#check the negative limits and correct them
def within_limits(phenotype, upper_lim, lower_lim,range):
    while True:
        phenotype_calc = phenotype + random.uniform(-range, range)
        if phenotype_calc < upper_lim and phenotype_calc > lower_lim:
            phenotype = phenotype_calc
            break
    return phenotype
        


def mutation(anan):
    popul = anan.copy()
    for individual in popul:
        # print("muttuobe indivv :", individual)
        individual.fitness = 1    # mutated individuals are not evaluated
        for gene in individual.genes:
            # print("muttuobe gene :", gene)
            if random.random() < mutation_prob:
                while not check_circle(gene):
                    gene.x = int(within_limits(gene.x, w+w_margin, -w_margin, w/4))
                    gene.y = int(within_limits(gene.y, h+h_margin, -h_margin, h/4))
                    gene.s = int(within_limits(gene.s, s_max, 0, 10))
                gene.r = int(within_limits(gene.r, 255, 0, 64))
                gene.g = int(within_limits(gene.g, 255, 0, 64))
                gene.b = int(within_limits(gene.b, 255, 0, 64))
                gene.a = within_limits(gene.a, 1, 0, 0.25)
    return popul

# def mutation(population):
#     popul = population.copy()
#     for individual in popul:
#         individual.fitness = 1    # mutated individuals are not evaluated
#         for gene in individual.genes:
#             if random.random() < mutation_prob:
#                 while not check_circle(gene):
#                     gene.x = int(within_limits(gene.x, w+w_margin, -w_margin, w/4))
#                     gene.y = int(within_limits(gene.y, h+h_margin, -h_margin, h/4))
#                     gene.s = int(within_limits(gene.s, s_max, 0, 10))
#                 gene.r = int(within_limits(gene.r, 255, 0, 64))
#                 gene.g = int(within_limits(gene.g, 255, 0, 64))
#                 gene.b = int(within_limits(gene.b, 255, 0, 64))
#                 gene.a = within_limits(gene.a, 1, 0, 0.25)
#     return popul



# def mutation(population):
#     popul = population.copy()
#     for individual in popul:
#         individual.fitness = 1    # mutated individuals are not evaluated
#         for gene in individual.genes:
#             if random.random() < mutation_prob:
#                 if mutation_type == 0:
#                     while not check_circle(gene):
#                         gene.x = random.randint(-w_margin, w+w_margin)
#                         gene.y = random.randint(-h_margin, h+h_margin)
#                         gene.s = random.randint(0, s_max)
#                     gene.r = random.randint(0, 255)
#                     gene.g = random.randint(0, 255)
#                     gene.b = random.randint(0, 255)
#                     gene.a = random.uniform(0,1)
#                 elif mutation_type == 1:
#                     while not check_circle(gene):
#                         gene.x = int(within_limits(gene.x, w+w_margin, -w_margin, w/4))
#                         gene.y = int(within_limits(gene.y, h+h_margin, -h_margin, h/4))
#                         gene.s = int(within_limits(gene.s, s_max, 0, 10))
#                     gene.r = int(within_limits(gene.r, 255, 0, 64))
#                     gene.g = int(within_limits(gene.g, 255, 0, 64))
#                     gene.b = int(within_limits(gene.b, 255, 0, 64))
#                     gene.a = within_limits(gene.a, 1, 0, 0.25)
#     return popul



# # # Mutation : single phenotype
# def mutation(population):
#     for individual in population:
#         for gene in individual.genes:
#             if mutation_type == 0:
#                 while not check_circle(gene):
#                     if random.random() < mutation_prob:
#                         gene.x = random.randint(-w_margin, w+w_margin)
#                     if random.random() < mutation_prob:
#                         gene.y = random.randint(-h_margin, h+h_margin)
#                     if random.random() < mutation_prob:
#                         gene.s = random.randint(0, s_max)
#                 if random.random() < mutation_prob:
#                     gene.r = random.randint(0, 255)
#                 if random.random() < mutation_prob:
#                     gene.g = random.randint(0, 255)
#                 if random.random() < mutation_prob:
#                     gene.b = random.randint(0, 255)
#                 if random.random() < mutation_prob:
#                     gene.a = random.uniform(0,1)
#             elif mutation_type == 1:
#                 while not check_circle(gene):
#                     if random.random() < mutation_prob:
#                         gene.x = int(within_limits(gene.x, w+w_margin, -w_margin, w/4))
#                     if random.random() < mutation_prob:
#                         gene.y = int(within_limits(gene.y, h+h_margin, -h_margin, h/4))
#                     if random.random() < mutation_prob:
#                         gene.s = int(within_limits(gene.s, s_max, 0, 10))
#                 if random.random() < mutation_prob:
#                     gene.r = int(within_limits(gene.r, 255, 0, 64))
#                 if random.random() < mutation_prob:
#                     gene.g = int(within_limits(gene.g, 255, 0, 64))
#                 if random.random() < mutation_prob:
#                     gene.b = int(within_limits(gene.b, 255, 0, 64))
#                 if random.random() < mutation_prob:
#                     gene.a = within_limits(gene.a, 1, 0, 0.25)
#     return population
# DONE : mutate only one phenotype, eg color, size, position, etc. 

# Main
def main():
    que = []
    start_time = time.time()
    population = init_population()
    best = population[0]
    best_temp = population[0]
    for i in range(num_generations):
        a = 0 
        print("START")
        for individual in population:
            a += 1
            # if individual.fitness == 1:
            print("individual:",a,"fitness:",  individual.fitness)
            individual = evaluate_individual(individual)
            print("individual:",a,"fitness:",  individual.fitness)


##
        for individual in population:
            if (individual.fitness > best.fitness) and (individual.fitness < 0):
                best = individual
        if print_info == True and i%1 == 0:
            print("Generation: ", i, "Best fitness: ", best.fitness)

        if i%10 == 0:
            que.append(best.fitness)
            plt.plot(que)
            plt.draw()
            plt.pause(0.1)
            plt.clf()

        
        if best.fitness > best_temp.fitness:
            best_temp = best
            image = np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8)
            image.fill(255)
            best.genes.sort(key=lambda x: x.s, reverse=True)   ####
            for gene in best.genes:
                overlay = image.copy()
                cv2.circle(overlay, (gene.x, gene.y), gene.s, (gene.r, gene.g, gene.b), -1)
                image = cv2.addWeighted(overlay, gene.a, image, 1 - gene.a, 0)
                cv2.imshow("image", image)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()
            cv2.waitKey(1)

        # print("population size: ", len(population))
        # print(population)


        elites = elitism(population)
        
        # print("population size: ", len(population))

        # elitless = population.copy()
        # population_all = population.copy() 
        # population = population.copy()+elites.copy()

        # print("population size: ", len(population))
        parents = parent_selection(population)
        # print("population size: ", len(population))
        children = crossover(parents)
        # print("population size: ", len(population))

        # print("population size: ", len(population))

        # selected = natural_selection(inc_elites)

        # inc_elites = []

        # population = mutation(population)
        pop = mutation(population)
        # mutation(children)
        # mutated =children+population
 
        population  =  elites + children+pop
        children = []
        elites = []
        mutated = []

        # print("population size: ", len(population))

        for individual in population:
            print("population fitness at end: ", individual.fitness)
        # print("population size: ", len(population))
        # population = []
        # population = populationa.copy()


    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    best = population[0]
    # print(best.fitness)
    for individual in population:
        # print(individual.fitness)
        if individual.fitness > best.fitness:
            best = individual
            print(best.fitness)
    return best




# Run
best_case = main()
image = np.zeros([source_image.shape[0],source_image.shape[1],3],dtype=np.uint8)
image.fill(255)
best_case.genes.sort(key=lambda x: x.s, reverse=True)
for gene in best_case.genes:
    overlay = image.copy()
    cv2.circle(overlay, (gene.x, gene.y), gene.s, (gene.r, gene.g, gene.b), -1)
    image = cv2.addWeighted(overlay, gene.a, image, 1 - gene.a, 0)
    cv2.imshow("image", image)
    cv2.waitKey(100)
    # cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()
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
