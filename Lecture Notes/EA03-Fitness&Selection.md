## **Evolutionary Algorithms** ##
-------------------------

topics
* Selective pressure
* Selection intensity
* Fitness-proportionate Selection
* Premature convergence
* Vanishing selective pressure
* Adapting of the fitness function

### **Selective Pressure** ###
Selective pressure is the probability that an individual is selected for reproduction. It is a measure of the intensity of selection. The higher the selective pressure, the higher the probability that the best individuals are selected for reproduction. The lower the selective pressure, the higher the probability that the worst individuals are selected for reproduction.

#### **Exploration vs. Exploitation** ####
The selective pressure determines the balance between exploration and exploitation. A high selective pressure leads to exploitation, a low selective pressure leads to exploration. Exploration means that the search space is searched for new solutions. Exploitation means that the search space is searched for the best solutions.

**Exploration of the space:**

– deviation of the individuals over Ω as wide as possible

– preferably big chances to find global optimum

    ⇒ smaller selective pressure is desired

**Exploitation (of good individuals):**

– Strive (try) for (perhaps local) optimum in the vicinity of good individuals

– Convergence to optimum

    ⇒ higher selective pressure is preferred

*convergion of the population* means that all individuals in the population are similar to each other.

### **Selection Intensity** ###
The selection intensity is the difference between the fitness of the best and the worst individual in the population. The higher the selection intensity, the higher the difference between the fitness of the best and the worst individual in the population. The lower the selection intensity, the lower the difference between the fitness of the best and the worst individual in the population.

### **Important Selection Methods** ###
#### **Roulette Wheel Selection** ####
The roulette wheel selection is a fitness-proportionate selection method. The probability that an individual is selected for reproduction is proportional to its fitness. The roulette wheel selection is a stochastic method. The roulette wheel selection is also called fitness-proportionate selection or fitness-proportional selection.

Computation of relative fitness values:
$$
\begin{aligned}
    \text{relative fitness value} &= \frac{\text{fitness value}}{\text{sum of all fitness values}} \\
    &= \frac{f_i}{\sum_{j=1}^n f_j}
\end{aligned}
$$

The selection algorithm:

1. Compute the relative fitness values of all individuals in the population.
2. Set the roulette wheel in motion.
3. Choose the individual that is selected by the roulette wheel.

repeat selection until the desired number of individuals is selected.

disadvantage:
- the roulette wheel selection is not suitable for minimization problems.
- the roulette wheel selection is not suitable for problems with negative fitness values.
- aggravated parallelization.
- domination can lead to premature convergence.

**Adaptation of the fitness function:** Determine relative fitness values from a modified fitness function, for example exponential fitness function:
$$
g(x) = e^{\frac{f(x)}{kT}}
$$
where $k$ is the Boltzmann constant and $T$ is the temperature.

**Stochastic Universal Sampling** (SUS) is a variant of the roulette wheel selection. The roulette wheel is divided into $n$ equally sized sectors, where $n$ is the number of individuals in the population. The roulette wheel is set in motion $n$ times. The first individual is selected from the first sector, the second individual is selected from the second sector, and so on.

Selection of the next population:
- Rotate Roulette-wheel once
- Choose one chromosome per mark
- Here:
    - 1 x A(1), 1 x A(2), 2 x A(3), 2 x A(5)
- Better-than-average individuals are taken into the next population definitely


#### **Rank-based Selection** ####
The rank-based selection is a fitness-proportionate selection method. The probability that an individual is selected for reproduction is proportional to its rank. The rank-based selection is a stochastic method. The rank-based selection is also called linear ranking selection.


#### **Tournament Selection** ####
1. Draw k individuals (2 ≤ k < |P|) randomly from P(t) (selection
without regarding the fitness, let k be the tournament size).

2. Individuals carry out the tournament and best indivual wins:
Tournament winner receives a descendant in the next population
3. All participants (even the winner) of the tournament are returned
to P(t)

#### **Elitism** ####
• unchanged transfer of the best individual (or the k, 1 ≤ k < |P best
individuals) into the next generation

• elite of a population never gets lost, hence elitism
Attention: elite is not exclued from normal selection: genetic operator
can improve them