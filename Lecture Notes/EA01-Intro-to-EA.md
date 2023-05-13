## **Evolutionary Algorithms** ##

### **Introduction** ###
Fundamental approach to solving problems by mimicking the process of natural selection.
- Analytical Solution:
    efficient but not always possible.

- Exhaustive Search:
    very inefficient, so only usable in small search spaces

- Random Search:
    inefficient, but usable in large search spaces.

- Guided Search:
    Precondition: similar elements in Ω have similar function values.

**Variation:** 
mutation and genetic recombination is used in sexual reproduction to create new variants.

**Inheritance:**
the new variants inherit the characteristics of their parents.

![Terms1](figures\EA01-Fundamental-Terms-1.png)
![Terms1](figures\EA01-Fundamental-Terms-2.png)

**Encoding:**
The representation of the solution in the search space.
Problem specfic. Choosen by the designer with respect to the problem.

**Fitness Function:**
The fitness function is the objective function of the problem. It is used to evaluate the quality of a solution. May contain multiple objectives and/or constraints.

**Genetic Operators-Mutation:**
The mutation operator is used to create new variants of a solution. It is used to introduce new information into the population using random changes.

**Genetic Operators-Crossover:**
Recombination operator. Used to combine information from two or more solutions to create new solutions.

### **N-Queens Problem** ###
The N-Queens problem is a well-known problem in computer science. The goal is to place N queens on an N x N chessboard in such a way that no queen can attack any other queen. The problem is a constraint satisfaction problem. The solution is a permutation of the numbers 1 to N. The fitness function is the number of pairs of queens that can attack each other.

![N-Queens](figures\EA01-N-Queens.png)

- 1 solution candidate = 1 chromosome with N genes
- 1 gene = 1 row with N possible values
- 1 value = position of the queen in the row

constraint: no two queens can be in the same row


//// **CONTINUE WITH PAGE 12** ////