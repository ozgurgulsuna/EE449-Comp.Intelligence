## **Evolutionary Algorithms** ##
-------------------------
### **The n-Queens Problem** ###
The N-Queens problem is a well-known problem in computer science. The goal is to place N queens on an N x N chessboard in such a way that no queen can attack any other queen. The problem is a constraint satisfaction problem. The solution is a permutation of the numbers 1 to N. The fitness function is the number of pairs of queens that can attack each other.

The algorithmic solution is given below:
1. place queens row-by-row from bottom to up
(or column by column from left to right)

2. consider each row as follows:
    - place one queen in a rank sequentially from left to right onto the squares of the board
    - for each placement: check if queen collides with queens in lower rows
    - if not, work on next row recursively
    - afterwards: shift queen one sqare rightwards

3. return solution if queen is placed in top row with no collision


**EA: Data Structure and Algoritm**
```	
typedef struct {    /* an individual*/
    int fitness;    /* fitness (numbw of collisions) */
    int n;          /* number of queens or rows */
    int genes[n];   /* possible positions of queens */
} individual;

typedef struct {    /* a population */
    int size;       /* number of individuals */
    individual *individuals[size];
    individual *best;   /* best individual */
} population;

int individual_init (individual *ind) {
    ind->fitness = 0;
    ind->n = 0;
    for (i=0; i<ind->n; i++) {
        ind->genes[i] = (int)(ind->n * drand());
    }
    return 0;
}

void pop_init (population *pop) {
    int i;
    for (i=0; i<pop->size; i++) {
        pop->individuals[i] = malloc(sizeof(individual));
        if (pop->individuals[i] == NULL) {
            return -1;
        }
        pop->individuals[i]->n = pop->size;
    }
    pop->best = malloc(sizeof(individual));
    if (pop->best == NULL) {
        return -1;
    }
    pop->best->n = pop->size;
}

int pop_eval (population *pop) {
    int i;
    for (i=0; i<pop->size; i++) {
        pop->individuals[i]->fitness = eval(pop->individuals[i]);
    }
    pop->best = get_best(pop);
    return 0;
}

pop_init(pop);
pop_eval(pop);
while ( generation_count => 0 ) {
    pop_select(pop,tm_size,elitist);
    pop_crossover(pop),fraction);
    pop_mutate(pop,mutation_rate);
    pop_eval(pop);
    generation_count--;
}


