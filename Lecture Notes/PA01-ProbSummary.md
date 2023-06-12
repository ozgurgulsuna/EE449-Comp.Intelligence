## Probabilistic Approaches ##
-----

Define event **S** to assign a probability to the occurrence of a particular event.

Possible outcomes of **S** are called **sample space** and are denoted by **Ω**.

Each event **E** is a subset of **Ω**.

The event space satisfies the following properties:
1. It contains the empty set and the sample space.
2. It is closed under complementation.
3. It is closed under union.

**Probability Distribution** , P over (Ω,S) is a function that assigns a probability to each event in S.

$ P(\alpha)  \geq 0 $ for all $\alpha \in S$

$ P(\Omega) = 1 $ trivial event

$ P(\alpha \cup \beta) = P(\alpha) + P(\beta) $ if $\alpha \cap \beta = \emptyset$

**Conditional Probability** is the probability of an event given that another event has occurred.

$ P(\alpha | \beta) = \frac{P(\alpha \cap \beta)}{P(\beta)} $

### **Bayes' Rule** ###
Probability of an event given another event has occurred.

$$
P(\alpha | \beta) = \frac{P(\beta | \alpha) P(\alpha)}{P(\beta)}
$$

$$
P(\alpha | \beta, \gamma) = \frac{P(\beta | \alpha, \gamma) P(\alpha | \gamma)}{P(\beta | \gamma)}
$$

### **Chain Rule** ###

$$
P(\alpha, \beta, \gamma) = P(\alpha | \beta, \gamma) P(\beta | \gamma) P(\gamma)
$$

### **Random Variables** ###
A random variable is a function that maps the sample space to the real numbers.

Grade (G) of a student defined by a function $f_g$ that maps each person to his/her grade is a random variable. 

$\Omega_G = \{    A, B, C \} $

$ f_G (w) = A$

$ P(G = A) = 0.15 , P(G = B) = 0.35 , P(G = C) = 0.5 $

**marginal distribution** is the probability distribution of a subset of random variables from a larger set.

**joint distribution** assigns a probability to each possible combination of values for the random variables. It is consistent with the marginal distributions.

$$
\begin{array}{|c|c|c|c|}
\hline
P(G,I)&\text{Intelligence Low} & \text{Intelligence High} & P(grade) \\
\hline
A & 0.07 & 0.18 & 0.25 \\
\hline
B & 0.28 & 0.09& 0.37 \\
\hline
C & 0.35 & 0.03 & 0.38 \\
\hline
P(intelligence) & 0.7 & 0.3 & 1 \\
\hline
\end{array}
$$

**conditional distribution** is the probability distribution of a subset of random variables given the values of the other variables.

$$
P(Intelligence = High) = 0.3 \\
P(Intelligence = High | Grade = A) = \frac{P(Intelligence = High, Grade = A)}{P(Grade = A)}= \frac{0.18}{0.25} = 0.72
$$

$$
P(Grade = A | Intelligence = High) = \frac{P(Intelligence = High, Grade = A)}{P(Intelligence = High)} = \frac{0.18}{0.3} = 0.6
$$


$$
\begin{array}{|c|c|c|c|}
\hline
P(G|I)&\text{Intelligence Low} & \text{Intelligence High} & P(grade) \\
\hline
A & 0.07 / 0.7 & 0.18/ 0.3 & 0.25 \\
\hline
B & 0.28/ 0.7 & 0.09/ 0.3& 0.37 \\
\hline
C & 0.35/ 0.7 & 0.03/ 0.3 & 0.38 \\
\hline
P(intelligence) & 0.7 & 0.3 & 1 \\
\hline
\end{array}
$$


$$
\begin{array}{|c|c|c|c|}
\hline
P(I|G)&\text{Intelligence Low} & \text{Intelligence High} & P(grade) \\
\hline
A & 0.07 / 0.25 & 0.18/ 0.25 & 0.25 \\
\hline
B & 0.28/ 0.37 & 0.09/ 0.37& 0.37 \\
\hline
C & 0.35/ 0.38 & 0.03/ 0.38 & 0.38 \\
\hline
P(intelligence) & 0.7 & 0.3 & 1 \\
\hline
\end{array}
$$

### **Independence** ###
Two random variables are independent if the probability of one does not depend on the other.

$$
P(\alpha | \beta) = P(\alpha) \text{ if } \alpha \text{ and } \beta \text{ are independent}
$$

$$
\text{in other words, if}
$$

$$
P(\beta) = 0 \text{ then the events are independent}
$$

### **Conditional Independence** ###
Two random variables are conditionally independent given a third random variable if the conditional probability of one does not depend on the other.

$$
P(\alpha | \beta \cap \gamma) = P(\alpha | \gamma) \text{ if } \alpha \text{ and } \beta \text{ are conditionally independent given } \gamma
$$

### **Querying** ###
Given a set of random variables, we can query the probability of a subset of them given the values of the other variables.

1. $P( \textbf{ Q }  | E=e)$ Query ($Q$) given that evidence $E$ is e.

**// CONTINUE FROM SLIDE 18 //**
if necessary






