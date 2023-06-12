## Neural Networks : Deep Learning ##
-----
### Multi-Layer Perceptron ###

The multi-layer perceptron (MLP) is a feedforward neural network that consists of multiple layers of perceptrons. The MLP is a generalization of the perceptron in that it can have one or more hidden layers. The MLP is a universal approximator, as shown by Cybenko (1989) and Hornik (1991). The MLP can be used for classification and regression.

### Fully-Connected Deep Networks (Deep MLP)
 * Problems concerning MLPs:
     - The number of parameters in the network can be very large.
        - to reduce CNN approach is used. Local connectivity and weight sharing are used to reduce the number of parameters.
    - The labelwork is intensive.
        - Auto Encoder, Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN) are used to reduce the labelwork. 

- Multiple layers are used to build an improved feature representation of the input data.

### Convolutional Neural Networks
CNN's are a special type of neural network that uses convolution in place of general matrix multiplication in at least one of their layers. CNN's are used for image and video recognition, recommender systems, natural language processing, and more.

Convolution in 2D
- Convolution is a mathematical operation that takes two functions f and g and produces a third function that represents how the shape of one is modified by the other.

in more mathematical terms with 2D input:
$$
f(x,y) \ast g(x,y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(a,b)g(x-a,y-b)da db
$$

also in discrete form:
$$
f(x,y) \ast g(x,y) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f(m,n)g(x-m,y-n)
$$

cross correlation:
$$
f(x,y) \ast g(x,y) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f(m,n)g(x+m,y+n)
$$ 


