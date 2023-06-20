## Probabilistic Approaches ##
-----
**Gaussian  Distribution in detail**
Univariate Gaussian Distribution
$$
\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{1}{2 \sigma^2} (x - \mu)^2 \right)
$$

Multivariate Gaussian Distribution
$$
\mathcal{N}(\mathbf{x} | \mathbf{\mu}, \mathbf{\Sigma}) = \frac{1}{(2 \pi)^{D/2}} \frac{1}{|\mathbf{\Sigma}|^{1/2}} \exp \left( - \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) \right)
$$

Ex.
$$
\sum = \begin{bmatrix}
    4 & 2 & -2 \\
    2 & 5 & -5 \\
    -2 & -5 & 8 \\
\end{bmatrix}
$$

Covariance is calculated as follows: 
$$
Cov(X_i, X_j) = \frac{1}{N} \sum_{n=1}^{N} (x_{i,n} - \mu_i)(x_{j,n} - \mu_j)
$$


$$
Cov(X_1, X_3) = -2 \\
Cov(X_2, X_3) = -5
$$

