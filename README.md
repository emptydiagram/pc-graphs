PyTorch implementation of the paper "Learning on Arbitrary Graph Topologies via Predictive Coding".

We have a directed graph $G = (V, E)$. Vertex $i$ has scalar value $x_i$, while the edge $(i, j)$ has a weight $\theta_{ij}$. The vertices that link to vertex $i$ form a prediction

$$\mu_i = \sum_j \theta_{ji} f(x_j)$$

where $f$ is a nonlinearity. The prediction error for vertex $i$ is:

$$\epsilon_i = x_i - \mu_i = x_i - \sum_j \theta_{ji} f(x_j)$$

and the total prediction error is:

$$\mathcal{E} := \frac{1}{2} \sum_k \epsilon_k^2$$

Learning happens in two phases: *inference*, folowed by *weight update*. The paper distinguishes between *sensory vertices* and *internal vertices*. During both phases, sensory vertices are clamped to inputs.

The inference phase fixes the weights and updates the values of internal vertices using gradient descent for a fixed number $T$ of steps to minimize $\mathcal{E}$:

$$\begin{align*}
    \frac{\partial \mathcal{E}}{\partial x_i} & = \frac{1}{2} \sum_k \frac{\partial}{\partial x_i} \epsilon_k^2 \\
    & = \sum_k \epsilon_k \frac{\partial}{\partial x_i} ( x_k - \sum_j \theta_{jk} f(x_j)) \\
    & = \sum_k \epsilon_k (\delta_{ik} - \theta_{ik} f'(x_i)) \\
    & = \epsilon_i - f'(x_i) \sum_k \epsilon_k \theta_{ik}
    \end{align*}$$

The weight update phase fixes the values, and then makes a single weight update (again using gradient descent):

$$\begin{align*}
    \frac{\partial \mathcal{E}}{\partial \theta_{ij}} & = \frac{1}{2} \sum_k \frac{\partial}{\partial \theta_{ij}} \epsilon_k^2 \\
    & = \sum_k \epsilon_k \frac{\partial}{\partial \theta_{ij}} ( x_k - \sum_u \theta_{uk} f(x_u)) \\
    & = - \sum_k \epsilon_k \frac{\partial}{\partial \theta_{ij}} (\sum_u \theta_{uk} f(x_u)) \\
    & = - \epsilon_j f(x_i)
\end{align*}$$

To vectorize, we use row-major convention (all vectors are row vectors) and define vectors


$$x = \begin{bmatrix} x_1, \ldots, x_n \end{bmatrix}$$

$$\epsilon = \begin{bmatrix} \epsilon_1, \ldots, \epsilon_n \end{bmatrix} = x - f(x) \theta$$

The vectorized inference updates use:

$$\frac{\partial \mathcal{E}}{\partial x} = \epsilon - f'(x) \odot (\epsilon \theta^\top)$$

although, by using this, you do need to zero-out components for sensory vertices, since they aren't updated. The weight update uses:

$$\frac{\partial \mathcal{E}}{\partial \theta} = f(x)^{\top} \epsilon$$
