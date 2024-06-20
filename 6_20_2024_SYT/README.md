# Standard Young Tableaux
In this assignment, you will learn to generate data for [Standard Young Tableaux](https://en.wikipedia.org/wiki/Young_tableau). Standard Young tableaux (SYT) show up very often in a field called algebraic combinatorics. It turns out that lots of information about permutations can be derived from SYTs of a specific shape.

Your goal in this assignment will be to first generate all SYTs of a specified shape $(\lambda_1, \lambda_2, \ldots, \lambda_k)$. In general, this can be quite a slow process. Therefore, you will next implement a function that can generate a random SYT of a specified shape. You will implement two different algorithms for random generation and compare and contrast the two.

If you are interested in learning more, [this paper by Greene, Nijenhuis and Wilf](https://www2.math.upenn.edu/~wilf/website/Probabilistic%20proof.pdf) details an algorithm that can randomly generate SYTs uniformly at random in a fast manner that is better than both of the naive approaches in this assignment.
