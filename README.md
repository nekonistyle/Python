Python 3.8.0

(NNH2.py)
Let n, a1, and a2 be natural numbers.
The program finds parameters of (n,a1,a2,1) ReLU neural networks that express given a1*a2 data.
The algorithm is shown in the paper: "Expressive Number of Two or More Hidden Layer ReLU Neural Neworks" as the proof of Theorem 3.
https://ieeexplore.ieee.org/document/8951658

We strongly recommend to use "Fraction" as a division operator.
If we use "/", this program sometimes returns parameters with large error because of divisions by very large numbers.


All code are written by Kenta Inoue.
