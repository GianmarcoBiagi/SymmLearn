# SymmLearn.jl


## What Is This Package Useful For?


SymmLearn.jl provides a framework to predict energies and forces of atomic systems using machine learning.
It automates the generation of atom-centered representations via symmetry functions and the application of species-specific neural networks. 

The framework is based on the Behler-Parrinello Neural Network approach: each neural network corresponds to a single atom type and predicts its contribution to the total energy. The total energy of the system is obtained by summing all atomic neural network contributions, and forces are computed by derivating the total energy respect to each atom coordinates.


Read more about Behler-Parrinello potentials [here](https://doi.org/10.1103/PhysRevLett.98.146401).




## Citation and license (TODO)



This package is written in [Julia](http://julialang.org/), a modern high-level,
high-performance dynamic programming language designed for technical computing.

The `SymmLearn.jl` package is licensed under the MIT "Expat" License. The
original author is Gianmarco Biagi.