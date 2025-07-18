# SymmLearn
his project implements a Machine Learning Force Field (MLFF) using Julia and the Flux.jl deep learning framework.
The main goal is to predict the total energy of atomic systems based on interatomic distances through a network of networks approach:

    Each neural network corresponds to a specific atom type and predicts its atomic energy from its local environment.

    The total energy of the system is obtained by summing the contributions of all atoms.

This architecture is inspired by second-generation neural network potentials, commonly used in materials science and molecular simulations.
