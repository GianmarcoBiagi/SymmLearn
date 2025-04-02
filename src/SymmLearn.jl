module SymmLearn

using Flux
using ProgressMeter
using LinearAlgebra
using Random
using Statistics
using ExtXYZ
using CUDA
using cuDNN


include("MLTrain.jl")
include("ReadFile.jl")

export xyz_to_nn_input,create_model,train_model!

end
