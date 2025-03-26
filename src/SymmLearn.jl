module SymmLearn

using Flux
using ProgressMeter
using LinearAlgebra
using Random

include("MLTrain.jl")
include("ReadFile.jl")

export create_model,train_model!

end
