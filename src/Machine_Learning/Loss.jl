using Enzyme

"""
    loss(model, x, y; λ=1.0f0, forces=true) -> Float32

Compute the training loss as mean squared error on energies plus, optionally,
a weighted force-matching term.

This is a user-level function; the developer version is `loss_train`.

# Arguments
- `model::Vector{Chain}`: Species-specific neural network models.
- `x`: Input atomic structure(s).
- `y`: Reference labels containing energies and forces.
- `λ::Float32` (optional): Weight for the force contribution. Default = 1.0.
- `forces::Bool` (optional): If true, include force loss. If false, only energy loss. Default = true.

# Returns
- `Float32`: Total loss (energy-only if `forces=false`, energy + λ·forces otherwise).
```julia
# Assume models, x_batch, y_batch are defined
total_loss = loss(models, x_batch, y_batch; λ=0.5f0, forces=true)
println("Computed loss: ", total_loss)```
"""
function loss(model, x, y; λ::Float32=1.0f0 , forces::Bool = true)
    one_sample = false
    # Compute the energy contribution
    y_energy = extract_energies(y)
    distances = distance_layer(x)
 

    # --- Energy prediction + loss ---
    e_pred = dispatch_train(distances, model)
    e_loss = mean((e_pred .- y_energy).^2)


    #if forces compute the forces contribution
    if forces 

        f =  extract_forces(y)
        f_matrix = distance_derivatives(x)
      

        f_loss = mean(force_loss(model, distances,  f , f_matrix))

    else

        f_loss = 0f0

    end
    

    # --- Total loss ---
    return e_loss + λ * f_loss
end


"""
    energy_loss(model, x, y) -> AbstractArray

Compute the squared error between predicted energy and reference energy.

# Arguments
- `model::Function`: A callable model.
- `x`: Input structure.
- `y`: Reference scalar energy.

# Returns
- Squared error as an array.
"""
function energy_loss( model, x, y)

    e_guess = dispatch_train(x , model)


    return (e_guess .- y).^2
end

"""
    calculate_force(model, x::AbstractVector) -> AbstractVector

Compute the negative gradient of the scalar model output w.r.t. input `x`, i.e., the predicted forces.

# Arguments
- `model::Function`: Callable model.
- `x::AbstractVector`: Single input structure.

# Returns
- `AbstractVector`: Predicted forces of same size as `x`.
"""
function calculate_force( x::AbstractVector , model)

    # Enzyme gradient: returns a tuple (grad w.r.t model, grad w.r.t x)
    grad  = Enzyme.gradient(set_runtime_activity(Reverse) , (x,m) -> dispatch_train(x,m),  x , Const(model))

    d_matrix = [-g.dist for g in grad[1]]
  

    return d_matrix

end




"""
    force_loss(model, x::AbstractVector, f, f_matrix) -> Float32

Compute the mean squared error (MSE) between predicted and reference forces for a single atomic structure.

# Description
This is the **single-structure version** of `force_loss`.  
It calculates the predicted forces by applying the model to the input representation `x`
and mapping gradients via `f_matrix`, then compares them to the reference forces `f` using MSE.

# Arguments
- `model`: Callable neural network model.
- `x::AbstractVector`: Input atomic structure or per-atom representation.
- `f::AbstractMatrix{Float32}`: Reference forces, shape `(num_atoms, 3)`.
- `f_matrix::AbstractArray{Float32, 3}`: Distance derivative matrix for mapping gradients to Cartesian forces.

# Returns
- `Float32`: Mean squared error between predicted and reference forces for the structure.

# See also
`force_loss(model, X::Matrix{G1Input}, F::Array{Float32,3}, F_matrix::Array{Float32,4})` for the batched version.
"""
function force_loss(model, x::AbstractVector,  f , f_matrix)

    d_matrix = calculate_force(x , model)
    n_atoms = size(x , 1)

    predicted_forces = zeros(Float32 , n_atoms , 3)
    for i in 1:n_atoms

        predicted_forces[i , :] =  d_matrix[i] * f_matrix[i, :, :] 
    
    end

    return mean((predicted_forces .- f) .^2)
end

"""
    force_loss(model, X::Matrix{G1Input}, F::Array{Float32,3}, F_matrix::Array{Float32,4}) -> Vector{Float32}

Compute force losses for a batch of atomic structures.

# Description
This is the **batch version** of `force_loss`.  
Each row of `X` corresponds to one structure. The function computes the predicted forces
for each structure using the provided model and derivative matrices, then evaluates
the mean squared error (MSE) against the reference forces.

# Arguments
- `model`: Callable neural network model.
- `X::Matrix{G1Input}`: Batch of inputs, one row per structure.
- `F::Array{Float32,3}`: Reference forces for each structure, shape `(num_samples, num_atoms, 3)`.
- `F_matrix::Array{Float32,4}`: Force derivative matrices for each structure, shape `(num_samples, num_atoms, 3, 3)`.

# Returns
- `Vector{Float32}`: Force loss for each structure in the batch.

# See also
`force_loss(model, x::AbstractVector, f, f_matrix)` for the single-structure version.
"""
function force_loss(model, X::Matrix{G1Input}, F::Array{Float32, 3}, F_matrix::Array{Float32, 4})
    # Map each example in the batch to its force loss
    
    losses = map(1:size(X, 1)) do i

      force_loss(model, X[i , :] , F[i , : , :] , F_matrix[i , : , : , :])

    end

    return losses
end

"""
    loss_train(models, x, y, fconst; λ=1.0f0) -> Float32

Compute the combined energy and force loss for training.

This is a developer-level function; the public API is `loss`.

# Arguments
- `models`: Callable model(s) for energy prediction.
- `x`: Input structure(s).
- `y`: Reference energies corresponding to `x`.
- `fconst`: Precomputed force contributions (treated as constant).
- `λ::Float32` (optional): Weight for the force contribution. Default = 1.0.

# Returns
- `Float32`: Total loss combining mean energy loss and weighted mean force term.
"""
function loss_train(models, x, y, fconst; λ::Float32=1.0f0)    
  e_loss = energy_loss( models , x , y)
  return (mean(e_loss) .+ λ .* mean(fconst))
end




