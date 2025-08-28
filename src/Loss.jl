using Enzyme

"""
    loss(model, x, y; λ=0.5f0) -> Float32

Compute the total training loss as a weighted sum of energy and force errors.  
This function extracts energies and forces from the reference data `y`,  
computes predictions using `dispatch_train`, and uses the derivatives of the  
distance layer to map gradients into Cartesian forces.

# Arguments
- `model::Vector{Chain}`: Species-specific neural network models.
- `x`: Input atomic structure(s).
- `y`: Reference labels containing energy and forces.
- `λ::Float32`: Weight applied to the force contribution (default = 1.0).
- `forces::Bool`: if the loss should be computed taking into account the forces or not ( default = true).


# Returns
- `Float32`: Total loss combining energy and force terms.
"""
function loss(model, x, y; λ::Float32=1.0f0 , forces::Bool = true)

    # Compute the energy contribution
    y_energy = extract_energies(y)
    distances = distance_layer(x)

    # --- Energy prediction + loss ---
    e_pred = dispatch_train(distances, model)
    e_loss = mean((e_pred .- y_energy).^2)


    #if forces compute the forces contribution
    if forces 

        f = extract_forces(y_train ; ndims = 2)
        f_matrix = distance_derivatives(x_train)

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

"""disp
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
    force_loss(model, x::AbstractVector, f::AbstractVector) -> Float32

Compute mean squared error (MSE) between predicted and reference forces for a single input.

# Arguments
- `model::Function`: Callable model.
- `x::AbstractVector`: Single input structure.
- `f::AbstractVector`: Reference forces.

# Returns
- `Float32`: MSE loss for this input.
"""
function force_loss(model, x::AbstractVector,  f , f_matrix)
    d_matrix = calculate_force(x , model)

    predicted_forces = reduce(hcat, (d_matrix[i] * f_matrix[i, :, :] for i in 1:size(d_matrix, 1)))


    return mean((predicted_forces .- f') .^2)
end

"""
    force_loss(model, X::AbstractMatrix, F::AbstractMatrix) -> Vector{Float32}

Compute force losses for a batch of inputs.

# Arguments
- `model::Function`: Callable model.
- `X::AbstractMatrix`: Batch of inputs (rows = examples).
- `F::AbstractMatrix`: Corresponding reference forces.

# Returns
- `Vector{Float32}`: Force loss for each example in the batch.
"""
function force_loss(model, X::Matrix{G1Input}, F::AbstractMatrix , F_matrix)
    # Map each example in the batch to its force loss
 
    losses = map(1:size(X, 1)) do i

      force_loss(model, X[i , :] , F[i , :] , F_matrix[i , : , : ,:])

    end

    return losses
end

"""
    loss_train(model, x, y, fconst; λ=0.5f0) -> Float32

Compute the combined energy + force loss.

# Arguments
- `model::Function`: Callable model.
- `x`: Input structure.
- `y`: Reference energy.
- `fconst`: Precomputed force term (treated as constant).
- `λ::Float32`: Weight for the force contribution (default 0.5).

# Returns
- `Float32`: Total loss combining energy and force term.
"""
function loss_train(models, x, y, fconst; λ::Float32=1.0f0)    
  e_loss = energy_loss( models , x , y)
  return (mean(e_loss) .+ λ .* mean(fconst))
end




