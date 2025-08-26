using Flux
using Random
using Statistics




"""
    struct G1Layer

A custom layer structure that contains two sets of weights for neural network connections, as well as additional parameters related to the system.

#### Fields:
- `W_eta::AbstractArray`
  - A weight matrix for the connection "eta" in the neural network layer. This represents the parameters associated with the input feature "eta".
- `W_Fs::AbstractArray`
  - A weight matrix for the connection "Fs" in the neural network layer. This represents the parameters associated with the input feature "Fs".
- `cutoff::Float32`
  - A scalar value represeing the cutoff radius for interactions within the system.
- `charge::Float32`
  - A scalar value representing the atomic charge for the layer.

### Example:
```julia
# Create a G1Layer object with example data
W_eta_example = rand(5, 5)  # Example weight matrix for "eta" with 5x5 dimensions
W_Fs_example = rand(5, 5)   # Example weight matrix for "Fs" with 5x5 dimensions
cutoff_example = 5.0f0      # Example cutoff radius value
charge_example = 1.0f0      # Example atomic charge value

# Create the G1Layer object
layer = G1Layer(W_eta_example, W_Fs_example, cutoff_example, charge_example)

# Print the values
println("Layer created: ", layer)

"""
struct G1Layer
    W_eta::Vector{Float32}
    W_Fs::Vector{Float32}
    cutoff::Float32
    charge::Float32
end



Flux.@layer G1Layer trainable = (W_eta,W_Fs,)

"""
    G1Layer(N_G1::Int, cutoff::Float32, charge::Float32) -> G1Layer

Creates an instance of the custom `G1Layer` with the specified number of radial symmetry functions (`N_G1`), 
cutoff radius, and atomic charge. The layer initializes the `Fs` parameters uniformly across the 
physically relevant distance range (from 0 Å to the cutoff) with a small random jitter, 
and sets `eta` values based on the spacing between `Fs` to ensure adequate coverage and sensitivity 
to interatomic distances.

### Arguments:
- `N_G1::Int`: The number of G1 radial symmetry functions.
- `cutoff::Float32`: The cutoff radius for the layer's calculations.
- `charge::Float32`: The atomic charge associated with the layer, used as a scaling factor.

### Returns:
- An instance of `G1Layer` with physically-informed initial weights for `Fs` and `eta`.
- `Fs` are distributed across the relevant distance range with small random jitter.
- `eta` values are proportional to the spacing between `Fs` to ensure Gaussian functions cover the range effectively.
"""

function G1Layer(N_G1::Int, cutoff::Float32, charge::Float32; seed::Union{Int,Nothing} = nothing)
    # Scegli RNG: deterministico se seed è Int, globale se seed = nothing
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)

    # Centri Fs distribuiti tra r_min e cutoff
    W_Fs = collect(range(0f0, cutoff, length=N_G1)) .+ 0.05f0 .* rand(rng, Float32, N_G1)

    # Larghezze delle Gaussiane coerenti con la distanza tra i centri
    delta = maximum(W_Fs) - minimum(W_Fs)
    eta_base = 4.0f0 / delta^2
    W_eta = eta_base .* (0.8f0 .+ 0.4f0 .* rand(rng, Float32, N_G1))

    return G1Layer(W_eta, W_Fs, cutoff, charge)
end




"""
    (layer::G1Layer)(x::AbstractMatrix{Float32}) -> Matrix{Float32}

Applies the `G1Layer` neural network layer to a batch of atomic environments.

This method computes the response of each neuron (or symmetry function) in the `G1Layer` by summing 
the contributions from neighboring atoms. Each contribution depends on the distance between atoms, 
a radial cutoff function, and learned parameters controlling peak position and width. The atomic charge 
is used as a multiplicative factor.

### Arguments
- `layer::G1Layer`: The `G1Layer` instance, which includes:
    - `W_eta::Matrix{Float32}`: Exponential decay weights, shape `(hidden_dim, input_dim)`.
    - `W_Fs::Matrix{Float32}`: Peak position weights, shape `(hidden_dim, input_dim)`.
    - `cutoff::Float32`: Cutoff radius for neighbor interaction.
    - `charge::Float32`: Atomic charge scaling factor.
- `x::AbstractMatrix{Float32}`: Input distances of shape `(batch_size, N_neighbors)`, 
  where each row represents distances of one atom to its neighbors.

### Returns
- `output::Matrix{Float32}`: Matrix of shape `(hidden_dim, batch_size)`, containing the 
  symmetry function outputs for each atom in the batch.

### Example
```julia
layer = G1Layer(1, 5, 2.5f0, 1.0f0)  # 1 input dim, 5 symmetry functions, cutoff 2.5, charge 1.0
x = rand(Float32, 3, 10)             # Batch of 3 atoms, each with 10 neighbor distances
output = layer(x)                    # Output shape: (5, 3)

"""



function (layer::G1Layer)(x::AbstractMatrix{Float32})
    # x: (n_batch, n_neighbors)
    n_batch, n_neighbors = size(x)
    n_features = size(layer.W_eta, 1)

    output = zeros(Float32, n_features, n_batch)

    @inbounds for b in 1:n_batch
        for f in 1:n_features
            # Calcola tutti i contributi dei vicini in-place senza allocazioni temporanee
            s = 0f0
            for n in 1:n_neighbors
                dx = x[b, n] - layer.W_Fs[f]
                s += fc(x[b, n], layer.cutoff) * exp(-layer.W_eta[f] * dx * dx)
            end
            output[f, b] = 0.1f0 * layer.charge * s
        end
    end

    return output
end




"""
    distance_matrix_layer(input::Matrix{Vector{AtomInput}}; lattice=nothing)

Compute the pairwise distances between atoms for each batch in a matrix of batches.

If a `lattice` is provided, distances are computed using the **minimum-image convention**
under periodic boundary conditions (PBC). Otherwise, simple Cartesian distances are used.

# Arguments
- `input::Matrix{Vector{AtomInput}}`: A matrix of batches. Each element is a vector of `AtomInput`
  objects containing `.species::Int` and `.coord::AbstractVector` (3D coordinates at least).
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional 3x3 lattice matrix for PBC.

# Returns
- `Matrix{G1Input}`: Same shape as `input`. Each `G1Input` contains `species` and a 1×(N-1) matrix of distances.
"""
function distance_matrix_layer(input::Matrix{Vector{AtomInput}}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    ϵ = Float32(1e-7)   # small epsilon to avoid zero division
    batches, _ = size(input)
    N_atoms = length(input[1])  # number of atoms per batch

    output = Matrix{G1Input}(undef, batches, N_atoms)

    @inbounds for I in 1:batches
        batch = input[I]
        out_batch = Vector{G1Input}(undef, N_atoms)

        for i in 1:N_atoms
            distances = Matrix{Float32}(undef, 1, N_atoms - 1)  # store distances for atom i
            idx = 1

            for j in 1:N_atoms
                if j == i
                    continue
                end

                if lattice === nothing
                    # Cartesian distance
                    dx, dy, dz = batch[j].coord[1:3] .- batch[i].coord[1:3]
                    distances[1, idx] = sqrt(dx^2 + dy^2 + dz^2 + ϵ)
                else
                    # Minimum-image distance under PBC
                    distances[1, idx] = d_pbc(batch[i].coord[1:3], batch[j].coord[1:3], lattice)
                end

                idx += 1
            end

            out_batch[i] = G1Input(batch[i].species, distances)
        end

        output[I, :] = out_batch
    end

    return output
end

"""
    distance_matrix_layer(input::Vector{AtomInput}; lattice=nothing)

Compute pairwise distances between atoms in a single vector of `AtomInput` objects.

If a `lattice` is provided, distances are computed using **minimum-image convention**
under periodic boundary conditions (PBC). Otherwise, simple Cartesian distances are used.

# Arguments
- `input::Vector{AtomInput}`: Vector of `AtomInput` objects containing `.species` and `.coord`.
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional 3x3 lattice matrix for PBC.

# Returns
- `Vector{G1Input}`: Each `G1Input` contains `species` and a 1×(N-1) matrix of distances from all other atoms.
"""
function distance_matrix_layer(input::Vector{AtomInput}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    ϵ = Float32(1e-7)
    N_atoms = length(input)

    output = Vector{G1Input}(undef, N_atoms)

    for i in 1:N_atoms
        distances = Matrix{Float32}(undef, 1, N_atoms-1)  # store distances for atom i
        idx = 1

        for j in 1:N_atoms
            if j == i
                continue
            end

            if lattice === nothing
                # Cartesian distance
                dx, dy, dz = input[j].coord[1:3] .- input[i].coord[1:3]
                distances[1, idx] = sqrt(dx^2 + dy^2 + dz^2 + ϵ)
            else
                # Minimum-image distance under PBC
                distances[1, idx] = d_pbc(input[i].coord[1:3], input[j].coord[1:3], lattice)
            end

            idx += 1
        end

        output[i] = G1Input(input[i].species, distances)
    end

    return output
end



"""
    distance_derivatives(input::Matrix{Vector{AtomInput}}; lattice=nothing)

Compute the analytical derivatives of pairwise interatomic distances for a batch of atomic systems.

If `lattice` is provided, distances and derivatives are computed using the **minimum-image convention**
under periodic boundary conditions (PBC). Otherwise, standard Cartesian distances are used.

# Arguments
- `input::Matrix{Vector{AtomInput}}`: 2D array of atomic systems. Each element is a vector of `AtomInput`
  objects containing `.coord` fields with 3D coordinates.
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional 3x3 lattice matrix for PBC.

# Returns
- `Array{Float32, 4}`: Tensor of shape `(n_batch, n_atoms, n_atoms-1, 3)`:
  - `outputs[b, i, j, :]` → derivative of distance `d(i, j+1)` w.r.t atom `i`.
  - `outputs[b, j+1, i, :]` → derivative of distance `d(i, j+1)` w.r.t atom `j+1`.
  - Derivatives w.r.t. other atoms are zero (not stored).

# Notes
- If two atoms coincide (distance numerically zero), the derivative is set to `(0, 0, 0)`.
"""
function distance_derivatives(input::Matrix{Vector{AtomInput}}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    # Ensure input has batch dimension
    if ndims(input) == 1
        input = reshape(input, (1, :))
    end

    n_batch, _ = size(input)
    n_atoms = length(input[1, 1])

    # Output tensor: (batch, i_atom, j_atom, coord)
    outputs = Array{Float32, 4}(undef, n_batch, n_atoms, n_atoms-1, 3)

    for b in 1:n_batch
        batch = input[b, 1]

        for i in 1:n_atoms
            ri = batch[i].coord

            for j in i:(n_atoms-1)
                rj = batch[j+1].coord

                # Compute difference vector
                if lattice === nothing
                    diff = ri .- rj
                    dij = sqrt(sum(diff.^2))
                else
                    # Minimum-image vector and distance
                    dij, rvec, _ = d_pbc(ri, rj, lattice; return_image=true)
                    diff = rvec
                end

                # Compute derivatives
                if dij > 1e-12
                    grad_i = diff ./ dij
                    grad_j = -grad_i
                else
                    grad_i = zeros(Float32, 3)
                    grad_j = zeros(Float32, 3)
                end

                # Store in output tensor
                outputs[b, i, j, :] = grad_i
                outputs[b, j+1, i, :] = grad_j
            end
        end
    end

    return outputs
end



"""
    distance_derivatives(input::Vector{AtomInput}; lattice=nothing)

Compute the analytical derivatives of pairwise interatomic distances for a single atomic system.

If `lattice` is provided, distances and derivatives are computed using the **minimum-image convention**
under periodic boundary conditions (PBC). Otherwise, standard Cartesian distances are used.

# Arguments
- `input::Vector{AtomInput}`: Vector of atoms. Each `AtomInput` must have a `.coord` field with 3D coordinates.
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional 3x3 lattice matrix for PBC.

# Returns
- `Array{Float32, 3}`: Tensor of shape `(n_atoms, n_atoms-1, 3)`:
  - `outputs[i, j, :]` → derivative of distance `d(i, j+1)` w.r.t atom `i`.
  - `outputs[j+1, i, :]` → derivative of distance `d(i, j+1)` w.r.t atom `j+1`.
  - Derivatives w.r.t. other atoms are zero (not stored).

# Notes
- If two atoms coincide (distance numerically zero), the derivative is set to `(0, 0, 0)`.
"""
function distance_derivatives(input::Vector{AtomInput}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    n_atoms = length(input)

    # Output tensor: (i_atom, j_atom, coord)
    outputs = Array{Float32, 3}(undef, n_atoms, n_atoms - 1, 3)

    for i in 1:n_atoms
        ri = input[i].coord

        for j in i:(n_atoms-1)
            rj = input[j+1].coord

            # Compute difference vector
            if lattice === nothing
                diff = ri .- rj
                dij = sqrt(sum(diff.^2))
            else
                # For PBC: use minimum-image distance and vector
                dij, rvec, _ = d_pbc(ri, rj, lattice; return_image=true)
                diff = rvec  # vector along minimum-image
            end

            # Compute derivatives
            if dij > 1e-12
                grad_i = diff ./ dij   # derivative w.r.t. atom i
                grad_j = -grad_i       # derivative w.r.t. atom j+1
            else
                grad_i = zeros(Float32, 3)
                grad_j = zeros(Float32, 3)
            end

            # Store derivatives in output tensor
            outputs[i, j, :] = grad_i
            outputs[j+1, i, :] = grad_j
        end
    end

    return outputs
end







"""
    build_branch(atom::String, G1_number::Int, R_cutoff::Float32) -> Chain

Builds a per-species subnetwork consisting of:
- A `G1Layer` configured with species charge.
- A single Dense layer with tanh activation to output scalar atomic energy.

The atomic charge is scaled from `element_to_charge` using a factor of 0.1.
"""



function build_branch(Atom_name::String, G1_number::Int, R_cutoff::Float32 , depth::Int ; seed::Union{Int,Nothing} = nothing)
    ion_charge = element_to_charge[Atom_name]
    if depth == 2
    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge) ; seed),
        LayerNorm(G1_number),
        Dense(G1_number, 32, tanh), 
        LayerNorm(32),
        Dense(32, 16, tanh),
        LayerNorm(16),
        Dense(16, 8, tanh),
        LayerNorm(8),
        Dense(8, 1)
    )
    elseif depth == 1

    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge)),
        LayerNorm(G1_number),
        Dense(G1_number , 16 , tanh),
        LayerNorm(16),
        Dense(16,8,tanh),
        LayerNorm(8),
        Dense(8, 1)   
    )



    end
end



"""
    build_species_models(unique_species::Vector{String}, species_idx::Dict{String,Int}, G1_number::Int, R_cutoff::Float32)

Creates an array of Flux `Chain` models, one for each unique species.  
The array is indexed according to `species_idx`, so that each species can be dispatched
to the correct model based on its numeric index.  

# Arguments
- `unique_species::Vector{String}`: List of species names.
- `species_idx::Dict{String,Int}`: Mapping from species name to numeric index.
- `G1_number::Int`: Number of neurons in the G1Layer.
- `R_cutoff::Float32`: Cutoff radius for the G1Layer.

# Returns
- `species_models::Vector{Chain}`: Array of models, ready for Enzyme differentiation.
"""
function build_species_models(unique_species::Vector{String}, species_idx::Dict{String,Int}, 
                              G1_number::Int, R_cutoff::Float32 ; depth = 2::Int , seed::Union{Int,Nothing} = nothing)
    n_species = length(unique_species)
    species_models = Vector{Chain}(undef, n_species)
    
    for spec in unique_species
        idx = species_idx[spec]
        species_models[idx] = build_branch(spec, G1_number, R_cutoff , depth ; seed = seed)
    end
    
    return species_models
end


"""
    dispatch(atoms::Vector{AtomInput}, species_models::Vector{Chain})

Applies the correct model to each atom in `atoms` based on its `species`.  
Only the numerical `features` are passed to the model, so this is compatible with Enzyme.

# Arguments
- `atoms::Vector{AtomInput}`: Batch of atoms with `species` and `features`.
- `species_models::Vector{Chain}`: Array of models, indexed by species numeric ID.

# Returns
- `outputs::Vector{Float32}`: Output of the correct model for each atom.
"""
function dispatch(distances::Vector{G1Input}, species_models::Vector{Chain})


    n_atoms = size(distances , 1)

    outputs = Vector{Float32}(undef, n_atoms)


    @inbounds for i in 1:n_atoms

        distance = distances[i]
        model = species_models[distance.species]
        outputs[i] = model(distance.dist)[1]  # assuming model outputs a 1-element vector
    end

    outputs = sum(outputs)

    return outputs
end

function dispatch(distances::Matrix{G1Input}, species_models::Vector{Chain})
    
    
    n_batches , n_atoms = size(distances)
     
    outputs = Matrix{Float32}(undef, (n_batches , n_atoms))

    @inbounds for i in 1:n_atoms
        # All batches for atom i
        distance_col = distances[:, i]

        # Determine species (assume same across batches)
        model = species_models[distance_col[1].species]

        # Preallocate buffer for model input
        n_neighbors = length(distance_col[1].dist)
        batch_input = Array{Float32}(undef, n_batches, n_neighbors)

        # Fill buffer without dynamic concatenation
        for b in 1:n_batches
            batch_input[b, :] = distance_col[b].dist
        end

        # Forward pass
        outputs[:, i] = vec(model(batch_input))

    end

    final_outputs = dropdims(sum(outputs, dims = 2); dims=2)

    return  final_outputs
end

function dispatch_wd(atoms, species_models::Vector{Chain})

    distance = distance_matrix_layer(atoms)

    return(dispatch(distance , species_models))

end

"""
    predict_forces(x, model; flat=false)

Compute atomic forces from a trained model given atomic positions.

# Arguments
- `x::AbstractArray`: Batched input positions of shape `(n_batches, n_atoms, 3)`.  
- `model`: The trained ML model used to predict the potential energy.  
- `flat::Bool=false`: If `false`, returns forces with shape `(n_batches, n_atoms, 3)`.  
  If `true`, returns a 1D vector where forces are flattened in the order:
"""



function predict_forces(x , model ; flat = false)
    dist = distance_matrix_layer(x)
    derivatives = distance_derivatives(x)

    n_batches , _ = size(x)
    n_atoms = size(x[1] , 1)

    predicted_forces = zeros(Float32 , (n_batches , n_atoms , 3))

    for b in 1:n_batches
        grad = calculate_force(dist[b, :], model)
        temp = zeros(Float32, n_atoms , 3)
        for i in 1:n_atoms
            contrib = sum(grad[i] .* derivatives[b,i, :, :], dims=1)  # (1,3)
            temp[i , 1:3] .= vec(contrib)
        end
        predicted_forces[b, : , :] .= temp
    end

    if flat
        flat_forces = Float32[]
        for b in 1:n_batches
            for i in 1:n_atoms
                append!(flat_forces, predicted_forces[b,i,:])
            end
        end
        return flat_forces
    else
        return predicted_forces
    end
end

