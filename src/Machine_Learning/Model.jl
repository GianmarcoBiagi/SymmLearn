using Flux
using Random
using Statistics

"""
    G1Layer(W_eta::Vector{Float32}, W_Fs::Vector{Float32}, cutoff::Float32, charge::Float32)

Custom neural network layer with weights and system-specific parameters.

# Fields
- `W_eta::Vector{Float32}`: Weight vector for the "eta" connection.
- `W_Fs::Vector{Float32}`: Weight vector for the "Fs" connection.
- `cutoff::Float32`: Cutoff radius for interactions.
- `charge::Float32`: Atomic charge associated with the layer.

# Example
```julia
W_eta_example = rand(Float32, 5)
W_Fs_example  = rand(Float32, 5)
cutoff_example = 5.0f0
charge_example = 1.0f0

layer = G1Layer(W_eta_example, W_Fs_example, cutoff_example, charge_example)
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
    G1Layer(N_G1::Int, cutoff::Float32, charge::Float32; seed::Union{Int,Nothing}=nothing) -> G1Layer

Create a G1 radial symmetry function layer with physically-informed initialization.

- `Fs` are distributed linearly across the distance range [r_min, cutoff] with small random jitter.
- `eta` values are set according to the average spacing between `Fs`, with slight random perturbations.
- This initialization avoids extreme contributions from very small distances that can bias energy predictions.

# Arguments
- `N_G1::Int`: Number of G1 functions.
- `cutoff::Float32`: Cutoff radius for interatomic interactions.
- `charge::Float32`: Atomic charge used as scaling factor.
- `seed::Int` or `nothing`: Optional RNG seed for reproducibility.

# Returns
- `G1Layer` instance with initialized `Fs` and `eta`.
"""
function G1Layer(N_G1::Int, cutoff::Float32, charge::Float32; seed::Union{Int,Nothing}=nothing)
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)

    # Avoid Fs too close to zero to prevent huge contributions
    r_min = 0.1f0
    W_Fs = range(r_min, cutoff, length=N_G1) .+ 0.01f0 .* rand(rng, Float32, N_G1)

    # Compute average spacing and set eta proportional to 1/(spacing^2)
    delta = diff(W_Fs)
    avg_spacing = mean(delta)
    eta_base = 1.0f0 / (avg_spacing^2)
    W_eta = eta_base .* (0.8f0 .+ 0.4f0 .* rand(rng, Float32, N_G1))

    return G1Layer(W_eta, W_Fs, cutoff, charge)
end




"""
    (layer::G1Layer)(x::AbstractMatrix{Float32}) -> Matrix{Float32}

Apply the `G1Layer` to a batch of atomic distances, computing radial symmetry function outputs.

# Arguments
- `layer::G1Layer`: Layer instance with fields:
    - `W_eta::Vector{Float32}`: Width parameters for each symmetry function.
    - `W_Fs::Vector{Float32}`: Peak positions for each symmetry function.
    - `cutoff::Float32`: Cutoff radius for neighbor interactions.
    - `charge::Float32`: Atomic charge scaling factor.
- `x::AbstractMatrix{Float32}`: Distance matrix of shape `(n_batch, n_neighbors)`.

# Returns
- `Matrix{Float32}`: Symmetry function outputs, shape `(n_features, n_batch)`,
  where `n_features = size(layer.W_eta, 1)`.

# Example
```julia
layer = G1Layer(5, 2.5f0, 1.0f0)  # 5 symmetry functions, cutoff 2.5, charge 1.0
x = rand(Float32, 3, 10)          # 3 atoms, 10 neighbors each
output = layer(x)                 # Output shape: (5, 3)


"""


function (layer::G1Layer)(x::AbstractMatrix{Float32})
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
    distance_layer(input::Matrix{Vector{AtomInput}}; lattice=nothing)

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
function distance_layer(input::Matrix{Vector{AtomInput}}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    ϵ = Float32(1e-7)   # small epsilon to avoid zero division
    batches, _ = size(input)
    N_atoms = length(input[1])  # number of atoms per batch

    output = Matrix{G1Input}(undef, batches, N_atoms)

    @inbounds for I in 1:batches
        batch = input[I]

        # per ogni atomo prealloca il vettore distanze
        dist_lists = [Matrix{Float32}(undef, 1, N_atoms-1) for _ in 1:N_atoms]
        positions  = [1 for _ in 1:N_atoms]

        for i in 1:N_atoms
            for j in (i+1):N_atoms
                if lattice === nothing
                    dx, dy, dz = batch[j].coord[1:3] .- batch[i].coord[1:3]
                    d = sqrt(dx^2 + dy^2 + dz^2 + ϵ)
                else
                    d = d_pbc(batch[i].coord[1:3], batch[j].coord[1:3], lattice)
                end

                # scrivi nei vettori dei due atomi
                dist_lists[i][1, positions[i]] = d
                positions[i] += 1

                dist_lists[j][1, positions[j]] = d
                positions[j] += 1
            end
        end

        # costruisci il G1Input per ogni atomo
        out_batch = Vector{G1Input}(undef, N_atoms)
        for i in 1:N_atoms
            out_batch[i] = G1Input(batch[i].species, dist_lists[i])
        end

        output[I, :] = out_batch
    end

    return output
end


"""
    distance_layer(input::Vector{AtomInput}; lattice=nothing)

Compute pairwise distances between atoms in a single vector of `AtomInput` objects.

If a `lattice` is provided, distances are computed using **minimum-image convention**
under periodic boundary conditions (PBC). Otherwise, simple Cartesian distances are used.

# Arguments
- `input::Vector{AtomInput}`: Vector of `AtomInput` objects containing `.species` and `.coord`.
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional 3x3 lattice matrix for PBC.

# Returns
- `Vector{G1Input}`: Each `G1Input` contains `species` and a 1×(N-1) matrix of distances from all other atoms.
"""
function distance_layer(input::Vector{AtomInput}; lattice::Union{Nothing, Matrix{Float32}}=nothing)
    ϵ = Float32(1e-7)
    N_atoms = length(input)

    # prealloca per ogni atomo un vettore di distanze
    dist_lists = [Matrix{Float32}(undef, 1, N_atoms-1) for _ in 1:N_atoms]

    # indici "locali" per riempire i vettori (perché hanno dimensione N_atoms-1)
    positions = [1 for _ in 1:N_atoms]

    for i in 1:N_atoms
        for j in (i+1):N_atoms
            if lattice === nothing
                dx, dy, dz = input[j].coord[1:3] .- input[i].coord[1:3]
                d = sqrt(dx^2 + dy^2 + dz^2 + ϵ)
            else
                d = d_pbc(input[i].coord[1:3], input[j].coord[1:3], lattice)
            end

            # scrivi in distanze di i
            dist_lists[i][1, positions[i]] = d
            positions[i] += 1

            # scrivi in distanze di j
            dist_lists[j][1, positions[j]] = d
            positions[j] += 1
        end
    end

    # costruisci i G1Input
    output = Vector{G1Input}(undef, N_atoms)
    for i in 1:N_atoms
        output[i] = G1Input(input[i].species, dist_lists[i])
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
    build_branch(atom::String, G1_number::Int, R_cutoff::Float32; depth=2, seed=nothing) -> Chain

Construct a per-species neural network subbranch for atomic energy prediction.

# Arguments
- `atom::String`: Atomic species name.
- `G1_number::Int`: Number of radial G1 symmetry functions.
- `R_cutoff::Float32`: Cutoff radius for the G1Layer.
- `depth::Int` (optional): Number of hidden layers. `1` or `2`. Default = 2.
- `seed::Int` or `nothing` (optional): RNG seed for G1Layer initialization.

# Returns
- `Chain`: Flux.jl neural network chain consisting of:
    - `G1Layer` with scaled atomic charge.
    - LayerNorm layers.
    - Dense layers with `swish` activation.
    - Final Dense layer outputs scalar atomic energy.
"""




function build_branch(Atom_name::String, G1_number::Int, R_cutoff::Float32 , depth::Int = 2 ; seed::Union{Int,Nothing} = nothing)
    ion_charge = element_to_charge[Atom_name]
    if depth == 2
    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge) ; seed),
        LayerNorm(G1_number),
        Dense(G1_number, 64, swish), 
        LayerNorm(64),
        Dense(64, 32, swish),
        LayerNorm(32),
        Dense(32, 8, swish),
        LayerNorm(8),
        Dense(8, 1)
    )
    elseif depth == 1

    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge)),
        LayerNorm(G1_number),
        Dense(G1_number , 16 , swish),
        LayerNorm(16),
        Dense(16,8, swish),
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
    dispatch_train(distances::Vector{G1Input}, species_models::Vector{Chain})

Internal function for training.  
Applies the correct model to each atom in the input vector of distances,  
based on its `species` field.  

# Arguments
- `distances::Vector{G1Input}`: A batch of atomic inputs,  
   where each element contains the atom's `species` ID and its neighbor distances.  
- `species_models::Vector{Chain}`: One neural network model per species,  
   indexed by the integer `species` ID.

# Returns
- `output::Float32`: The aggregated scalar prediction for the whole batch  
   (sum of per-atom outputs).
"""
function dispatch_train(distances::Vector{G1Input}, species_models::Vector{Chain})
    n_atoms = length(distances)
    outputs = Vector{Float32}(undef, n_atoms)

    @inbounds for i in 1:n_atoms
        distance = distances[i]
        model = species_models[distance.species]
        outputs[i] = model(distance.dist)[1]  # assumes scalar output
    end

    return sum(outputs)
end


"""
    dispatch_train(distances::Matrix{G1Input}, species_models::Vector{Chain})

Internal function for training.  
Handles a batched input of distances. Each column corresponds to one atom,  
and each row to one batch element. For each atom, the correct species model  
is applied across the batch.  

# Arguments
- `distances::Matrix{G1Input}`: Batched atomic inputs.  
   - Size: `(n_batches, n_atoms)`  
   - Each entry holds `species::Int` and `dist::Vector{Float32}`.  
   - Species is assumed to be the same across all batches for a given atom.  
- `species_models::Vector{Chain}`: One neural network model per species.

# Returns
- `outputs::Vector{Float32}`: A vector of length `n_batches`,  
   each element is the aggregated prediction (sum over atoms)  
   for that batch.
"""
function dispatch_train(distances::Matrix{G1Input}, species_models::Vector{Chain})
    n_batches, n_atoms = size(distances)
    outputs = Matrix{Float32}(undef, n_batches, n_atoms)

    @inbounds for i in 1:n_atoms
        # Extract one atom across all batches
        distance_col = distances[:, i]

        # Determine correct model (species consistent across batches)
        model = species_models[distance_col[1].species]

        # Preallocate batch input buffer
        n_neighbors = length(distance_col[1].dist)
        batch_input = Array{Float32}(undef, n_batches, n_neighbors)

        # Fill buffer
        for b in 1:n_batches
            batch_input[b, :] = distance_col[b].dist
        end

        # Forward pass (produces one scalar per batch element)
        outputs[:, i] = vec(model(batch_input))
    end

    # Aggregate per-atom predictions into per-batch scalars
    return dropdims(sum(outputs, dims = 2); dims = 2)
end


"""
    dispatch(atoms, species_models::Vector{Chain}; lattice::Union{Nothing, Matrix{Float32}} = nothing)

Public API function.  
Computes the distance representation of a set of atoms (optionally within a lattice),  
then applies the appropriate species-specific models via `dispatch_train`.  

# Arguments
- `atoms`: Atomic structure input, suitable for `distance_layer`.  
- `species_models::Vector{Chain}`: One neural network model per species.  
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional lattice matrix for periodic systems.  
   Defaults to `nothing` (no periodic boundary conditions).

# Returns
- `outputs`: Model predictions, either a scalar (single batch)  
   or a vector (batched input), depending on the input format.
"""
function dispatch(atoms, species_models::Vector{Chain}; lattice::Union{Nothing, Matrix{Float32}} = nothing)
    distances = distance_layer(atoms; lattice = lattice)
    return dispatch_train(distances, species_models)
end


"""
    predict_forces(x, model; flat=false) -> Array{Float32}

Compute predicted atomic forces for a batch of structures using a trained model.

# Arguments
- `x`: Input atomic structures, either a `Matrix{Vector{AtomInput}}` (batched) or compatible with `distance_layer`.
- `model`: Species-specific neural network models used for force prediction.
- `flat::Bool` (optional): If `true`, return forces flattened as a 1D vector;  
  if `false` (default), return a 3D array `(n_batches, n_atoms, 3)`.

# Returns
- `Array{Float32}`: Predicted forces for all atoms:
    - `(n_batches, n_atoms, 3)` if `flat=false`
    - Flattened 1D array if `flat=true`

# Description
1. Compute pairwise distances with `distance_layer`.
2. Compute distance derivatives with `distance_derivatives`.
3. For each batch, compute force contributions from model gradients.
4. Optionally flatten the resulting force array.
"""




function predict_forces(x , model ; flat = false)
    dist = distance_layer(x)
    derivatives = distance_derivatives(x)

    n_batches , _ = size(x)
    n_atoms = size(x[1] , 1)

    predicted_forces = zeros(Float32 , (n_batches , n_atoms , 3))

    for b in 1:n_batches
        grad = calculate_force(dist[b, :], model)
        temp = zeros(Float32, n_atoms , 3)
        for i in 1:n_atoms

            contrib =  (grad[i] * derivatives[b,i, :, :])  # (1,3)
            temp[i , 1:3] .= vec(contrib)
       
        end

        predicted_forces[b, : , :] .= temp
    end

    if flat
 
        return vcat(predicted_forces...)
    else
        return predicted_forces
    end
end

