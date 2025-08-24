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

Creates an instance of the custom `G1Layer` layer with the specified input dimension, hidden dimension,
cutoff radius, and atomic charge. The layer is initialized with random weights for the `eta` and `Fs` parameters.

### Arguments:
- `N_G1::Int`: The number of G1 symmetry functions used.
- `cutoff::Float32`: The cutoff radius for the layer's calculations.
- `charge::Float32`: The atomic charge associated with the layer.

### Returns:
- An instance of the `G1Layer` layer with random weights for `eta` and `Fs`.
"""
function G1Layer( N_G1::Int, cutoff::Float32, charge::Float32)
    # Initialize weights for eta and Fs with random values
   W_eta = 0.25f0 .+ 2.5f0 .* rand(Float32, N_G1)
    W_Fs  = 0.25f0 .+ 2.5f0 .* rand(Float32, N_G1)

    # Create and return the G1Layer instance
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
    n_features = length(layer.W_eta)

    output = zeros(Float32, n_features, n_batch)

    @inbounds for b in 1:n_batch
        for f in 1:n_features
            # Calcola tutti i contributi dei vicini in-place senza allocazioni temporanee
            s = 0f0
            for n in 1:n_neighbors
                dx = x[b, n] - layer.W_Fs[f]
                s += fc(x[b, n], layer.cutoff) * exp(-layer.W_eta[f] * dx * dx)
            end
            output[f, b] = layer.charge * s
        end
    end

    return output
end




"""
    distance_matrix_layer(input::Matrix{Vector{AtomInput}})

Compute pairwise distances between atoms for each batch in a matrix of batches.

# Arguments
- `input`: A matrix of batches, where each element is a `Vector{AtomInput}`.
  Each `AtomInput` has `.species::Int` and `.features::AbstractVector`
  where `features` contains at least the 3D coordinates.

# Returns
- A matrix of the same size as `input`. Each element is a vector of `AtomInput`.
  For each atom, the `species` is copied from the input, while `features`
  contains the distances from all other atoms in that batch (length `N-1`).
"""
function distance_matrix_layer(input::Matrix{Vector{AtomInput}})
    ϵ = Float32(1e-7)
    
    batches , _ = size(input)
    N_atoms = size(input[1])[1]

    output = Matrix{G1Input}(undef, (batches , N_atoms ))


    for I in 1:batches
        batch = input[I]
        N = length(batch)
        out_batch = Vector{G1Input}(undef, N)

        for i in 1:N
            xi, yi, zi = batch[i].coord[1:3]
            distances = Matrix{Float32}(undef, (1 , N-1))
            idx = 1

            for j in 1:N
                if j == i
                    continue
                end
                xj, yj, zj = batch[j].coord[1:3]
                dx, dy, dz = xj - xi, yj - yi, zj - zi
                distances[1 , idx] = sqrt(dx*dx + dy*dy + dz*dz + ϵ)
                idx += 1
            end

            out_batch[i] = G1Input(batch[i].species, distances)
        end

        output[I , :] = out_batch
    end

    return output
end



function distance_matrix_layer(input::Vector{Vector{AtomInput}})
    ϵ = Float32(1e-7)

    output = Vector{Vector{G1Input}}(undef, length(input))


    for (b, batch) in enumerate(input)
        N = length(batch)
        out_batch = Vector{G1Input}(undef, N)

        for i in 1:N
            xi, yi, zi = batch[i].coord[1:3]
            distances = Matrix{Float32}(undef, (1 , N-1))
            idx = 1

            for j in 1:N
                if j == i
                    continue
                end
                xj, yj, zj = batch[j].coord[1:3]
                dx, dy, dz = xj - xi, yj - yi, zj - zi
                distances[idx] = sqrt(dx*dx + dy*dy + dz*dz + ϵ)
                idx += 1
            end

            # costruiamo un nuovo AtomInput con stessa species ma features = distanze
            out_batch[i] = G1Input(batch[i].species, distances)
        end

        output[b] = out_batch
    end

    return output
end




"""
    build_branch(atom::String, G1_number::Int, R_cutoff::Float32) -> Chain

Builds a per-species subnetwork consisting of:
- A `G1Layer` configured with species charge.
- A single Dense layer with tanh activation to output scalar atomic energy.

The atomic charge is scaled from `element_to_charge` using a factor of 0.1.
"""



function build_branch(Atom_name::String, G1_number::Int, R_cutoff::Float32 , depth::Int)
    ion_charge = element_to_charge[Atom_name]
    if depth == 2
    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge)),
        Dense(G1_number, 15, tanh), 
        Dense(15, 10, tanh),
        Dense(10, 5, tanh),
        Dense(5, 1)
    )
    elseif depth == 1

    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge)),
        Dense(G1_number , 4 , tanh),
        Dense(4, 3,tanh),  
        Dense(3,3,tanh),
        Dense(3, 1)   
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
                              G1_number::Int, R_cutoff::Float32 ; depth = 2::Int)
    n_species = length(unique_species)
    species_models = Vector{Chain}(undef, n_species)
    
    for spec in unique_species
        idx = species_idx[spec]
        species_models[idx] = build_branch(spec, G1_number, R_cutoff , depth)
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
function dispatch(atoms::Vector{Vector{AtomInput}}, species_models::Vector{Chain})

    

    n_batches = size(atoms[1])[1]
    n_atoms = size(atoms[1,1])[1]

    outputs = Vector{Float32}(undef, n_atoms)


    distances = distance_matrix_layer(atoms)


    @inbounds for i in 1:n_atoms
        distance = distances[1][i]
        model = species_models[distance.species]
        outputs[i] = model(distance.dist)[1]  # assuming model outputs a 1-element vector
    end

    outputs = sum(outputs)

    return outputs
end

function dispatch(atoms::Matrix{Vector{AtomInput}}, species_models::Vector{Chain})
    n_batches = size(atoms)[1]
    n_atoms = size(atoms[1,1])[1]


    
    outputs = Matrix{Float32}(undef, (n_batches , n_atoms))


    distances = distance_matrix_layer(atoms)



    @inbounds for i in 1:n_atoms
        distance = distances[: , i]

        model = species_models[distance[1].species]
        
        batch = vcat([d.dist for d in distance]...)

        outputs[: , i] = model(batch) # assuming model outputs a 1-element vector

    end

    final_outputs = sum(outputs , dims = 2)

    return final_outputs
end

