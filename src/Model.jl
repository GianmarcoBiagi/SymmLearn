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

    # Riscalare i pesi per broadcasting
    W_eta_exp = reshape(layer.W_eta, n_features, 1, 1)   # (features, 1, 1)
    W_Fs_exp  = reshape(layer.W_Fs,  n_features, 1, 1)   # (features, 1, 1)

    # Espandere input per broadcasting
    x_exp = reshape(x, 1, n_batch, n_neighbors)          # (1, batch, neighbors)

    # Calcolare cutoff e contributi
    fc_x = fc.(x_exp, layer.cutoff)                      # (1, batch, neighbors)
    diff_sq = (x_exp .- W_Fs_exp).^2                     # (features, batch, neighbors)
    exp_term = exp.(-diff_sq .* W_eta_exp)               # (features, batch, neighbors)
    contribution = layer.charge .* fc_x .* exp_term      # (features, batch, neighbors)

    # Sommare su tutti i vicini
    sum_neighbors = sum(contribution, dims=3)           # (features, batch, 1)

    # Rimuovere dimensione singleton e restituire (features, n_batch)

    return dropdims(sum_neighbors, dims=3)
end


"""
    distance_matrix_layer(coords::AbstractMatrix)

Compute the pairwise distances between atoms for a batch of coordinate arrays.

# Arguments
- `coords::AbstractMatrix`: a matrix of shape `(batch, 3N)`, where `N` is the number of atoms
  and each row contains the concatenated x,y,z coordinates of all atoms.

# Returns
- `distances::Array{Float64,3}`: array of shape `(batch, N, N-1)`, where for each atom
  `i` the distances to all other atoms are stored (excluding self-distance).

# Notes
- Implementation avoids dynamic allocations (no `vcat`, no slicing).
- Uses nested loops instead of broadcasting to make it compatible with Enzyme.
"""




function distance_matrix_layer(coords::AbstractMatrix)
    batch, threeN = size(coords)
    N = div(threeN, 3)
    ϵ = Float32(1e-7)

    distances = Array{Float32}(undef, batch, N, N-1)

    @inbounds for b in 1:batch
        for i in 1:N
            idx = 1
            xi = coords[b, 3i-2]
            yi = coords[b, 3i-1]
            zi = coords[b, 3i]
            for j in 1:N
                if j == i
                    continue
                end
                dx = coords[b, 3j-2] - xi
                dy = coords[b, 3j-1] - yi
                dz = coords[b, 3j] - zi
                distances[b, i, idx] = sqrt(dx*dx + dy*dy + dz*dz + ϵ)
                idx += 1
            end
        end
    end

    return distances
end

function distance_matrix_layer(x::AbstractVector{<:AbstractFloat})
    return distance_matrix_layer(reshape(x, 1, :))
end









"""
    build_branch(atom::String, G1_number::Int, R_cutoff::Float32) -> Chain

Builds a per-species subnetwork consisting of:
- A `G1Layer` configured with species charge.
- A single Dense layer with tanh activation to output scalar atomic energy.

The atomic charge is scaled from `element_to_charge` using a factor of 0.1.
"""



function build_branch(Atom_name::String, G1_number::Int, R_cutoff::Float32)
    ion_charge = element_to_charge[Atom_name]
    return Chain(
        G1Layer(G1_number, R_cutoff, Float32(ion_charge)),
        BatchNorm(G1_number),
        Dense(G1_number, 8, tanh), 
        Dense(8, 4, tanh),
        Dense(4, 2, tanh),
        Dense(2, 1)
    )
end


"""
    build_model(
        species_order::Vector{String},
        G1_number::Int,
        R_cutoff::Float32
    ) -> Chain

Builds a Flux model that predicts the total energy of a molecular structure by summing
atom-wise contributions, **sharing subnetwork weights across atoms of the same species**.

### Workflow
1. Computes the full pairwise distance matrix (excluding self-distances) once per batch.
2. Splits the distance matrix rows so that each branch processes one atom.
3. Each branch applies a species-specific subnetwork to its distances.
4. All atomic contributions are summed to obtain the total energy.

### Arguments
- `species_order::Vector{String}`: List of atom species in the fixed order of the input coordinates.
- `G1_number::Int`: Number of G1 radial symmetry functions in the input layer.
- `R_cutoff::Float32`: Cutoff radius for neighbor interactions in `G1Layer`.

### Returns
- `Chain`: A Flux model with structure:
    ```
    Chain(
        distance_matrix_layer_no_self,
        Parallel(vcat, branches...),
        sum_layer
    )
    ```
"""

# Struttura per sommare lungo la prima dimensione
struct SumLayer
    dims::Int
end
(s::SumLayer)(x) = sum(x; dims=s.dims)

# Branch con tipo concreto
struct AtomBranch{M}
    idx::Int
    model::M
end
(ab::AtomBranch)(x) = ab.model(x[:, ab.idx, :])

"""
    build_model(species_order, G1_number, R_cutoff)

Costruisce un modello Chain statico e type-stable,
compatibile con Enzyme.
"""
function build_model(
    species_order::Vector{String},
    G1_number::Int,
    R_cutoff::Float32
)
    # Costruisci branches come NTuple di tipo concreto
    branches = ntuple(i -> AtomBranch(
            i,
            build_branch(species_order[i], G1_number, R_cutoff)
        ),
        length(species_order)
    )

    parallel_layer = Parallel(vcat, branches...)

    return Chain(
        distance_matrix_layer,
        parallel_layer,
        SumLayer(1)  # somma lungo dim=1
    )
end

