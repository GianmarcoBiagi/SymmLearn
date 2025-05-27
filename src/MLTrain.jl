using Flux
using Random
using ProgressMeter
using Statistics
using Zygote

"""
    struct MyLayer

A custom layer structure that contains two sets of weights for neural network connections, as well as additional parameters related to the system.

#### Fields:
- `W_eta::AbstractArray`
  - A weight matrix for the connection "eta" in the neural network layer. This represents the parameters associated with the input feature "eta".
- `W_Fs::AbstractArray`
  - A weight matrix for the connection "Fs" in the neural network layer. This represents the parameters associated with the input feature "Fs".
- `cutoff::Float32`
  - A scalar value representing the cutoff radius for interactions within the system.
- `charge::Float32`
  - A scalar value representing the atomic charge for the layer.

### Example:
```julia
# Create a MyLayer object with example data
W_eta_example = rand(5, 5)  # Example weight matrix for "eta" with 5x5 dimensions
W_Fs_example = rand(5, 5)   # Example weight matrix for "Fs" with 5x5 dimensions
cutoff_example = 5.0f0      # Example cutoff radius value
charge_example = 1.0f0      # Example atomic charge value

# Create the MyLayer object
layer = MyLayer(W_eta_example, W_Fs_example, cutoff_example, charge_example)

# Print the values
println("Layer created: ", layer)

"""
struct MyLayer
    W_eta::AbstractArray  # Weights for the "eta" connection
    W_Fs::AbstractArray   # Weights for the "Fs" connection
    cutoff::Float32       # Cutoff radius
    charge::Float32       # Atomic charge
end

Flux.@layer MyLayer

"""
    MyLayer(input_dim::Int, hidden_dim::Int, cutoff::Float32, charge::Float32) -> MyLayer

Creates an instance of the custom `MyLayer` layer with the specified input dimension, hidden dimension,
cutoff radius, and atomic charge. The layer is initialized with random weights for the `eta` and `Fs` parameters.

### Arguments:
- `input_dim::Int`: The number of input features (dimension of the input vector).
- `hidden_dim::Int`: The number of neurons in the hidden layer.
- `cutoff::Float32`: The cutoff radius for the layer's calculations.
- `charge::Float32`: The atomic charge associated with the layer.

### Returns:
- An instance of the `MyLayer` layer with random weights for `eta` and `Fs`.
"""
function MyLayer(input_dim::Int, hidden_dim::Int, cutoff::Float32, charge::Float32)
    # Initialize weights for eta and Fs with random values
    W_eta = 0.25f0 .+ 2.25f0 .* rand(Float32, hidden_dim, input_dim)  # Initialize eta weights (Float32)
    W_Fs = 0.25f0 .+ 2.25f0 .* rand(Float32, hidden_dim, input_dim)   # Initialize Fs weights (Float32)

    # Create and return the MyLayer instance
    return MyLayer(W_eta, W_Fs, cutoff, charge)
end



"""
    (layer::MyLayer)(x)

Forward pass for the custom `MyLayer` neural network layer over a batch of inputs.

This function computes the output of `MyLayer` by summing the contributions from all neighboring atoms 
for each sample in the batch. Each contribution is calculated using the cutoff function `fc`, the difference 
between the input `x` and the learned weights `W_Fs`, the exponential decay weighted by `W_eta`, 
and scaled by the atomic charge.

### Arguments:
- `layer::MyLayer`: The layer instance containing parameters:
    - `W_eta`: Weights controlling the decay width of each function, shape `(hidden_dim, input_dim)`.
    - `W_Fs`: Weights representing the peak positions, shape `(hidden_dim, input_dim)`.
    - `cutoff`: Cutoff radius applied via the function `fc`.
    - `charge`: Atomic charge used as a scaling factor.
- `x`: A 2D array of shape `(batch_size, N_neighbors)` where each row corresponds to one sample's distances
  to neighboring atoms.

### Returns:
- A 2D array of shape `(hidden_dim, batch_size)` containing the output activations for each batch sample.

### Example:
```julia
layer = MyLayer(1, 5, 2.5f0, 1.0f0)  # Layer with 1 input dim, 5 features, cutoff=2.5, charge=1.0
x = rand(Float32, 3, 10)             # Batch of 3 samples, each with 10 distances
output = layer(x)                    # Forward pass producing output of shape (3, 5)
println(output)
"""

function (layer::MyLayer)(x::AbstractMatrix{Float32})  # x shape: (batch_size, N_neighbors)
    batch_size, N_neighbors = size(x)
    hidden_dim, input_dim = size(layer.W_eta)  # input_dim should be 1 here

    # Reshape x to (batch_size, 1, N_neighbors) to broadcast with weights (hidden_dim, 1)
    # Actually we want to broadcast over hidden_dim, batch_size, N_neighbors

    # Expand weights W_eta and W_Fs from (hidden_dim, input_dim=1) to (hidden_dim, batch_size, N_neighbors)
    # Expand x from (batch_size, N_neighbors) to (1, batch_size, N_neighbors)

    # Use broadcasting:
  

    x_expanded = reshape(x, 1, batch_size, N_neighbors)          # (1, batch_size, N_neighbors)
    W_Fs_expanded = reshape(layer.W_Fs, hidden_dim, 1, 1)       # (hidden_dim, 1, 1)
    W_eta_expanded = reshape(layer.W_eta, hidden_dim, 1, 1)     # (hidden_dim, 1, 1)

    # Compute fc(x), assuming fc is scalar function applied elementwise
    fc_x = fc.(x_expanded, layer.cutoff)                         # (1, batch_size, N_neighbors)

    # Broadcast calculation
    # diff squared: (x - W_Fs)^2
    diff_sq = (x_expanded .- W_Fs_expanded) .^ 2                # (hidden_dim, batch_size, N_neighbors)

    # exponent term
    exp_term = exp.(-diff_sq .* W_eta_expanded)                  # (hidden_dim, batch_size, N_neighbors)

    # total contribution per hidden_dim, batch, neighbor
    contribution = layer.charge .* fc_x .* exp_term              # broadcasting fc_x (1,batch,neighbors) over hidden_dim

    # sum over neighbors axis (3rd dim)
    sum_over_neighbors = sum(contribution, dims=3)               # shape (hidden_dim, batch_size, 1)

    # permute dims to (batch_size, hidden_dim)
    output = permutedims(dropdims(sum_over_neighbors, dims=3), (2, 1))  # (batch_size, hidden_dim)
    return output'
end

"""
    distance_layer(x::Array{Float32, 3}, central_atom_idx::Int, lattice::Union{Nothing, Matrix{Float32}}=nothing) -> Array{Float32, 2}

Compute distances between a central atom and all other atoms in a batch of atomic structures.

This function is typically used as a preprocessing step in atom-centered neural network models.
It calculates the pairwise distances between a specified central atom and all other atoms
in each structure of a batch. Optional periodic boundary conditions (PBC) are supported.

### Arguments
- `x::Array{Float32, 3}`: A tensor of shape `(batch_size, num_atoms, 3)`, where each structure contains `num_atoms` atoms with 3D coordinates.
- `central_atom_idx::Int`: Index of the central atom in each structure (1-based indexing).
- `lattice::Union{Nothing, Matrix{Float32}}=nothing`: Optional 3×3 lattice matrix. If provided, PBC are applied using the minimum image convention.

### Returns
- `Array{Float32, 2}`: A matrix of shape `(batch_size, num_atoms - 1)` containing distances from the central atom to all other atoms for each structure in the batch. The distance to the central atom itself is excluded.

### Notes
- The output excludes the distance from the central atom to itself.
- If `lattice` is specified, distances are computed using periodic boundary conditions.
- The order of output distances matches the input atom order, with the central atom removed.

### Example
```julia
x = rand(Float32, 8, 5, 3)  # 8 structures, each with 5 atoms in 3D
distances = distance_layer(x, 2)  # Compute distances from atom 2
"""


function distance_layer(x::Array{Float32, 3}, central_atom_idx::Int, lattice::Union{Nothing, Matrix{Float32}}=nothing)
    batch_size, N, _ = size(x)
    use_pbc = lattice !== nothing
    inv_lat = use_pbc ? inv(lattice) : nothing

    xi = reshape(view(x, :, central_atom_idx, :), batch_size, 1, :)
    dx = x .- xi
"""
    if use_pbc
        dx = map(1:batch_size) do b
            map(1:N) do a
                dx_ba = dx[b, a, :]
                dx_frac = inv_lat * dx_ba
                dx_frac_wrapped = dx_frac .- round.(dx_frac)
                (lattice * dx_frac_wrapped)'  # row vector
            end |> x -> reduce(vcat, x)'  # (N, 3)
        end |> x -> reduce(hcat, x) |> x -> reshape(x, batch_size, N, 3)
    end"""

    distances_all = sqrt.(sum(dx.^2, dims=3))[:, :, 1]

    if central_atom_idx == 1
        distances = distances_all[:, 2:end]
    elseif central_atom_idx == N
        distances = distances_all[:, 1:end-1]
    else
        distances = hcat(distances_all[:, 1:central_atom_idx-1], distances_all[:, central_atom_idx+1:end])
    end

    return distances
end







"""
    assemble_model(
      species_models::Dict{String,Chain},
      species_order::Vector{String},
      lattice::Union{Nothing, Matrix{Float32}} = nothing
    ) -> Chain

Constructs a composite Flux model that, for each atom in the structure:
1. Computes interatomic distances from atomic coordinates using `DistanceLayer`.
2. Applies the appropriate species-specific subnetwork to predict the per-atom energy.
3. Sums all atomic energies to obtain the total energy per structure.

### Arguments:
- `species_models::Dict{String,Chain}`  
  A dictionary mapping each atomic species (e.g., `"H"`, `"C"`, `"O"`) to its
  corresponding `Flux.Chain` subnetwork.

- `species_order::Vector{String}`  
  A length-N vector specifying, for atom positions `1…N`, which species occupies each
  slot. Must only contain species keys from `species_models`.

- `lattice::Union{Nothing, Matrix{Float32}} = nothing`  
  The lattice matrix used for periodic boundary conditions (PBC). If `nothing`, no PBC
  will be applied. This lattice is shared across all `DistanceLayer`s for simplicity.

### Returns:
- `Chain`  
  A Flux model where each branch processes one atom:
  - `DistanceLayer(i, lattice)` computes distances from atom `i` to all others.
  - The corresponding species-specific model processes the distances.
  The outputs (scalars) from all branches are summed to return the total energy.

### Example:
```julia
species_list = ["H", "O"]
species_models = create_species_models(species_list, G1_number, R_cutoff)
species_order = ["H", "H", "O"]
model = assemble_model(species_models, species_order)

# Input `x` should be a tensor of shape (num_atoms, 3, batch_size)
# Then: model(ntuple(i -> x[i, :, :], num_atoms))


"""

function assemble_model(
  species_models::Dict{String,Chain},
  species_order::Vector{String},
  lattice::Union{Nothing, Matrix{Float32}} = nothing
)
  N = length(species_order)

  branches = ntuple(i -> Chain(
      x -> distance_layer(x, i, lattice),
      species_models[species_order[i]]  
  ), N)

  parallel = Parallel(vcat, branches...)
  
  # This sums each structure's atomic energies to total energy, per batch
  sum_layer = x_tuple -> sum(x_tuple; dims=1)[:]  # restituisce un vettore 1D

  



  return Chain(parallel, sum_layer)
end






"""
    create_species_models(species::Vector{String}, G1_number::Int, R_cutoff::Float32) -> Dict{String, Chain}

Given a list of unique atomic species, construct and return a dictionary mapping
each species name to its corresponding Flux.Chain model.  Each model branch
is built by calling `build_branch(name, G1_number, R_cutoff)`.

# Arguments
- `species::Vector{String}`: List of species , e.g. `["H","H","C","O","O",...]`.
- `G1_number::Int`: Number of symmetry‑function features fed into the branch.
- `R_cutoff::Float32`: Cutoff radius used by the custom `MyLayer`.

# Returns
- `Dict{String, Chain}`: A dictionary where
    `dict["H"]` is the Chain for hydrogen,
    `dict["C"]` is the Chain for carbon, etc.
"""
function create_species_models(
    species::Vector{String},
    G1_number::Int,
    R_cutoff::Float32
)   
    unique_species = Set(species)
    unique_species = collect(unique_species)
    models = Dict{String, Chain}()
    for sp in unique_species
        # build_branch returns a Flux.Chain for that species
        models[sp] = build_branch(sp, G1_number, R_cutoff)
    end
    return models
end



"""
    create_species_models(
        species::Vector{String},
        G1_number::Int,
        R_cutoff::Float32
    ) -> Dict{String, Chain}

Creates and returns a dictionary mapping each unique atomic species to its corresponding
Flux.Chain model. Each model is built using the `build_branch` function, which defines
a subnetwork architecture specific to that species.

### Arguments:
- `species::Vector{String}`:  
  List of atomic species appearing in the dataset, e.g. `["H", "H", "C", "O", "O", …]`.

- `G1_number::Int`:  
  Number of symmetry function (G1) features passed to each branch.

- `R_cutoff::Float32`:  
  Cutoff radius passed to the custom `MyLayer`, used to define interaction range.

### Returns:
- `Dict{String, Chain}`:  
  A dictionary mapping each species name to its corresponding Flux.Chain model.
  For example:  
  - `dict["H"]` is the model for hydrogen  
  - `dict["C"]` is the model for carbon  
  - etc.

### Example:
```julia
species_list = ["H", "C", "O", "H", "O"]
G1_number = 5
R_cutoff = 6.0f0
species_models = create_species_models(species_list, G1_number, R_cutoff)

"""

function build_branch(Atom_name::String,G1_number::Int,R_cutoff::Float32)
    ion_charge = 0.1f0 * element_to_charge[Atom_name]
    return Chain(
        MyLayer(1, G1_number, R_cutoff, ion_charge),
        Dense(G1_number, 15, tanh),
        Dense(15, 10, tanh),
        Dense(10, 5, tanh),
        Dense(5, 1)
    )
end

"""
    loss_function(model, data, energies)

Computes the mean squared error (MSE) loss between predicted and reference total energies.

# Arguments
- `model::Function`:  
  A callable model (e.g. a `Flux.Chain`) that maps input structures to predicted energies.

- `data::Vector`:  
  A vector of structures, where each structure is typically a tuple of atomic features 
  (e.g. `(x[:,1,:], x[:,2,:], ..., x[:,N,:])`).

- `energies::Vector{Float32}`:  
  A 1D array of reference total energies, one per structure.

# Returns
- `Float64`:  
  The mean squared error (MSE) between predicted and reference energies.

# Notes
- This function applies the model to each structure in `data` using broadcasting (`model.(data)`).
- It assumes the model returns a scalar energy per structure.
- The final loss is computed as the mean of squared differences.

# Example
```julia
species = ["H", "C", "O"]
species_order = ["H", "H", "C", "O", ..., "H"]  # length 40
models = create_species_models(species, 5, 6.0f0)
model = assemble_model(models, species_order)

data = [ntuple(i -> rand(Float32, 32), 40) for _ in 1:10]  # batch of 10 structures
energies = rand(Float32, 10)  # reference energies

loss = loss_function(model, data, energies)
println("Loss: ", loss)

"""


function loss_function(model, data, target::Vector{Dict{Symbol, Any}}; lambda = 0.1)
    # data shape: (batch_size, N_atoms, features)
    # model output shape: (batch_size,)
    energy = [d[:energy] for d in target]
    forces = [d[:forces] for d in target]
 
    
    preds = model(data)  
    forces_preds = calculate_forces(model,data)
    # forward pass on entire batch
    energy_losses = abs2.(preds .- energy)  # elementwise squared error
    #force_losses = abs2.(forces_preds .- forces)
    force_losses = 0
    losses = energy_losses 
    
    return mean(losses)            # mean loss over batch
end

function calculate_forces(model,data)

  n_batches=size(data)[1]
  #forces=[Zygote.gradient(model,data[i,:,:]) for i in n_batches]
  #println(forces)
  forces=0
  return(forces)
  
end




"""
    train_model!(model, x_train, y_train, x_val, y_val, loss_function; 
                 initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
                 epochs=3000, batch_size=32, verbose=true)

Trains a neural network model to predict total energies of atomic structures.

# Arguments
- `model::Flux.Chain`:  
  The model to train (typically combining branches for each species).

- `x_train::Any`:  
  Training data, as a vector of structures. Each structure is a tuple of atomic features.

- `y_train::Vector{Float32}`:  
  Ground truth total energies for the training structures.

- `x_val::Any`:  
  Validation data, same format as `x_train`.

- `y_val::Vector{Float32}`:  
  Ground truth total energies for the validation structures.

- `loss_function::Function`:  
  Function that computes the loss as `loss_function(model, data, targets)`.

- `initial_lr::Float32=0.01`:  
  Initial learning rate for the `Adam` optimizer.

- `min_lr::Float32=1e-5`:  
  Minimum learning rate before stopping further decay.

- `decay_factor::Float32=0.5`:  
  Factor by which to reduce the learning rate when validation loss plateaus.

- `patience::Int=25`:  
  Number of epochs with no significant validation improvement before decaying the learning rate.

- `epochs::Int=3000`:  
  Maximum number of training epochs.

- `batch_size::Int=32`:  
  Number of structures per batch during training.

- `verbose::Bool=true`:  
  Whether to print training status and updates.

# Returns
- `best_model::Flux.Chain`:  
  The best-performing model on the validation set.

- `loss_train::Vector{Float64}`:  
  Training loss per epoch.

- `loss_val::Vector{Float64}`:  
  Validation loss per epoch.

# Notes
- The learning rate is adaptively reduced if validation loss does not improve for `patience` epochs.
- The best model (lowest validation loss) is saved and returned.
"""

function train_model!(
  model,
  x_train::Any, 
  y_train::Vector{Dict{Symbol, Any}}, 
  x_val::Any, 
  y_val::Vector{Dict{Symbol, Any}}, 
  loss_function::Function;
  initial_lr=0.01, min_lr=1e-5, decay_factor=0.5, patience=25, 
  epochs=3000, batch_size=32, verbose=true
)

  # Initialize optimizer
  opt_state = Flux.setup(Adam(initial_lr), model)

  # Store best models and best loss
  best_epoch = 0
  best_loss = Inf
  best_model = nothing

  # Loss tracking
  loss_train = zeros(Float32, epochs)
  loss_val = zeros(Float32, epochs)
  no_improve_count = 0

 

  @showprogress for epoch in 1:epochs
      for i in 1:batch_size:size(x_train, 1)
          end_index = min(i + batch_size - 1, size(x_train, 1))
          x_batch = x_train[i:end_index,:,:]
          y_batch = y_train[i:end_index]
          Flux.train!(loss_function, model, [(x_batch, y_batch)], opt_state)
      end

      # Compute losses for training and validation
      loss_train[epoch] = loss_function(model, x_train, y_train)
      loss_val[epoch] = loss_function(model, x_val, y_val)

      # Save best model if improved
      if loss_val[epoch] < best_loss * 0.98
          best_epoch = epoch
          best_loss = loss_val[epoch]
          best_model = deepcopy(model)
          no_improve_count = 0
      else
          no_improve_count += 1
      end

      # Adjust learning rate if no improvement
      if no_improve_count >= patience
          new_lr = max(initial_lr * decay_factor, min_lr)
          opt_state = Flux.setup(Adam(new_lr), model)
          no_improve_count = 0
          if verbose
              println("Reducing learning rate to $(new_lr) at epoch $epoch")
          end
      end
  end

  if verbose
      println("Final Training Loss: $(loss_function(model, x_train, y_train))")
      println("Final Validation Loss: $(loss_function(model, x_val, y_val))")
      println("The best model was found at epoch: $best_epoch")
  end

  return best_model, loss_train, loss_val
end


