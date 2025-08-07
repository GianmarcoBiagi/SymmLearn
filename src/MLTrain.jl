using Flux
using Random
using ProgressMeter
using Statistics
using Enzyme



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
    W_eta::AbstractArray  # Weights for the "eta" connection
    W_Fs::AbstractArray   # Weights for the "Fs" connection
    cutoff::Float32       # Cutoff radius
    charge::Float32       # Atomic charge
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
    W_eta = 0.25f0 .+ 2.25f0 .* rand(Float32, N_G1)  # Initialize eta weights (Float32)
    W_Fs = 0.25f0 .+ 2.25f0 .* rand(Float32, N_G1)   # Initialize Fs weights (Float32)
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



function (layer::G1Layer)(
  x::AbstractMatrix{Float32}
  )

    batch_size, N_neighbors = size(x)
    N_G1 = length(layer.W_eta)

    x_expanded = reshape(x, 1, batch_size, N_neighbors)
    W_Fs_expanded = reshape(layer.W_Fs, N_G1, 1, 1)
    W_eta_expanded = reshape(layer.W_eta, N_G1, 1, 1)

    fc_x = fc.(x_expanded, layer.cutoff)
    diff_sq = (x_expanded .- W_Fs_expanded).^2
    exp_term = exp.(-diff_sq .* W_eta_expanded)
    contribution = layer.charge .* fc_x .* exp_term
    sum_over_neighbors = sum(contribution, dims=3)

    return dropdims(sum_over_neighbors, dims=3) # (batch_size, hidden_dim)
end




"""
    distance_layer(x::Array{Float32, 2}, central_atom_idx::Int) -> Array{Float32, 2}

Computes the distances between a central atom and all other atoms in each structure of a batch.

This function is typically used in atom-centered neural networks. It returns pairwise distances 
from a specified central atom to all others in the structure, **excluding the self-distance**, 
and assumes input coordinates are given in a flat `(batch_size, num_atoms * 3)` format.

### Arguments
- `x::Array{Float32, 2}`: Tensor of shape `(batch_size, num_atoms * 3)`, where each row contains the flattened 3D coordinates of atoms.
- `central_atom_idx::Int`: 1-based index of the central atom for which distances are computed.

### Returns
- `Array{Float32, 2}`: Matrix of shape `(batch_size, num_atoms - 1)` with distances from the central atom to all others, excluding itself.

### Notes
- The self-distance is **not calculated at all**, improving efficiency.
- Atom ordering in the output matches the input, with the central atom removed.
- A small epsilon (`eps(Float32)`) is added before taking the square root to avoid numerical instability.
- Periodic boundary conditions (PBC) are not currently supported in this version.

### Example
```julia
x = rand(Float32, 8, 5 * 3)          # 8 structures, 5 atoms each (flattened coords)
distances = distance_layer(x, 2)     # Distances from atom 2 to all others


"""


function distance_layer(x, idx::Int)
    B, total_coords = size(x)
    N_atoms = div(total_coords, 3)

    coords = reshape(x, B, 3, N_atoms)     # (B, 3, N)
    coords = permutedims(coords, (1, 3, 2))  # (B, N, 3)

    xi = reshape(coords[:, idx, :], B, 1, 3)

    mask = trues(N_atoms)
    mask[idx] = false
    dx = coords[:, mask, :] .- xi  # (B, N-1, 3)

    dx2 = sum(dx .^ 2, dims=3)
    distances = sqrt.(dx2 .+ eps(Float32))[:, :, 1]

    return distances
end






"""
    build_branch(atom::String, G1_number::Int, R_cutoff::Float32) -> Chain

Builds a per-species subnetwork consisting of:
- A `G1Layer` configured with species charge.
- A single Dense layer with tanh activation to output scalar atomic energy.

The atomic charge is scaled from `element_to_charge` using a factor of 0.1.
"""



function build_branch(Atom_name::String, G1_number::Int, R_cutoff::Float32)
    ion_charge = 0.1f0 * element_to_charge[Atom_name]
    return Chain(
        G1Layer(G1_number, R_cutoff, ion_charge),
        Dense(G1_number, 15, tanh),
        Dense(15, 10, tanh),
        Dense(10,5, tanh),
        Dense(5, 1)
    )
end


"""
    build_model(
        species_order::Vector{String},
        G1_number::Int,
        R_cutoff::Float32;
        lattice::Union{Nothing, Matrix{Float32}} = nothing
    ) -> SumBranchesLayer

Construct a Flux model that predicts the total energy of a molecular structure by summing atom-wise energy predictions.

For each atom, the model computes its distance to all other atoms, processes the resulting distances through a 
species-specific subnetwork, and sums all atomic contributions to obtain the total energy.

### Arguments
- `species_order::Vector{String}`: Vector listing the chemical species (e.g., `"H"`, `"C"`, `"O"`) of each atom in the structure.
- `G1_number::Int`: Number of radial symmetry functions (G1) used in the input layer.
- `R_cutoff::Float32`: Cutoff radius applied in the `G1Layer` to limit neighbor interactions.
- `lattice::Union{Nothing, Matrix{Float32}}`: Optional `3×3` lattice matrix. If provided, periodic boundary conditions can be applied (⚠️ not currently used in implementation).

### Returns
- A Flux-compatible model (`SumBranchesLayer`) composed of:
    - One `BranchLayer` per atom in the input, each handling distance computation and atom-specific energy prediction.
    - The outputs from all branches are summed to obtain the total predicted energy.

### Notes
- Each atomic branch uses a `DistanceLayer` to compute distances, and a `G1Layer` + Dense layer to predict energy.
- The current implementation assumes flattened coordinate inputs of shape `(batch_size, N_atoms * 3)`.

### Example
```julia
species_order = ["H", "H", "O"]
model = build_total_model_inline(species_order, 5, 6.0f0)
y_pred = model(x)  # where x is a tuple of size-3 coordinate arrays
"""

function build_model(
    species_order::Vector{String},
    G1_number::Int,
    R_cutoff::Float32
) 

    # Costruisci modelli specie unici
    unique_species = collect(Set(species_order))
    species_models = Dict{String, Chain}()
    for sp in unique_species
        species_models[sp] = build_branch(sp, G1_number, R_cutoff)
    end

    # Numero atomi
    N = length(species_order)

    # Crea i branch per ogni atomo con DistanceLayer + modello specie
    branches = ntuple(i -> Chain(
        x -> distance_layer(x, i),
        species_models[species_order[i]]
    ), N)

    # Parallelizza i branch
    parallel = Parallel(vcat,branches...)

    # Somma le energie atomiche per batch (output finale)
    sum_layer = x_tuple -> sum(x_tuple; dims=1)[:]

    # Modello completo
    return Chain(parallel, sum_layer)

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


function loss_function(model, data, target ; lambda = 0.1)
    # data shape: (batch_size, N_atoms, features)
    # model output shape: (batch_size,)


    if typeof(target) == Dict{Symbol, Any}

      energies = target[:energy]
      forces = target[:forces]'

    else

      energies = [d[:energy] for d in target]
      forces = transpose(hcat([d[:forces] for d in target]...))

    end
 
    energies_preds = model(data)

    forces_preds = calculate_forces(model,data)

    # forward pass on entire batch
    energy_losses = (energies .- energies_preds) .^ 2

    force_losses = mean(((forces .- forces_preds) .^ 2) , dims = 2)


    losses = energy_losses .+ lambda .* force_losses

    return mean(losses)            # mean loss over batch
end

function calculate_forces(model, data)
    n_batches, n_coord = size(data)

    # Funzione helper per calcolare la forza per una singola riga
    function calc_force(x_row)
        dx_i = zeros(Float32, size(x_row))  # non può essere evitato per Enzyme
        grad = Flux.gradient((m, x) -> m(x)[1], Const(model), Enzyme.Duplicated(x_row, dx_i))[2]
        return -grad
    end

    # Applica la funzione a ogni riga di data, restituisce matrice forze
    forces = map(1:n_batches) do i
        x_row = data[i:i, :]  # sottoarray 1×n
        calc_force(x_row)
    end

    return reduce(vcat, forces)  # stacka tutte le forze in una matrice batch × coord
end





"""
    loss_function_no_forces(model, data, energies)

check documentation for loss_function, this is the same but without using the forces


"""


function loss_function_no_forces(model, data, target::Union{Vector{Dict{Symbol, Any}}, Dict{Symbol, Any}})

    if typeof(target) == Dict{Symbol, Any}

      energy = target[:energy]

    else

      energy = [d[:energy] for d in target]

    end
    
    preds = model(data)

    loss = abs2.(preds .- energy) 

    return mean(loss)          
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

- `loss_train::Vector{Float32}`:  
  Training loss per epoch.

- `loss_val::Vector{Float32}`:  
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
  initial_lr=0.1, min_lr=1e-5, decay_factor=0.5, patience=25,
  epochs=3000, batch_size=32, verbose=true
)

  opt = Flux.setup(Adam(initial_lr), model)


  current_lr = initial_lr
  best_model = deepcopy(model)
  best_epoch = 0
  best_loss = Inf
  no_improve_count = 0

  loss_train = zeros(Float32, epochs)
  loss_val = zeros(Float32, epochs)

  @showprogress for epoch in 1:epochs
    # Shuffle data
    idx = randperm(size(x_train, 1))
    x_train = x_train[idx, :]
    y_train = y_train[idx]

    for i in 1:batch_size:size(x_train, 1)
      end_idx = min(i + batch_size - 1, size(x_train, 1))
      x_batch = x_train[i:end_idx, :]
      y_batch = y_train[i:end_idx]

      dup_model = Enzyme.Duplicated(model)

      grads = Flux.gradient((m,x,y) -> loss_function(m,x,y), dup_model, Const(x_batch), Const(y_batch))
      

      Flux.update!(opt, model, grads[1])

      
    end

    # Loss evaluation
    loss_train[epoch] = loss_function(model, x_train , y_train)
    loss_val[epoch] = loss_function(model, x_val , y_val)

    if verbose && epoch%10 == 0
      println("Epoch $(lpad(epoch,4)) | Train Loss: $(round(loss_train[epoch], digits=6)) | Val Loss: $(round(loss_val[epoch], digits=6))")
    end


    # Model checkpoint
    if loss_val[epoch] < best_loss * 0.98
      best_loss = loss_val[epoch]
      best_epoch = epoch
      best_model = deepcopy(model)
      no_improve_count = 0
    else
      no_improve_count += 1
    end

    # Learning rate decay
    if no_improve_count >= patience
      new_lr = max(current_lr * decay_factor, min_lr)
      opt = Flux.setup(Adam(new_lr), model)
      current_lr = new_lr
      no_improve_count = 0
      if verbose
        println("Reducing learning rate to $new_lr at epoch $epoch")
      end
    end
  end

  if verbose
    println("Final Train Loss: $(loss_function(model, x_train, y_train))")
    println("Final Val Loss:   $(loss_function(model, x_val, y_val))")
    println("Best Model Found at Epoch $best_epoch with Val Loss: $best_loss")
  end

  return model, best_model, loss_train, loss_val
end
