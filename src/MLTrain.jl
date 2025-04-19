using Flux
using Random
using ProgressMeter
using Statistics
using CUDA
using cuDNN

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

Forward pass for the custom `MyLayer` neural network layer.

This function computes the output of `MyLayer` by summing the contributions from all neighboring atoms.
Each contribution is calculated using the cutoff function `fc`, the difference between the input `x[j]` and the learned
weights `W_Fs`, the exponential decay weighted by `W_eta`, and scaled by the atomic charge.

### Arguments:
- `layer::MyLayer`: The layer instance containing parameters:
    - `W_eta`: Weights controlling the decay width of each function.
    - `W_Fs`: Weights representing the peak positions.
    - `cutoff`: Cutoff radius applied via the function `fc`.
    - `charge`: Atomic charge used as a scaling factor.
- `x`: A 1D array where each element `x[j]` corresponds to a distance (or feature) for a neighboring atom.

### Returns:
- A vector of length `hidden_dim` (Float32 elements), representing the output of the layer for the given input.

### Example:
```julia
layer = MyLayer(1, 5, 2.5f0, 1.0f0)   # Layer with 1 input dim, 5 features, cutoff=2.5, charge=1.0
x = rand(10)                          # 10 distances to neighboring atoms
output = layer(x)                    # Forward pass
println(output)

"""

function (layer::MyLayer)(x)
    N_neighboring_atoms = length(x)  # Number of neighboring atoms
    hidden_dim = size(layer.W_eta, 1)   

    # Define a function to compute each contribution
    function contribution(j)
        layer.charge .* fc(x[j], layer.cutoff) .* exp.(-(x[j] .- layer.W_Fs).^2 .* layer.W_eta)
    end

    # Compute the total contribution without mutation
    return sum(contribution(j) for j in 1:N_neighboring_atoms)
end


"""
    Base.show(io::IO, layer::MyLayer)

Custom implementation of the `show` function for the `MyLayer` type.

This function defines how a `MyLayer` object is displayed when printed. It provides a concise textual summary indicating that this is a custom layer implementing the G1 symmetry functions.

### Arguments:
- `io::IO`: The output stream (e.g., `stdout`, file, etc.) where the information should be printed.
- `layer::MyLayer`: The `MyLayer` instance to be displayed.

### Example:
```julia
layer = MyLayer(1, 5, 2.5f0, 1.0f0)
show(stdout, layer)

"""

# Custom implementation of the show function for displaying MyLayer
function Base.show(io::IO, layer::MyLayer)
    print(io, "MyLayer(N_atoms-1 => G1_Number, G1 function)")
end



"""
    assemble_model(
      species_models::Dict{String,Chain},
      species_order::Vector{String}
    ) -> Chain

Constructs a composite Flux model that applies the correct species-specific subnetwork
to each atom in a fixed order, then sums their scalar outputs to produce a total energy
per structure.

### Arguments:
- `species_models::Dict{String,Chain}`  
  A dictionary mapping each atomic species (e.g. `"H"`, `"C"`, `"O"`) to its
  corresponding `Flux.Chain` subnetwork (usually created via `create_species_models`).

- `species_order::Vector{String}`  
  A length-N vector specifying, for atom slots `1…N`, which species occupies each
  slot. Must only contain species keys found in `species_models`.

### Returns:
- `Chain`  
  A `Flux.Chain` where the first layer is a `Flux.Parallel` of N branches,
  each branch being the model corresponding to that atom’s species. The output is
  a tuple of N per-atom scalar predictions (shape `(batch_size,)` each), which the
  final layer reduces into a single `(batch_size,)` vector representing the total
  energy of each structure.

### Example:
```julia
# 1. Define available species and create their models
species_list = ["H", "C", "O"]
species_models = create_species_models(species_list, G1_number, R_cutoff)

# 2. Define atom types for each position in the input
species_order = ["H", "H", "O", "C", …]  # length = 40

# 3. Assemble the full model
model = assemble_model(species_models, species_order)

# 4. Apply to input: a tuple of 40 elements, one per atom
loss(x, y) = Flux.Losses.mse(model(ntuple(i -> x[:, i, :], 40)), y)

# 5. Train
opt = Flux.Adam(1e-3)
Flux.train!(loss, params(model), data, opt)

"""

function assemble_model(
    species_models::Dict{String,Chain},
    species_order::Vector{String}
)
    N = length(species_order)
    branches = ntuple(i -> species_models[species_order[i]], N)
    p = Parallel(vcat,branches...)
    sum_layer = x_tuple -> reduce(+, x_tuple)
    return Chain(p, sum_layer)
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


function loss_function(model, data, energies::Vector{Float32})
    mean(abs2.(model.(data) .- energies))
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
    y_train::Vector{Float32}, 
    x_val::Any, 
    y_val::Vector{Float32}, 
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
            x_batch = x_train[i:end_index]
            y_batch = y_train[i:end_index]
            Flux.train!(loss_function, model, [(x_batch, y_batch)], opt_state)
        end

        # Compute losses
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

