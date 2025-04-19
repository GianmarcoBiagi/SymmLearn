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
    W_eta = Float32.(0.25.+2.25.*rand(hidden_dim, input_dim))  # Initialize eta weights (Float32)
    W_Fs = Float32.(0.25.+2.25.*rand(hidden_dim, input_dim))    # Initialize Fs weights (Float32)

    # Create and return the MyLayer instance
    return MyLayer(W_eta, W_Fs, cutoff, charge)
end

"""
    (layer::MyLayer)(x)

Forward pass for the custom layer `MyLayer`.

This function calculates the output of the `MyLayer` layer by performing a summation over the neighboring atoms. For each neighboring atom, the function computes a contribution based on the input `x`, the layer's parameters `charge`, `W_Fs`, and `W_eta`, as well as a cutoff function `fc`. The result is summed and returned.

### Arguments
- `layer::MyLayer`: The custom layer object containing weights (`W_eta`, `W_Fs`), charge, and cutoff.
- `x`: An array representing the positions or other relevant data of the neighboring atoms.

### Returns
- `Float32`: A summed value for the layer's forward pass, calculated as a `Float32`.

### Example
```julia
layer = MyLayer(input_dim=5, hidden_dim=3, cutoff=1.5f0, charge=1.0f0)
x = rand(10)  # Example input for the neighboring atoms
output = layer(x)  # Forward pass
println(output)
"""

function (layer::MyLayer)(x)
    N_neighboring_atoms = length(x)  # Number of neighboring atoms (assuming x represents atom positions or other data)+
    sum = zeros(Float32, size(layer.W_eta)[1])  # Initialize sum as a zero vector of size hidden_dim
    
    # Iterate over each neighboring atom
    for j in 1:N_neighboring_atoms
        # Sum the contributions from each neighboring atom
        
        if typeof(x[j]) != Float32
            println("Attenzione, questo non è un Float32: ",x[j])
            println(x)
            exit(0)
        end
        
        sum .= sum .+ layer.charge .* fc(x[j], layer.cutoff) .* exp.(-(x[j] .- layer.W_Fs) .^ 2 .* layer.W_eta)
    end

    # Return the result as a Float32 array
    return sum
end

"""
    Base.show(io::IO, layer::MyLayer)

Custom implementation of the `show` function for displaying the `MyLayer` object.

This function allows you to display the `MyLayer` object in a human-readable format. It prints out the weights `W_eta` and `W_Fs` of the layer, as well as a description indicating that this is a custom layer with two links per input neuron.

### Arguments
- `io::IO`: The output stream (e.g., `stdout`, a file, etc.) where the `MyLayer` information will be printed.
- `layer::MyLayer`: The `MyLayer` object to be displayed.

### Example
```julia
layer = MyLayer(input_dim=5, hidden_dim=3, cutoff=1.5f0, charge=1.0f0)
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

Constructs a Flux model that applies the correct species‐specific subnetwork
to each atom in a fixed ordering, then sums their scalar outputs to produce
one total energy per structure.

# Arguments
- `species_models::Dict{String,Chain}`  
  A dictionary mapping each atomic species (e.g. `"H"`, `"C"`, `"O"`) to its
  corresponding `Flux.Chain` subnetwork (as produced by `create_species_models`).

- `species_order::Vector{String}`  
  A length‑N vector giving, for atom slots `1…N`, which species occupies that
  slot.  Must use exactly the same keys as in `species_models`.

# Returns
- `Chain`  
  A `Flux.Chain` whose first layer is a `Flux.Parallel` of length N branches.
  Branch `i` is exactly `species_models[species_order[i]]`.  Its output is a
  tuple of N per‑atom scalars (shape `(batch_size,)` each), which the final
  anonymous layer sums into one `(batch_size,)` vector of total energies.

# Example

```julia
# 1) Suppose you have
species_list  = ["H","C","O"]
species_models = create_species_models(species_list, G1_number, R_cutoff)

# 2) And your data always has N=40 atom‐slots, with known species at each slot:
species_order = ["H","H","O","C", …]  # length 40

# 3) Build the full model:
model = assemble_atomic_model(species_models, species_order)

# 4) Now `model` expects, as input, a tuple of 40 elements, each of shape
#    (batch_size, features...), e.g. (x[:,1,:], x[:,2,:], …, x[:,40,:]).
#    It returns a 1‑D array of length batch_size.

# 5) You can train with:
loss(x,y) = Flux.Losses.mse(model(ntuple(i-> x[:,i,:], 40)), y)
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
    build_branch(Atom_name::String, G1_number::Float32, R_cutoff::Float32) -> Chain

Constructs a species-specific neural network branch for a given atom type.

This function returns a `Flux.Chain` model tailored to a specific atomic species, 
using a custom `MyLayer` followed by several dense layers. The model structure is:

- `MyLayer`: Applies element-specific transformations based on pairwise distances, 
   a cutoff radius, and the ion charge.
- `Dense` layers: A sequence of fully connected layers with `tanh` activations that process 
   the atomic descriptor vector to output a scalar energy prediction for the atom.

# Arguments
- `Atom_name::String`: The name of the atom (e.g. `"H"`, `"O"`, `"C"`). Used to determine the ion charge via `element_to_charge`.
- `G1_number::Int`: The number of symmetry functions or features computed by `MyLayer`.
- `R_cutoff::Float32`: The cutoff radius beyond which atomic interactions are ignored in `MyLayer`.

# Returns
- `Flux.Chain`: A model that takes atomic environment features and predicts a per-atom energy contribution.

# Example
```julia
model = build_branch("O", 6.0f0, 5.0f0)
output = model(input_vector)  # input_vector should match the expected dimensionality
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
- `model : A Flux.Chain object being the model to train.
- `data::Array`: A 3D array where:
    - The first dimension represents different structures.
    - The second dimension represents atoms within a structure.
    - The third dimension contains atomic charge (index 1) and features (index 2 to end).
- `energies::Vector{Float64}`: A 1D array of reference total energies for each structure.


# Returns
- `Float64`: The mean squared error (MSE) between predicted and reference energies.

# Notes
- This function loops over structures and atoms, predicting energy contributions per atom.
- It automatically selects the correct model based on the atomic charge.
- The final loss is computed as the mean squared error.

# Example
```julia
ions = ["Cs", "Pb", "I"]
models = create_model(ions, R_cutoff=5.0)

data = rand(10, 5, 40)  # Example dataset (10 structures, 5 atoms per structure, 40 features)
energies = rand(10)     # Example reference energies

loss = loss_function(data, energies, models)
println("Loss: ", loss)
"""
function loss_function(
    model,  # the model
    data,       # The features
    energies::Vector{Float32}     # Reference total energies
)   

    total_loss = 0.0
    n_structures = size(data)[1]

    loss=sum(abs2.(map(model,data)-energies))

    for k in 1:n_structures

        temp=model(data[k])
        total_loss+=abs2(temp-energies[k])
    
    end
    return  loss / n_structures # Average Loss
end 



"""
    train_model!(model, x_train, y_train, x_val, y_val, loss_function; 
                 initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
                 epochs=3000, batch_size=32, verbose=true)

Trains a set of neural network models for predicting atomic energies.

# Arguments
- ``model : A Flux.Chain object being the model to train.
- `x_train::T`: Training data, divided in tuples (structures × atoms × features).
- `y_train::Vector{Float32}`: Training total energies.
- `x_val::T`: Validation data, divided in tuples (same shape as `x_train`).
- `y_val::Vector{Float32}`: Validation total energies.
- `loss_function::Function`: Function computing the loss (should be `loss_function(data, energies, models, element_charge)`).
- `species: array withe the species list of the system.
- `initial_lr::Float32=0.01`: Initial learning rate for the optimizer.
- `min_lr::Float32=1e-5`: Minimum learning rate before stopping.
- `decay_factor::Float32=0.5`: Factor by which learning rate decreases when no improvement.
- `patience::Int=25`: Number of epochs without improvement before reducing learning rate.
- `epochs::Int=3000`: Maximum number of training epochs.
- `batch_size::Int=32`: Batch size for training.
- `verbose::Bool=true`: Whether to print progress updates.

# Returns
- `Dict{String, Chain}`: The trained models.
"""
function train_model!(
    model,
    x_train::T, 
    y_train::Vector{Float32}, 
    x_val::T, 
    y_val::Vector{Float32}, 
    loss_function::Function;
    initial_lr=0.01, min_lr=1e-5, decay_factor=0.5, patience=25, 
    epochs=3000, batch_size=32, verbose=true
) :: T where T
    # Initialize optimizer
    opt_state = Flux.setup(Adam(initial_lr), model)


    # Store best models and best loss
    best_epoch = 0
    best_loss = Inf
    best_model = deepcopy(model)

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

        # Save best models
        if loss_val[epoch] < best_loss * 0.95
            best_epoch = epoch
            best_loss = loss_val[epoch]
            best_model = deepcopy(model)
        end


        # Adjust learning rate if no improvement
        if no_improve_count >= patience
            new_lr = max(opt.lr * decay_factor, min_lr)
            opt_state = Flux.setup(Adam(new_lr), model)
            no_improve_count = 0
            if verbose
                println("Reducing learning rate to $(new_lr) at epoch $epoch")
            end
        end
    end



    if verbose
        println("Final Training Loss: $(loss_function(models, x_train, y_train))")
        println("Final Validation Loss: $(loss_function(models, x_val, y_val))")
        println("The best model was found at epoch: $best_epoch")
    end

    return best_model
end
