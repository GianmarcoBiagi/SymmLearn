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
    W_eta = Float32.(rand(Uniform(0.25, 2.5), hidden_dim, input_dim))  # Initialize eta weights (Float32)
    W_Fs = Float32.(rand(Uniform(0.0, 2.5), hidden_dim, input_dim))    # Initialize Fs weights (Float32)

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
    N_neighboring_atoms = size(x)[1]  # Number of neighboring atoms (assuming x represents atom positions or other data)
    sum = zeros(size(layer.W_eta)[1])  # Initialize sum as a zero vector of size hidden_dim

    # Iterate over each neighboring atom
    for j in 1:N_neighboring_atoms
        # Sum the contributions from each neighboring atom
        sum .= sum .+ layer.charge .* fc(x[j], layer.cutoff) .* exp.(-(x[j] .- layer.W_Fs) .^ 2 .* layer.W_eta)
    end

    # Return the result as a Float32 array
    return Float32.(sum)
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
    println(io, "CustomLayer with two links per input neuron (eta and Fs)")
    println(io, "Weights eta: ", layer.W_eta)
    println(io, "Weights Fs: ", layer.W_Fs)
end

"""
    data_preprocess(input_data, output_data; split=[0.7, 0.3], verbose=false)

Preprocesses the data by splitting it into training, validation, and test sets and applying Z-score normalization.

# Arguments
- `input_data`: Input dataset.
- `output_data`: Output dataset (target values).
- `split`: A vector specifying the proportions for data splitting (default `[0.7, 0.3]`).
- `verbose`: If `true`, prints dataset dimensions (default `false`).

# Returns
A tuple containing:
- `(x_train, y_train)`: Training set.
- `(x_val, y_val)`: Validation set.
- `(x_test, y_test)`: Test set.
- `y_mean`: Mean of training data (for denormalization).
- `y_std`: Standard deviation of training data (for denormalization).
"""
function data_preprocess(input_data, output_data; split=[0.7, 0.3]::Vector{Float64}, verbose=false)
    # Ensure the split ratios sum to 1
    if sum(split) != 1
        error("Error: The train-test split ratio is incorrect. The sum must be equal to 1, but got $(sum(split)).")
    end

    # Partitioning the dataset
    ((x_train, tempX), (y_train, tempY)) = partition([input_data, output_data], [0.7, 0.3])
    ((x_val, x_test), (y_val, y_test)) = partition([tempX, tempY], [0.5, 0.5])


    # Ensure y_train, y_val, and y_test are Float32
    y_train = Float32.(y_train)
    y_val = Float32.(y_val)
    y_test = Float32.(y_test)

    # Apply Z-score normalization
    y_mean_1 = mean(y_train)
    y_std_1 = std(y_train)

    # First normalization
    y_train .= (y_train .- y_mean_1) ./ y_std_1

    # Recalculate mean and standard deviation
    y_mean_2 = mean(y_train)
    y_std_2 = std(y_train)

    # Second normalization
    y_train .= (y_train .- y_mean_2) ./ y_std_2

    # Compute final mean and standard deviation (to invert normalization later)
    y_mean = y_mean_2 * y_std_1 + y_mean_1
    y_std = y_std_2 * y_std_1

    # Apply the same normalization to validation and test sets
    y_val .= (y_val .- y_mean) ./ y_std
    y_test .= (y_test .- y_mean) ./ y_std

    # Print dataset dimensions if verbose mode is enabled
    if verbose
        println("x_train dimensions: ", size(x_train))
        println("x_val dimensions: ", size(x_val))
        println("x_test dimensions: ", size(x_test))
    end

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), y_mean, y_std
end

"""
    create_model(ions::Vector{String}, R_cutoff::Float64, G1_number::Int = 5, verbose::Bool = false) -> Dict{String, Chain}

Creates a set of neural network models, one for each ion in the input list, using the specified cutoff radius and 
number of input features. The models are stored in a dictionary where the keys follow the format `"ion_model"`.

### Arguments
- `ions::Vector{String}`: List of ion symbols for which models will be created.
- `R_cutoff::Float64`: Cutoff radius used in the `MyLayer` layer.
- `G1_number::Int = 5`: Number of input features in the first layer (default = 5).
- `verbose::Bool = false`: If `true`, prints the model structures after creation.

### Returns
- `Dict{String, Chain}`: A dictionary where each key is `"ion_model"` and the corresponding value is a neural network (`Chain`).

### Example
```julia
models = create_model(["Li", "Na", "K"], 6.5, 8, verbose=true)
"""

function create_model(
    ions::Vector{String}, 
    R_cutoff::Float64, 
    G1_number::Int = 5, 
    verbose::Bool = false
)
    # Get ion charges using element_charge dictionary
    ion_charges = 0.1 * getindex.(Ref(element_charge), ions)

    # Number of ions
    n_of_ions = length(ions)

    # Dictionary to store models
    models = Dict{String, Any}()

    # Create a neural network model for each ion
    for i in 1:n_of_ions
        models["$(ions[i])_model"] = Chain(
            MyLayer(1, G1_number, R_cutoff, ion_charges[i]),
            Dense(G1_number, 15, tanh),
            Dense(15, 10, tanh),
            Dense(10, 5, tanh),
            Dense(5, 1)
        )
    end

    # Print model details if verbose is enabled
    if verbose
        for (name, model) in models
            println("Model: ", name)
            println(model)
            println("────────────────────────────────")
        end
    end

    return models  # Return the dictionary of models
end

"""
    loss_function(data, energies, models)

Computes the mean squared error (MSE) loss between predicted and reference total energies.

# Arguments
- `data::Array`: A 3D array where:
    - The first dimension represents different structures.
    - The second dimension represents atoms within a structure.
    - The third dimension contains atomic charge (index 1) and features (index 2 to end).
- `energies::Vector{Float64}`: A 1D array of reference total energies for each structure.
- `models::Dict{String, Any}`: A dictionary mapping ion names (e.g., `"Cs"`, `"Pb"`, `"I"`) to their neural network models.

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
    data::Array{Float32,3}, 
    energies::Vector{Float32}, 
    models::Dict{String, Chain}, 
    element_charge::Dict{String, Float64} ) 


    total_loss = 0.0,
    num_structures = size(data, 1) # Number of structures
    for k in 1:num_structures  # Loop over structures
        predicted_energy = 0.0f0
        num_atoms = size(data, 2)  # Number of atoms per structure

        for i in 1:num_atoms  # Loop over atoms
            charge = data[k, i, 1]  # Atomic charge
          features = data[k, i, 2:end]  # Feature vector

         # Find corresponding ion name based on charge
         ion_name = findfirst(x -> isapprox(element_charge[x] * 0.1, charge; atol=1e-3), keys(element_charge))

         # If the ion is found and exists in the model dictionary, use it
         if ion_name !== nothing && haskey(models, "$(ion_name)_model")
             predicted_energy += models["$(ion_name)_model"](features)[1]
         end
        end

        # Accumulate squared error
        total_loss += abs2(predicted_energy - energies[k])
    end

    return total_loss / num_structures  # Mean squared error (MSE)
end

"""
    train_model!(models, x_train, y_train, x_val, y_val, loss_function; 
                 initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
                 epochs=3000, batch_size=32, verbose=true)

Trains a set of neural network models for predicting atomic energies.

# Arguments
- `models::Dict{String, Chain}`: Dictionary mapping ion names to their neural network models.
- `x_train::Array{Float32,3}`: Training data (structures × atoms × features).
- `y_train::Vector{Float32}`: Training total energies.
- `x_val::Array{Float32,3}`: Validation data (same shape as `x_train`).
- `y_val::Vector{Float32}`: Validation total energies.
- `loss_function::Function`: Function computing the loss (should be `loss_function(data, energies, models, element_charge)`).
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
    models::Dict{String, Chain},
    x_train::Array{Float32,3}, 
    y_train::Vector{Float32}, 
    x_val::Array{Float32,3}, 
    y_val::Vector{Float32}, 
    loss_function::Function;
    initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
    epochs=3000, batch_size=32, verbose=true
)
    # Initialize optimizer
    opt = Flux.Adam(initial_lr)

    # Store best models and best loss
    best_epoch = 0
    best_loss = Inf
    best_models = deepcopy(models)

    # Loss tracking
    loss_train = zeros(Float32, epochs)
    loss_val = zeros(Float32, epochs)
    no_improve_count = 0

    @showprogress for epoch in 1:epochs
        for i in 1:batch_size:size(x_train, 1)
            end_index = min(i + batch_size - 1, size(x_train, 1))
            x_batch = x_train[i:end_index, :, :]
            y_batch = y_train[i:end_index]
            
            # Training step
            Flux.train!(loss_function, Flux.params(values(models)...), [(x_batch, y_batch)], opt)
        end

        # Compute losses
        loss_train[epoch] = loss_function(x_train, y_train, models)
        loss_val[epoch] = loss_function(x_val, y_val, models)

        # Save best model
        if loss_val[epoch] < best_loss * 0.95
            best_epoch = epoch
            best_loss = loss_val[epoch]
            best_models = deepcopy(models)
        end


        # Adjust learning rate if no improvement
        if no_improve_count >= patience
            new_lr = max(opt.lr * decay_factor, min_lr)
            opt = Flux.Adam(new_lr)
            no_improve_count = 0
            if verbose
                println("Reducing learning rate to $(new_lr) at epoch $epoch")
            end
        end
    end

    # Assign best models
    models .= best_models


    if verbose
        println("Final Training Loss: $(loss_function(x_train, y_train, models))")
        println("Final Validation Loss: $(loss_function(x_val, y_val, models))")
        println("The best model was found at epoch: $best_epoch")
    end

    return models
end




