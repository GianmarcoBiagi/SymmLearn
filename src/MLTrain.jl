using Flux
using Random
using ProgressMeter
using Statistics

"""
    struct ModelContainer

A container for storing a neural network model along with its associated name.  
This struct is **immutable**, meaning that once created, the `name` and `model` fields cannot be reassigned.  
However, the model itself remains mutable, allowing modifications to its parameters.

# Fields
- `name::String` : The name of the model (e.g., associated with a specific ion or element).
- `model::Flux.Chain` : The neural network model, which can be trained and updated.

# Example
```julia
using Flux

# Create a simple neural network model
m = Chain(Dense(2, 5, relu), Dense(5, 1))

# Store the model in an immutable container
container = ModelContainer("MyModel", m)

# Modify the model parameters (allowed)
Flux.train!(m, rand(2, 10), rand(1, 10), ADAM())

# Attempting to reassign a new model (not allowed)
container.model = Chain(Dense(2, 5), Dense(5, 1))  # ERROR: "setfield! immutable struct"
"""

struct ModelContainer
    name::String
    model::Flux.Chain  # The model remains mutable
end


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
    N_neighboring_atoms = size(x)[1]  # Number of neighboring atoms (assuming x represents atom positions or other data)
    sum = zeros(Float32, size(layer.W_eta)[1])  # Initialize sum as a zero vector of size hidden_dim

    # Iterate over each neighboring atom
    for j in 1:N_neighboring_atoms
        # Sum the contributions from each neighboring atom
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
    println(io, "CustomLayer with two links per input neuron (eta and Fs)")
    println(io, "Weights eta: ", layer.W_eta)
    println(io, "Weights Fs: ", layer.W_Fs)
end

"""
    partition(data,parts;shuffle,dims,rng)

Partition (by rows) one or more matrices according to the shares in `parts`.

# Parameters
* `data`: A matrix/vector or a vector of matrices/vectors
* `parts`: A vector of the required shares (must sum to 1)
* `shufle`: Whether to randomly shuffle the matrices (preserving the relative order between matrices)
* `dims`: The dimension for which to partition [def: `1`]
* `copy`: Wheter to _copy_ the actual data or only create a reference [def: `true`]
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Notes:
* The sum of parts must be equal to 1
* The number of elements in the specified dimension must be the same for all the arrays in `data`

# Example:
```julia
julia> x = [1:10 11:20]
julia> y = collect(31:40)
julia> ((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3])
 ```
 """
function partition(data::AbstractArray{T,1},parts::AbstractArray{Float64,1};shuffle=true,dims=1,copy=true,rng = Random.GLOBAL_RNG) where T <: AbstractArray
        # the sets of vector/matrices
        N = size(data[1],dims)
        all(size.(data,dims) .== N) || @error "All matrices passed to `partition` must have the same number of elements for the required dimension"
        ridx = shuffle ? Random.shuffle(rng,1:N) : collect(1:N)
        return partition.(data,Ref(parts);shuffle=shuffle,dims=dims,fixed_ridx = ridx,copy=copy,rng=rng)
end

function partition(data::AbstractArray{T,Ndims}, parts::AbstractArray{Float64,1};shuffle=true,dims=1,fixed_ridx=Int64[],copy=true,rng = Random.GLOBAL_RNG) where {T,Ndims}
    # the individual vector/matrix
    N        = size(data,dims)
    nParts   = size(parts)
    toReturn = toReturn = Array{AbstractArray{T,Ndims},1}(undef,nParts)
    if !(sum(parts) ≈ 1)
        @error "The sum of `parts` in `partition` should total to 1."
    end
    ridx = fixed_ridx
    if (isempty(ridx))
       ridx = shuffle ? Random.shuffle(rng, 1:N) : collect(1:N)
    end
    allDimIdx = convert(Vector{Union{UnitRange{Int64},Vector{Int64}}},[1:i for i in size(data)])
    current = 1
    cumPart = 0.0
    for (i,p) in enumerate(parts)
        cumPart += parts[i]
        final = i == nParts ? N : Int64(round(cumPart*N))
        allDimIdx[dims] = ridx[current:final]
        toReturn[i]     = copy ? data[allDimIdx...] : @views data[allDimIdx...]
        current         = (final +=1)
    end
    return toReturn
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
function data_preprocess(input_data, target; split=[0.7, 0.3]::Vector{Float64}, verbose=false)


    # Partitioning the dataset
    ((x_train, tempX), (y_train, tempY)) = partition([input_data, target], split)
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
    create_model(ions::Vector{String}, R_cutoff::Float32, G1_number::Int = 5, verbose::Bool = false) -> Dict{String, Chain}

Creates a set of neural network models, one for each ion in the input list, using the specified cutoff radius and 
number of input features. The models are stored in a dictionary where the keys follow the format `"ion_model"`.

### Arguments
- `ions::Vector{String}`: List of ion symbols for which models will be created.
- `R_cutoff::Float32`: Cutoff radius used in the `MyLayer` layer.
- `G1_number::Int = 5`: Number of input features in the first layer (default = 5).
- `verbose::Bool = false`: If `true`, prints the model structures after creation.

### Returns
- `Tuple(ModelContainer)`: A tuple where each model and the corresponding name are stored.

### Example
```julia
models = create_model(["Li", "Na", "K"], 6.5, 8, verbose=true)
"""

function create_model(
    ions::Vector{String}, 
    R_cutoff::Float32, 
    G1_number::Int = 5, 
    verbose::Bool = false
)
    # Get ion charges using element_charge dictionary
    ion_charges = 0.1f0 * getindex.(Ref(element_to_charge), ions)

    # Number of ions
    n_of_ions = length(ions)

    # Create an array to store ModelContainer instances
    models = Vector{ModelContainer}(undef, n_of_ions)

    # Create a neural network model for each ion and assign to the array
    for i in 1:n_of_ions
        ion_name = ions[i]
        
        # Create the model
        model = Chain(
            MyLayer(1, G1_number, R_cutoff, ion_charges[i]),
            Dense(G1_number, 15, tanh),
            Dense(15, 10, tanh),
            Dense(10, 5, tanh),
            Dense(5, 1)
        )
        
        # Store the model inside ModelContainer
        models[i] = ModelContainer(ion_name, model)
    end

    # Print model details if verbose is enabled
    if verbose
        for container in models
            println("A model has been created called: ", container.name)
            println(container.model)
            println("────────────────────────────────")
        end
    end

    # Convert models array into an immutable Tuple
    models_final = Tuple(models)

    return models_final  # Return the immutable Tuple of models
end



"""
    loss_function(models, data, energies)

Computes the mean squared error (MSE) loss between predicted and reference total energies.

# Arguments
- `models::Tuple{ModelContainer}`: A Tuple containing ModelContainer objects.
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
    models::Tuple,  # Struct mapping element names to ML models
    data::Array{Float32,3},       # Data: (structures, atoms, features)
    energies::Vector{Float32}     # Reference total energies
) 

    total_loss = 0.0
    n_structures = size(data)[1]
    
    n_of_ions=size(models,1)
    

    for k in 1:n_structures # ciclo sui dataset
        temp = 0.0
        n_atoms = size(data)[2] # numero di atomi
    
        for i in 1:n_atoms # ciclo sugli atomi del dataset
            charge = data[k, i, 1]
            features = data[k, i, 2:40]

            ion_name=charge_to_element[charge]

            for i in 1:n_of_ions
        
               if "$(ion_name)_model" == models[i].name
                temp += models[i][2](features)[1]
               end
            end
            
        end
    
    total_loss += abs2(temp - energies[k])
    end
    return total_loss / n_structures # Media della perdita
    end 






"""
    train_model!(models, x_train, y_train, x_val, y_val, loss_function; 
                 initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
                 epochs=3000, batch_size=32, verbose=true)

Trains a set of neural network models for predicting atomic energies.

# Arguments
- `models::Tuple`: A Tuple containing ModelContainer objects.
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
    models::Tuple{ModelContainer},
    x_train::Array{Float32,3}, 
    y_train::Vector{Float32}, 
    x_val::Array{Float32,3}, 
    y_val::Vector{Float32}, 
    loss_function::Function;
    initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
    epochs=3000, batch_size=32, verbose=true
)
    # Initialize optimizer
    opt_state = Flux.setup(Adam(initial_lr), models)

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
            Flux.train!(loss_function, models, [(x_batch, y_batch)], opt_state)
        end

        # Compute losses
        loss_train[epoch] = loss_function(models, x_train, y_train)
        loss_val[epoch] = loss_function(models, x_val, y_val)

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
    #models .= best_models


    if verbose
        println("Final Training Loss: $(loss_function(models, x_train, y_train))")
        println("Final Validation Loss: $(loss_function(models, x_val, y_val))")
        println("The best model was found at epoch: $best_epoch")
    end

    return models
end




