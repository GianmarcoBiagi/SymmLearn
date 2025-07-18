using ExtXYZ

"""
    extract_data(path::String)

### Description:
This function extracts key information from a dataset of atomic configurations stored in a file ina xyz format . It reads the frames (or configurations) using ExtXYZ, and for each configuration, it extracts:
- The number of atoms
- The unique species of atoms
- The positions and forces acting on each atom
- The cell matrix (i.e., the box or lattice dimensions)
- The total energy of the configuration

The function returns this data in a structured format for further analysis.

### Arguments:
- `path::String`: The path to the file containing the frame data. The file should contain the necessary details about multiple configurations, including atomic positions, species, forces, and energy.

### Returns:
A tuple containing the following elements:
1. `atoms_in_a_cell::Int`: The number of atoms in a single unit cell (assumed to be the same in all the frames).
2. `species::Vector{String}`: A vector containing the species (elements) present in the system.
3. `uniqe_species::Vector{String}`: A vector containing the unique species (elements) present in the system.
4. `all_cells::Array{Float64, 3}`: A 3D array containing the cell matrices (dimensions) for each configuration.
5. `dataset::Array{Float32, 3}`: A 3D array with the following data for each configuration:
   - Rows 1-3: Atomic positions (x, y, z)
   - Rows 4-6: Forces acting on atoms (fx, fy, fz)
6. `all_energies::Vector{Float64}`: A vector containing the total energy for each configuration.

### Functionality:
1. **Reading the data**: The function first reads the data from the provided path using `read_frames(path)`.
2. **Extracting number of configurations**: It calculates the total number of configurations by checking the size of the data.
3. **Pre-allocating arrays**: The function allocates memory for arrays to store the energies, cell matrices, atomic data, and forces for each configuration.
4. **Species extraction**: It extracts the unique species (elements) used in the configurations and stores them in the `species` array.
5. **Data extraction**:
   - The function extracts the cell matrices for each configuration.
   - It also extracts the total energy for each configuration.
   - The dataset is populated with atomic charges, positions, and forces for each atom in each configuration.
6. **Returning the data**: After all data has been extracted and stored in the arrays, the function returns the extracted data as a tuple.

### Example Usage:
```julia
atoms_in_a_cell, species,unique_species, all_cells, dataset, all_energies = extract_data("path/to/data.xyz")
"""

function extract_data(path::String)

    # Read the frame data from the provided file path
    frame = read_frames(path)

    # Get the number of configurations (i.e., the number of frames in the data)
    n_of_configs = size(frame)[1]
        
    # Get the number of atoms in a cell (from the first frame)
    atoms_in_a_cell = frame[1]["N_atoms"]

    # Pre-allocate the arrays to store the extracted data
    dataset = zeros(Float32, 6, atoms_in_a_cell, n_of_configs)  # 3D dataset to store atomic information for each configuration


    # Extract the unique species (elements) used in the system by removing duplicates from the species list
    species = frame[1]["arrays"]["species"]
    unique_species = Set(species)  # Convert species to a set to remove duplicates
    unique_species = collect(unique_species)  # Convert the set back into an array (optional, if you need it as an array)

    # Extract cell matrices for each configuration
    all_cells = [Float32.(frame[i]["cell"]) for i in 1:n_of_configs]

    # Extract energy values for each configuration
    all_energies = [Float32.(frame[i]["info"]["energy"]) for i in 1:n_of_configs]

    # Extract atom-specific data (charge, position, and forces) for each configuration
    for i in 1:n_of_configs
        for j in 1:atoms_in_a_cell
  

            # Store atomic positions (first 3 elements: x, y, z)
            dataset[1:3, j, i] = frame[i]["arrays"]["pos"][1:3, j]

            # Store forces (first 3 components: fx, fy, fz)
            dataset[4:6, j, i] = frame[i]["arrays"]["forces"][1:3, j]
        end
    end

    # Return the extracted data: number of atoms, species, cell matrices, dataset, and energies
    return atoms_in_a_cell, species, unique_species, all_cells, dataset, all_energies
end



"""
    create_nn_input(dataset, num_atoms::Int32)

This function processes a dataset of atomic information and extracts the atomic coordinates 
for each atom in each structure. This is used as input for a neural network model that requires 
Cartesian coordinates.

### Arguments
- `dataset::Array{Float32, 3}`: A 3D array where each slice along the third dimension is a structure, 
  each structure is a matrix of shape (3, num_atoms).
- `num_atoms::Int`: Number of atoms per structure.

### Returns
- `Array{Float32, 3}`: A 3D array of shape `(num_datasets, num_atoms, 3)` where:
    - `num_datasets` is the number of structures in the dataset.
    - Each `num_atoms x 3` slice contains the Cartesian coordinates of all atoms in a structure.

### Example
```julia
coord_input = create_nn_input(dataset, num_atoms=40)
println(coord_input[1, :, :])  # Prints coordinates of all atoms in the first structure


"""

function create_nn_input(dataset::Array{Float32, 3}, num_atoms::Int32)
    num_datasets = size(dataset, 3)

    # Output array: (num_datasets, num_atoms, 3)
    nn_input = Array{Float32, 3}(undef, num_datasets, num_atoms, 3)

    for i in 1:num_datasets
        current_structure = dataset[:, :, i]
        for j in 1:num_atoms
            pos = current_structure[1:3, j]  # Coordinates assumed to be in columns 2, 3, 4
            nn_input[i, j, :] = pos
        end
    end

    return nn_input
end



"""
    create_nn_target(dataset, all_energies)

Creates a structured target array for neural network training.

# Arguments
- `dataset::Array{Float32,3}`: A 3D array of shape (features, atoms, samples). Forces are assumed to be at indices 4:6 along the first axis.
- `all_energies::Vector{Float32}`: A 1D array of total energies for each sample, of length equal to size(dataset, 3).

# Returns
- `targets::Vector{Dict}`: A vector of dictionaries, one for each sample.
  Each dictionary contains:
    - `:energy`: the total energy of the system
    - `:forces`: a flat Vector{Float32} with all atomic forces concatenated (3 values per atom)

"""
function create_nn_target(dataset::Array{Float32,3}, all_energies::Vector{Float32})
    num_samples = size(dataset, 3)   # Number of systems
    num_atoms = size(dataset, 2)     # Number of atoms per system

    # Extract forces: shape will be (samples, atoms, 3)
    all_forces = permutedims(dataset[4:6, :, :], (3, 2, 1))

    targets = Vector{Dict}(undef, num_samples)

    for i in 1:num_samples
        # Flatten the (atoms, 3) matrix of forces into a single vector
        forces_vec = vec(all_forces[i, :, :])

        # Store energy and forces in a dictionary
        targets[i] = Dict(
            :energy => all_energies[i],
            :forces => forces_vec
        )
    end

    return targets
end



"""
    data_preprocess(input_data, target; split=[0.7, 0.15, 0.15])

Preprocesses input and target data for training a neural network model. It splits the dataset, 
performs separate normalization for energies and forces, and structures the output.

# Arguments
- `input_data`: Input features (e.g. atomic environments), shaped as `(n_structures, n_atoms, ...)`.
- `target`: A vector of dictionaries where each entry has:
    - `:energy`: total system energy (Float32).
    - `:forces`: vector of forces for each atom (Float32 vector of length 3 * n_atoms).
- `split`: A vector of Float64 values indicating train/validation/test split proportions (default `[0.7, 0.15, 0.15]`).

# Returns
A tuple containing:
- `(x_train, y_train)`: Training input and target.
- `(x_val, y_val)`: Validation input and target.
- `(x_test, y_test)`: Test input and target.
- `energy_mean, energy_std`: Mean and standard deviation used for energy normalization.
- `forces_mean, forces_std`: Mean and standard deviation used for force normalization.

# Notes
- Energies undergo a double Z-score normalization.
- Forces are normalized using standard Z-score.

"""
function data_preprocess(input_data, target; split=[0.6, 0.2, 0.2]::Vector{Float64})

    ##### --- Extract energy and forces --- #####
    energies = Float32[entry[:energy] for entry in target]
    forces   = [entry[:forces] for entry in target]  # Each is a vector of 3 * n_atoms

    # Convert list of force vectors into a matrix (each row = one structure)
    force_matrix = reduce(hcat, forces)'  # Shape: (n_structures, 3 * n_atoms)
    force_matrix = Float32.(force_matrix)

    ##### --- Split dataset --- #####
    ((x_train, x_val, x_test), 
     (e_train, e_val, e_test), 
     (f_train, f_val, f_test)) = partition([input_data, energies, force_matrix], split)

    ##### --- ENERGY: Double Z-score normalization --- #####
    e_mean1 = mean(e_train)
    e_std1  = std(e_train, corrected=false)
    e_train .= (e_train .- e_mean1) ./ e_std1

    e_mean2 = mean(e_train)
    e_std2  = std(e_train, corrected=false)
    e_train .= (e_train .- e_mean2) ./ e_std2

    # Final energy normalization constants (used for denormalization)
    energy_mean = e_mean2 * e_std1 + e_mean1
    energy_std  = e_std2 * e_std1

    # Apply same global normalization to val and test
    e_val  .= (e_val .- energy_mean) ./ energy_std
    e_test .= (e_test .- energy_mean) ./ energy_std

    ##### --- FORCES: Standard Z-score normalization --- #####
    all_train_forces = reduce(vcat, eachrow(f_train))
    forces_mean = mean(all_train_forces)
    forces_std  = std(all_train_forces, corrected=false)

    f_train .= (f_train .- forces_mean) ./ forces_std
    f_val   .= (f_val   .- forces_mean) ./ forces_std
    f_test  .= (f_test  .- forces_mean) ./ forces_std

    ##### --- Repack normalized targets --- #####
    n_atoms = size(input_data)[2]
  

    y_train = [Dict(:energy => e_train[i], :forces => reshape(f_train[i, :], (n_atoms, 3))) for i in 1:length(e_train)]
    y_val   = [Dict(:energy => e_val[i],   :forces => reshape(f_val[i, :],   (n_atoms, 3))) for i in 1:length(e_val)]
    y_test  = [Dict(:energy => e_test[i],  :forces => reshape(f_test[i, :],  (n_atoms, 3))) for i in 1:length(e_test)]


    ##### --- Return --- #####
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), energy_mean, energy_std, forces_mean, forces_std
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
    xyz_to_nn_input(file_path::String)

Processes an XYZ file to generate input data for a neural network.

# Arguments
- `file_path::String`: The path to the XYZ file containing atomic structures and energies.

# Returns
- `Train`: Training dataset (input-output pairs).
- `Val`: Validation dataset (input-output pairs).
- `Test_data`: Test dataset (input-output pairs).
- `y_mean`: Mean energy value for normalization.
- `y_std`: Standard deviation of energy values for normalization.
- `species`: List of atomic species present in the dataset, can be used as an input for the `create_model` function.
- 'all_cells': List of all the lattice cells of the dataset

# Description
This function extracts atomic structure and energy information from the input XYZ file.
It then generates an input dataset suitable for neural network training. The data is preprocessed, 
normalized, and split into training, validation, and test sets.

# Dependencies
- `extract_data(file_path)`: Extracts atomic structures, species, cell information, and energies from the input file.
- `create_nn_input(dataset, all_cells, N_atoms)`: Generates the neural network input dataset from atomic data.
- `data_preprocess(nn_input_dataset, all_energies)`: Normalizes and splits the dataset into train, validation, and test sets.

# Example

```julia
# Assume `file_path` is the path to your XYZ file
Train, Val, Test_data, y_mean, y_std, species = xyz_to_nn_input("path_to_file.xyz")
println(Train)
println(Val)
"""

function xyz_to_nn_input(file_path::String)

    # Extract atomic and structural information from the input XYZ file
    N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)

    # Create the neural network input dataset
    nn_input_dataset = create_nn_input(dataset, N_atoms)

    #Create the neural network target 

    target = create_nn_target(dataset, all_energies)
  

    # Preprocess data: normalize, split into train, validation, and test sets
    Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std = data_preprocess(nn_input_dataset, target)
    
    # Return the processed datasets and normalization parameters
    return (Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std, species, all_cells)
end
