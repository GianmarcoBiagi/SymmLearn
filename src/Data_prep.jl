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
4. `all_cells::Array{Float32, 3}`: A 3D array containing the cell matrices (dimensions) for each configuration.
5. `dataset::Array{Float32, 3}`: A 3D array with the following data for each configuration:
   - Rows 1-3: Atomic positions (x, y, z)
   - Rows 4-6: Forces acting on atoms (fx, fy, fz)
6. `all_energies::Vector{Float32}`: A vector containing the total energy for each configuration.

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
atoms_in_a_cell, species, all_cells, dataset, all_energies = extract_data("path/to/data.xyz")
"""

function extract_data(path::String)

    # Read the frame data from the provided file path
    frame = read_frames(path)

    # Get the number of configurations (i.e., the number of frames in the data)
    n_of_configs = size(frame)[1]
        
    # Get the number of atoms in a cell (from the first frame)
    atoms_in_a_cell = frame[1]["N_atoms"]

    # Pre-allocate the arrays to store the extracted data
    all_energies = zeros(Float32, n_of_configs)  # Array to store energies of each configuration
    all_cells = zeros(Float32, 3, 3, n_of_configs)  # Array to store cell matrix for each configuration
    dataset = zeros(Float32, 6, atoms_in_a_cell, n_of_configs)  # 3D dataset to store atomic information for each configuration


    # Extract the unique species (elements) used in the system by removing duplicates from the species list
    species = frame[1]["arrays"]["species"]
    unique_species = Set(species)  # Convert species to a set to remove duplicates
    unique_species = collect(unique_species)  # Convert the set back into an array (optional, if you need it as an array)

    # Extract cell matrices for each configuration
    all_cells = [frame[i]["cell"] for i in 1:n_of_configs]

    # Extract energy values for each configuration
    all_energies = [frame[i]["info"]["energy"] for i in 1:n_of_configs]
    
    
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
    create_nn_input(dataset, all_lattice, num_atoms::Int)

This function processes a dataset of atomic information and lattice vectors to create an input array for a neural network. 
The array contains the charges, positions, and pairwise distances between atoms in a periodic box. 
Each dataset corresponds to a structure, and each structure consists of a number of atoms specified by `num_atoms`.

### Arguments
- `dataset::Array{Array{T}}`: A list of datasets, where each dataset corresponds to a structure.
    - Each dataset is a list of atoms, with each atom represented by a vector containing its charge and positions (x, y, z).
- `all_lattice::Array{Array{T}}`: A list of lattice vectors corresponding to each structure in `dataset`. Each lattice vector is used for periodic boundary conditions when calculating distances.
- `num_atoms::Int`: The number of atoms in each structure (the number of atoms per dataset). This value is used to define the shape of the output array.

### Returns
- `Array{Float32, 3}`: A 3D array of shape `(num_datasets, num_atoms, num_atoms)` where:
    - `num_datasets` corresponds to the number of structures in the dataset.
    - `num_atoms` is the number of atoms in each structure.
    - The elements in the array contain the neural network input, which includes atom charges and distances between pairs of atoms.

### Example

```julia
# Assume `dataset` contains atomic data and `all_lattice` contains corresponding lattice matrices
nn_input = create_nn_input(dataset, all_lattice, num_atoms=40)
println(nn_input)

"""

# Function to create the neural network input array
function create_nn_input(dataset, all_lattice, num_atoms::Int32)
    num_datasets = size(dataset)[3]

    # Input array shape: (num_datasets, num_atoms, num_atoms) of type Float32
    nn_input = Array{Float32, 3}(undef, num_datasets, num_atoms, num_atoms-1)

    for i in 1:num_datasets
        current_dataset = dataset[:,:,i]
        lattice_vectors = all_lattice[i,:]

        for j in 1:num_atoms
            atom_j = current_dataset[:,j]
            pos_j = atom_j[1:3]



            # Calculate the distance to other atoms
            slot_index = 1  # Start from the second slot
            for k in 1:num_atoms
                if j == k
                    continue  # Skip distance to itself
                end
                atom_k = current_dataset[:,k]
                pos_k = atom_k[1:3]

                # Calculate the distance with PBC
                distance = distance_with_pbc(pos_j, pos_k, lattice_vectors[1])

                nn_input[i, j, slot_index] = distance
                slot_index += 1  # Move to the next slot
            end
        end
    end

    return nn_input
end


"""
    data_preprocess(input_data, output_data; split=[0.7, 0.3], verbose=false)

Preprocesses the data by splitting it into training, validation, and test sets, applying Z-score normalization, 
and moving data to the GPU if available.

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
function data_preprocess(input_data, target; split=[0.7, 0.15, 0.15]::Vector{Float64}, verbose=false)
    # Convert target to Float32 early
    target = Float32.(target)

    # Partitioning the dataset in a single step
    ((x_train, x_val, x_test), (y_train, y_val, y_test)) = partition([input_data, target], split)

    ###### DOUBLE Z-SCORE NORMALIZATION FOR TARGET ######

    # First Z-score normalization
    y_mean_1 = mean(y_train)
    y_std_1 = std(y_train, corrected=false)
    y_train .= (y_train .- y_mean_1) ./ y_std_1

    # Recalculate mean and std after first normalization
    y_mean_2 = mean(y_train)
    y_std_2 = std(y_train, corrected=false)

    # Second normalization on already normalized values
    y_train .= (y_train .- y_mean_2) ./ y_std_2

    # Reconstruct global mean and std (for denormalization or validation/test normalization)
    y_mean = y_mean_2 * y_std_1 + y_mean_1
    y_std  = y_std_2 * y_std_1

    # Apply final normalization to val and test using global mean and std
    y_val .= (y_val .- y_mean) ./ y_std
    y_test .= (y_test .- y_mean) ./ y_std

    #####################################################

    # GPU check
    if CUDA.functional()
        device_name = CUDA.name(CUDA.device())  # Get GPU name
        x_train, y_train = cu(x_train), cu(y_train)
        x_val, y_val = cu(x_val), cu(y_val)
        x_test, y_test = cu(x_test), cu(y_test)

        println("Data successfully mounted on GPU: ", device_name)
    else
        println("No GPU available. Data remains on CPU.")
    end

    # Print dataset dimensions if verbose mode is enabled
    if verbose
        println("x_train dimensions: ", size(x_train))
        println("x_val dimensions: ", size(x_val))
        println("x_test dimensions: ", size(x_test))
    end

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), y_mean, y_std
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
- `Train`: Training dataset.
- `Val`: Validation dataset.
- `Test_data`: Test dataset.
- `y_mean`: Mean energy value for normalization.
- `y_std`: Standard deviation of energy values for normalization.
- `species`: List of atomic species present in the dataset, can be used as an input for the create_model function.

# Description
This function extracts atomic structure and energy information from the input file.
It then generates an input dataset suitable for neural network training. The data
is preprocessed, normalized, and split into training, validation, and test sets.

# Dependencies
- `extract_data(file_path)`: Extracts atomic structures, species, cell information, and energies from the input file.
- `create_nn_input(dataset, all_cells, N_atoms)`: Generates the neural network input dataset.
- `data_preprocess(nn_input_dataset, all_energies)`: Normalizes and splits the dataset into train, validation, and test sets.
"""
function xyz_to_nn_input(file_path::String)

    # Extract atomic and structural information from the input XYZ file
    N_atoms, species, all_cells, dataset, all_energies = extract_data(file_path)

    # Create the neural network input dataset
    create_nn_input(dataset, all_cells, N_atoms,species)

    # Preprocess data: normalize, split into train, validation, and test sets
    Train, Val, Test_data, y_mean, y_std = data_preprocess(nn_input_dataset, all_energies)
    
    # Return the processed datasets and normalization parameters
    return (Train, Val, Test_data, y_mean, y_std, species)
end
