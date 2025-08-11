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
    prepare_nn_data(dataset::Array{Float32, 3}, num_atoms::Int32) -> (Array{Float32, 2}, Array{Float32, 2})

Converts a 3D dataset of atomic positions and forces into two 2D arrays formatted for neural network input.

# Arguments
- `dataset::Array{Float32, 3}`:  
  A 3-dimensional array with dimensions `(6, num_atoms, num_samples)`.  
  The first 3 rows (`1:3`) correspond to atomic positions (x, y, z).  
  The next 3 rows (`4:6`) correspond to atomic forces (fx, fy, fz).

- `num_atoms::Int32`:  
  Number of atoms in each system/sample.

# Returns
- `nn_input::Array{Float32, 2}`:  
  A 2D array of shape `(num_samples, num_atoms * 3)` where each row contains concatenated atomic positions for one sample.

- `forces::Array{Float32, 2}`:  
  A 2D array of shape `(num_samples, num_atoms * 3)` where each row contains concatenated atomic forces for one sample.

# Description
This function processes the input dataset by extracting and flattening the atomic positions and forces from each sample into two separate 2D arrays.  
The output arrays are suitable for direct input into neural network models that expect flattened feature vectors per sample.
"""



function prepare_nn_data(dataset::Array{Float32, 3}, num_atoms::Int32)
    num_samples = size(dataset, 3)

    # Prepare nn_input array: (num_samples, num_atoms * 3)
    nn_input = Array{Float32, 2}(undef, num_samples, num_atoms * 3)
    # Prepare forces array: (num_samples, num_atoms * 3)
    forces = zeros(Float32, num_samples, num_atoms * 3)

    for i in 1:num_samples
        current_structure = dataset[:, :, i]

        for j in 1:num_atoms
            # Positions in rows 1:3
            nn_input[i, 1+(j-1)*3 : j*3] = current_structure[1:3, j]
            # Forces in rows 4:6
            forces[i, 1+(j-1)*3 : j*3] = current_structure[4:6, j]
        end
    end

    return nn_input, forces
end




"""
    data_preprocess(input_data, target; split=[0.6, 0.2, 0.2])

Preprocesses input features and target data for neural network training. Assumes each structure's
input is flattened as a row vector of length `3 * n_atoms`. The dataset is split, normalized, 
and target data repackaged into `Sample` structs containing normalized energy and force vectors.

# Arguments
- `input_data::Array{<:Real, 2}`: Input features with shape `(N_structures, 3 * n_atoms)`.
- `target::Vector{Dict}`: Vector of dictionaries, each with keys:
    - `:energy` (`Float32`): Total system energy.
    - `:forces` (`Vector{Float32}`): Flattened forces vector of length `3 * n_atoms`.
- `split::Vector{Float64}` (optional): Proportions for train/validation/test splits, summing to 1 (default `[0.6, 0.2, 0.2]`).

# Returns
A tuple containing:
- `(x_train, y_train)`: Training inputs and targets (`Vector{Sample}`).
- `(x_val, y_val)`: Validation inputs and targets.
- `(x_test, y_test)`: Test inputs and targets.
- `energy_mean, energy_std`: Mean and std used for energy normalization.
- `forces_mean, forces_std`: Mean and std used for force normalization.

# Notes
- Inputs must be flattened (shape `(N, 3 * n_atoms)`).
- Energies are normalized by single Z-score normalization.
- Forces are normalized feature-wise by Z-score normalization.
- Targets are returned as `Sample` structs with:
    - `.energy`: normalized scalar energy
    - `.forces`: normalized flattened force vector (`Vector{Float32}`, length `3 * n_atoms`)
"""


struct Sample
    energy::Float32
    forces::Array{Float32 , 2}  
end


function data_preprocess(input_data, energies , forces ; split=[0.6, 0.2, 0.2]::Vector{Float64})

    ϵ = Float32(1e-6)


    ##### --- Split dataset --- #####
    ((x_train, x_val, x_test), 
     (e_train, e_val, e_test), 
     (f_train, f_val, f_test)) = partition([input_data, energies, forces], split)

    ##### --- ENERGY: Single Z-score normalization --- #####
    e_mean = mean(e_train) 
    e_std  = std(e_train, corrected=false) + ϵ
    e_train .= (e_train .- e_mean) ./ e_std
    e_val   .= (e_val .- e_mean) ./ e_std
    e_test  .= (e_test .- e_mean) ./ e_std

    ##### --- FORCES: Feature-wise Z-score normalization --- #####
    forces_mean = mean(f_train, dims=1)
    forces_std  = std(f_train, dims=1, corrected=false) .+ ϵ

    f_train .= (f_train .- forces_mean) ./ forces_std
    f_val   .= (f_val   .- forces_mean) ./ forces_std
    f_test  .= (f_test  .- forces_mean) ./ forces_std


    ##### --- Repack as Sample structs --- #####
    y_train = [Sample(e_train[i], reshape(f_train[i, :], 1, :)) for i in eachindex(e_train)]
    y_val   = [Sample(e_val[i],   reshape(f_val[i, :], 1, :))   for i in eachindex(e_val)]
    y_test  = [Sample(e_test[i],  reshape(f_test[i, :], 1, :))  for i in eachindex(e_test)]


    ##### --- Return --- #####
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), e_mean, e_std, forces_mean, forces_std
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
    xyz_to_nn_input(file_path::String) -> (Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std, species, all_cells)

Processes an XYZ file containing atomic structures and energies to generate datasets formatted for neural network training.

# Arguments
- `file_path::String`:  
  Path to the XYZ file that includes atomic coordinates, species information, lattice cells, and energy values.

# Returns
- `Train`: Training dataset consisting of input-output pairs (positions, energies, forces), typically normalized.
- `Val`: Validation dataset for tuning model hyperparameters.
- `Test_data`: Test dataset used for final evaluation.
- `energy_mean::Float64`: Mean of the energy values across the dataset, used for normalization.
- `energy_std::Float64`: Standard deviation of the energy values, used for normalization.
- `forces_mean::Float64`: Mean of the forces values, used for normalization.
- `forces_std::Float64`: Standard deviation of the forces values, used for normalization.
- `species::Vector{String}`: List of atomic species present in the dataset; useful as input to model creation functions.
- `all_cells`: List of lattice cell information corresponding to each sample in the dataset.

# Description
This function performs the following steps:  
1. Extracts atomic structures, species, lattice cells, and energy values from the specified XYZ file using `extract_data`.  
2. Generates neural network input arrays for atomic positions and forces using `prepare_nn_data`.  
3. Normalizes energies and forces and splits the data into training, validation, and test sets through `data_preprocess`.  
The outputs are ready-to-use datasets and normalization parameters for machine learning workflows involving atomic simulations.

# Dependencies
- `extract_data(file_path)`: Parses the XYZ file to obtain raw atomic and energetic data.  
- `prepare_nn_data(dataset, num_atoms)`: Converts raw atomic data into flattened arrays suitable for NN input.  
- `data_preprocess(nn_input_dataset, all_energies, all_forces)`: Normalizes data and splits it into subsets.



"""

function xyz_to_nn_input(file_path::String)

    # Extract atomic and structural information from the input XYZ file
    N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)

    # Create the neural network input dataset and forces 
    nn_input_dataset , all_forces = prepare_nn_data(dataset, N_atoms)


    # Preprocess data: normalize, split into train, validation, and test sets
    Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std = data_preprocess(nn_input_dataset, all_energies, all_forces)
    
    # Return the processed datasets and normalization parameters
    return (Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std, species, all_cells)
end
