using ExtXYZ

"""
    AtomInput(species::Int, features::AbstractVector)

Container for one atom in a structure.
- `species`: integer index in 1..K identifying the species.
- `coord`: atomic coordinates.


"""
struct AtomInput{T<:AbstractVector}
    species::Int
    coord::T
end

Base.size(ai::AtomInput) = size(ai.coord)

function Base.show(io::IO, atom::AtomInput)
    coords_str = join(atom.coord, ", ")
    print(io, "AtomInput(id=$(atom.species), coordinates=[$coords_str])")
end

"""
    G1Input(species::Int, features::AbstractVector)

Container for one atom in a structure.
- `species`: integer index in 1..K identifying the species.
- `distances`: distances from the atom N and the other N-1 atoms


"""
struct G1Input{T<:AbstractMatrix}
    species::Int
    dist::T
end

Base.size(ai::G1Input) = size(ai.dist)

function Base.show(io::IO, obj::G1Input)
 
    print(io, "G1Input(id= $(obj.species), distances= $(obj.dist))")
end


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
    prepare_nn_data(dataset::Array{Float32,3},
                    species_order::Vector{String})

Convert a dataset of positions+forces into atom-wise inputs for the NN.

# Arguments
- `dataset`: Array of shape (6, num_atoms, num_samples).
- `species_order`: Vector of species strings, length = num_atoms.

# Returns
- `all_structures::Vector{Vector{AtomInput}}`:
    Each element is a structure (vector of atoms).
- `forces::Array{Float32,2}`:
    Shape (num_samples, num_atoms*3), flattened atomic forces.
- `species_idx::Dict{String,Int}`:
    Mapping from species name to index, consistent across atoms.
"""
function prepare_nn_data(dataset::Array{Float32,3},
                         species_order::Vector{String},
                         unique_species::Vector{String})

    _, num_atoms, num_samples = size(dataset)

    # build species → index map once
    species_idx = Dict(s => i for (i, s) in enumerate(unique_species))


    all_structures = Vector{Vector{AtomInput}}(undef, num_samples)
    forces = Array{Float32}(undef, num_samples, num_atoms * 3)

    for i in 1:num_samples
        current_structure = dataset[:, :, i]
        atoms = Vector{AtomInput}(undef, num_atoms)

        for j in 1:num_atoms
            sp = species_idx[species_order[j]]     # int index
            feats = @view current_structure[1:3, j] # positions as features
            atoms[j] = AtomInput(sp, feats)
            forces[i, (1+(j-1)*3):(3*j)] = current_structure[4:6, j]
        end

        all_structures[i] = atoms
    end

    return all_structures, forces, species_idx
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
    forces::Array{Float32,2}
    
end

function Base.show(io::IO, target::Sample)

    print(io, "Energy = $(target.energy), Forces = $(target.forces))")
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
    y_train = [Sample(e_train[i], f_train[i, : , :]) for i in eachindex(e_train)]
    y_val   = [Sample(e_val[i],   f_val[i, : , :])   for i in eachindex(e_val)]
    y_test  = [Sample(e_test[i],  f_test[i, : , :])  for i in eachindex(e_test)]


    ##### --- Return --- #####
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), e_mean, e_std, forces_mean, forces_std
end





"""
    partition(data::Vector{<:AbstractArray}, parts::Vector{Float64}; shuffle=true, rng=Random.GLOBAL_RNG)

Split a list of datasets into multiple parts (e.g., train/val/test).

- `data`: Vector of datasets (e.g., `[x, e, f]`), each can be array or vector of custom objects.
- `parts`: Vector of fractions summing to 1.0, e.g., `[0.7, 0.2, 0.1]`.
- `shuffle`: Whether to shuffle indices before splitting.
- `rng`: Random number generator.

Returns a tuple of vectors of splits for each dataset.
"""
function partition(data::Vector, parts::Vector{Float64}; shuffle=true, rng=Random.GLOBAL_RNG)
    N = size(data[1], 1)  # numero di campioni
    @assert all(size(d,1) == N for d in data) "All datasets must have the same first dimension"
    @assert sum(parts) ≈ 1 "Sum of `parts` must be 1.0"

    # Shuffle indices
    ridx = shuffle ? Random.shuffle(rng, 1:N) : collect(1:N)

    # Compute boundaries
    boundaries = [0; cumsum(round.(Int, parts .* N))]
    boundaries[end] = N  # ensure last index reaches N

    # Split each dataset
    splits = []
    for d in data
        # determine number of dimensions
        nd = ndims(d)
        tmp = []
        for i in 1:length(parts)
            idx = ridx[boundaries[i]+1 : boundaries[i+1]]
            if nd == 1
                push!(tmp, d[idx])
            elseif nd == 2
                push!(tmp, d[idx, :])
            else
                # per array ND maggiore di 2
                slicer = ntuple(j -> j==1 ? idx : Colon(), nd)
                push!(tmp, d[slicer...])
            end
        end
        push!(splits, tmp)
    end

    return tuple(splits...)
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
    nn_input_dataset , all_forces, species_idx = prepare_nn_data(dataset, species, unique_species)


    # Preprocess data: normalize, split into train, validation, and test sets
    Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std = data_preprocess(nn_input_dataset, all_energies, all_forces)
    
    # Return the processed datasets and normalization parameters
    return (Train, Val, Test_data, energy_mean, energy_std, forces_mean, forces_std, unique_species, species_idx, all_cells)
end
