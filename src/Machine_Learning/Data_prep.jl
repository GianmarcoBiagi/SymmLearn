using ExtXYZ
"""
    Sample(energy::Float32, forces::Array{Float32,2})

Container for target data of one atomic configuration.

# Fields
- `energy::Float32`: Total energy of the configuration.
- `forces::Array{Float32,2}`: Forces acting on the atoms, shape (num_atoms, 3).
"""
struct Sample
    energy::Float32
    forces::Array{Float32,2}
    
end

function Base.show(io::IO, target::Sample)

    print(io, "Energy = $(target.energy), Forces = $(target.forces))")
end

"""
    AtomInput(species::Int, coord::AbstractVector)

Container for one atom in a structure.
- `species`: integer index in 1..K identifying the species.
- `coord`: atomic coordinates as a vector.
"""
struct AtomInput{T<:AbstractVector}
    species::Int
    coord::T
end

Base.size(ai::AtomInput) = size(ai.coord)
Base.length(ai::AtomInput) = length(ai.coord)
Base.iterate(ai::AtomInput) = iterate(ai.coord)

function Base.show(io::IO, atom::AtomInput)
    coords_str = join(atom.coord, ", ")
    print(io, "AtomInput(id=$(atom.species), coordinates=[$coords_str])")
end

"""
    G1Input(species::Int, dist::AbstractMatrix)

Container for one atom in a structure.
- `species`: integer index in 1..K identifying the species.
- `dist`: matrix of distances between the atom and the others.
"""
struct G1Input{T<:AbstractMatrix}
    species::Int
    dist::T
end

Base.size(ai::G1Input) = size(ai.dist)
Base.length(ai::G1Input) = length(ai.dist)
Base.iterate(ai::G1Input) = iterate(ai.dist)

function Base.show(io::IO, obj::G1Input)
 
    print(io, "G1Input(id= $(obj.species), distances= $(obj.dist))")
end


"""
    extract_data(path::String)

### Description:
Extracts structural and energetic information from atomic configurations in `.xyz` format using `ExtXYZ`.

### Arguments:
- `path::String`: Path to the file containing the data.

### Returns:
A tuple containing:
1. `atoms_in_a_cell::Int`: Number of atoms in one cell (assumed constant across frames).
2. `species::Vector{String}`: Species of atoms in the first frame.
3. `unique_species::Vector{String}`: Unique species present in the system.
4. `all_cells::Vector{Matrix{Float32}}`: List of cell matrices for each configuration.
5. `dataset::Array{Float32, 3}`: 3D array with data per configuration:
   - Rows 1–3: Atomic positions (x, y, z).
   - Rows 4–6: Forces (fx, fy, fz).
6. `all_energies::Vector{Float32}`: Energies for each configuration.


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
```
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
                    species_order::Vector{String},
                    unique_species::Vector{String})

Convert a dataset of atomic positions and forces into atom-wise inputs for a neural network.

# Arguments
- `dataset`: Array of shape (6, num_atoms, num_samples).
- `species_order`: Vector of species strings, length = num_atoms, defining atom order.
- `unique_species`: Vector of unique species strings, used to build the mapping.

# Returns
- `all_structures::Vector{Vector{AtomInput}}`:
    Each element is a structure represented as a vector of atoms.
- `forces::Array{Float32,3}`:
    Shape (num_samples, num_atoms, 3), atomic forces for each sample.
- `species_idx::Dict{String,Int}`:
    Mapping from species name to integer index, consistent across atoms.
"""
function prepare_nn_data(dataset::Array{Float32,3},
                         species_order::Vector{String},
                         unique_species::Vector{String})

    _, num_atoms, num_samples = size(dataset)

    # build species → index map once
    species_idx = Dict(s => i for (i, s) in enumerate(unique_species))


    all_structures = Vector{Vector{AtomInput}}(undef, num_samples)
    forces = Array{Float32}(undef, num_samples, num_atoms , 3)

    for i in 1:num_samples
        current_structure = dataset[:, :, i]
        atoms = Vector{AtomInput}(undef, num_atoms)

        for j in 1:num_atoms
            sp = species_idx[species_order[j]]     # int index
            feats = @view current_structure[1:3, j] # positions as features
            atoms[j] = AtomInput(sp, feats)
            forces[i, j , 1:3] = current_structure[4:6, j]
        end

        all_structures[i] = atoms
    end

    return all_structures, forces, species_idx
end



"""
    data_preprocess(input_data, energies, forces; split=[0.6, 0.2, 0.2])

Preprocess input features and target data for neural network training. The dataset is split,
energies normalized, forces rescaled consistently with energies, and targets repackaged
into `Sample` structs.

# Arguments
- `input_data::Array{<:Real,2}`: Input features of shape `(N_structures, 3 * n_atoms)`.
- `energies::Vector{Float32}`: Total system energies for each structure.
- `forces::Array{Float32,3}`: Atomic forces of shape `(N_structures, n_atoms, 3)`.
- `split::Vector{Float64}` (optional): Fractions for train/validation/test splits,
  must sum to 1. Default `[0.6, 0.2, 0.2]`.

# Returns
A tuple containing:
- `x_train::Array, y_train::Vector{Sample}`: Training inputs and targets.
- `x_val::Array, y_val::Vector{Sample}`: Validation inputs and targets.
- `x_test::Array, y_test::Vector{Sample}`: Test inputs and targets.
- `(energy_mean::Float32, energy_std::Float32)`: Statistics used for energy normalization.

# Notes
- Energies are normalized by Z-score (zero mean, unit variance).
- Forces are scaled by the same energy standard deviation (`energy_std`).
- Targets are returned as `Sample` structs with:
    - `.energy`: normalized scalar energy.
    - `.forces`: rescaled force matrix `(n_atoms, 3)`.
"""
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
    f_train .= f_train ./ e_std
    f_test .= f_test ./ e_std
    f_val .= f_val ./ e_std


    ##### --- Repack as Sample structs --- #####
    y_train = [Sample(e_train[i], f_train[i, : , :]) for i in eachindex(e_train)]
    y_val   = [Sample(e_val[i],   f_val[i, : , :])   for i in eachindex(e_val)]
    y_test  = [Sample(e_test[i],  f_test[i, : , :])  for i in eachindex(e_test)]


    ##### --- Return --- #####
    return x_train[: , :], y_train, x_val[: , :], y_val, x_test[: , :], y_test, e_mean, e_std
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
    xyz_to_nn_input(file_path::String)

Process an XYZ file containing atomic structures and energies to generate datasets for neural network training.

# Arguments
- `file_path::String`: Path to the XYZ file containing coordinates, species, lattice cells, and energy values.

# Returns
A tuple with:
1. `x_train::Array, y_train::Vector{Sample}`: Training inputs and targets.
2. `x_val::Array, y_val::Vector{Sample}`: Validation inputs and targets.
3. `x_test::Array, y_test::Vector{Sample}`: Test inputs and targets.
4. `(energy_mean::Float32, energy_std::Float32)`: Energy normalization statistics.
5. `unique_species::Vector{String}`: Unique atomic species in the dataset.
6. `species_idx::Dict{String,Int}`: Mapping from species to integer indices.
7. `all_cells::Vector{Matrix{Float32}}`: Lattice cell matrices for each configuration.

# Description
Steps performed:
1. Extract structures, species, cells, and energies with `extract_data`.
2. Convert positions and forces into atom-wise inputs with `prepare_nn_data`.
3. Normalize energies and rescale forces consistently, then split into train/val/test sets with `data_preprocess`.

# Dependencies
- `extract_data(file_path)`: Parses raw atomic data from the XYZ file.  
- `prepare_nn_data(dataset, species, unique_species)`: Builds NN inputs and forces arrays.  
- `data_preprocess(nn_input_dataset, all_energies, all_forces)`: Normalizes and splits the dataset.
# Example
```julia
file_path = "example_structures.xyz"
x_train, y_train, x_val, y_val, x_test, y_test, (energy_mean, energy_std), unique_species, species_idx, all_cells = xyz_to_nn_input(file_path)
println("Training set size: ", length(x_train))
println("Unique species: ", unique_species)
```
"""
function xyz_to_nn_input(file_path::String)
    # Input validation
    if !isfile(file_path)
        println("Error: The specified file does not exist. Check the path: '$file_path'")
        exit(1)
    end
    if !endswith(file_path, ".xyz")
        println("Error: The file must have a .xyz extension. Rename or provide a valid file.")
        exit(1)
    end

    try
        # Extract atomic and structural information
        N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)
    catch e
        println("Error in extract_data: $(e). Verify that the XYZ file is properly formatted.")
        exit(1)
    end

    if N_atoms <= 0 || isempty(species) || isempty(unique_species)
        println("Error: No atoms or species found. Check the content of the XYZ file.")
        exit(1)
    end

    try
        # Create the neural network input dataset and forces
        nn_input_dataset, all_forces, species_idx = prepare_nn_data(dataset, species, unique_species)
    catch e
        println("Error in prepare_nn_data: $(e). Ensure that the extracted data are consistent and complete.")
        exit(1)
    end

    if isempty(nn_input_dataset) || isempty(all_forces)
        println("Error: Empty dataset or forces. Make sure the input data are valid.")
        exit(1)
    end

    try
        # Preprocess data: normalize and split into train, validation, and test sets
        x_train, y_train, x_val, y_val, x_test, y_test, mean, std = data_preprocess(nn_input_dataset, all_energies, all_forces)
    catch e
        println("Error in data_preprocess: $(e). Check the input data and normalization parameters.")
        exit(1)
    end

    if any(isnan.(mean)) || any(isnan.(std))
        println("Error: NaN values detected in normalization parameters. Remove or correct problematic data.")
        exit(1)
    end

    # Return processed datasets and normalization parameters
    return (x_train, y_train, x_val, y_val, x_test, y_test, mean, std, unique_species, species_idx, all_cells[1])
end

