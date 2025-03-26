"""
    process_dataset(file_path::AbstractString, N::Int) -> Tuple{Vector{Vector{Float32}}, Vector{Vector{Vector{Float32}}}, Vector{Float32}}

Processes a dataset from a file in .xyz format by extracting lattice vectors, energies, and atomic data for each structure. 
The function organizes the data into three parts: lattice vectors, atomic datasets, and energy values. The sections in the file are now identified by the number `N` provided as input, instead of the previous static value `40`.

### Arguments
- `file_path::AbstractString`: The path to the dataset file to be processed.
- `N::Int`: In practice it's the number of atoms in the cell, this number is used to identify the start of a new dataset section. When this number is found, the current dataset is saved and a new one is started.

### Returns
- `Tuple{Vector{Vector{Float32}}, Vector{Vector{Vector{Float32}}}, Vector{Float32}}`: A tuple containing:
  1. `all_lattice`: A vector of lattice vectors (each a `Vector{Float32}`) for all structures in the dataset.
  2. `dataset`: A vector of datasets, where each dataset corresponds to a structure and contains atomic information (charge, coordinates, and forces).
  3. `all_energies`: A vector of energy values (each a `Float32`) for all structures.

### Example
```julia
# Example usage:
file_path = "data.txt"
N = 40  # Specify the number of atoms in the cell
lattice_vectors, datasets, energies = process_dataset(file_path, N)
println(lattice_vectors)
println(datasets)
println(energies)

"""

function process_dataset(file_path::AbstractString, N::Int)
    # Initialize containers for the processed dataset
    dataset = []  # Holds all processed datasets
    current_dataset = []  # Current dataset for the current structure
    all_lattice = []  # Lattice vectors for all structures
    all_energies = []  # Energies for all structures

    open(file_path, "r") do file
        while !eof(file)
            line = readline(file) |> strip  # Remove whitespace from the line

            if line == string(N)
                # If the value N is found, save the current dataset and initialize a new one
                if !isempty(current_dataset)
                    push!(dataset, current_dataset)
                end
                current_dataset = []  # Reset current dataset
            elseif startswith(line, "pbc=")
                # Extract lattice vectors from the corresponding line
                lattice_line = split(line, "Lattice=")[2]
                lattice_data = extract_lattice_vector(lattice_line)
                energy = extract_energy(lattice_line)
                push!(all_lattice, Float32.(lattice_data))  # Ensure lattice data is of type Float32
                push!(all_energies, Float32(energy))  # Convert energy to Float32
            else
                # Split the line into parts and analyze coordinates and forces
                parts = split(line)
                if length(parts) < 7
                    continue  # Skip if the data is incomplete
                end
                element = parts[1]

                # Extract and convert coordinates and forces to Float32
                coordinates = Float32.(parse.(Float32, parts[2:4]))  
                forces = Float32.(parse.(Float32, parts[5:7]))  
                charge = Float32(get(element_charge, element, 0.0))  # Get element charge and convert to Float32
                
                # Add to the current dataset: charge, coordinates, and forces
                push!(current_dataset, [charge, coordinates..., forces...])
            end
        end
        
        # Add the last dataset if not empty
        if !isempty(current_dataset)
            push!(dataset, current_dataset)
        end
    end

    # Return the processed lattice vectors, datasets, and energies
    return all_lattice, dataset, all_energies
end




"""
    extract_lattice_vector(lattice_str::AbstractString) -> Vector{Float32}

Extracts the first three non-zero numerical values from a string representing a lattice vector. 
The string is expected to be in .xyz format

### Arguments
- `lattice_str::AbstractString`: A string representing a lattice vector, containing numeric values (e.g., "1.0 0.0 3.5 4.2").

### Returns
- `Vector{Float32}`: A vector containing the first three non-zero lattice values, converted to `Float32`. If fewer than three non-zero values are found, an empty array is returned.

### Example
```julia
# Example usage:
lattice_str = " pbc="T T T" Lattice=" 2.5 0.0 0.0 0.0 12.8 0.0 0.0 0.0 1.5" "
lattice_vector = extract_lattice_vector(lattice_str)
println(lattice_vector)  # Output: [2.5, 12.8, 1.5]

# If fewer than three non-zero values are found:
lattice_str = "0.0 0.0 0.0"
lattice_vector = extract_lattice_vector(lattice_str)
println(lattice_vector)  # Output: Float32[]
"""
# Function to extract the first 3 non-zero numbers from a lattice string
function extract_lattice_vector(lattice_str::AbstractString)
    # Extract numbers from the string, ignoring anything except digits and periods
    filtered = filter(x -> !isempty(x) && parse(Float64, x) != 0.0, 
                      split(replace(lattice_str, r"[^\d.\s]" => "")))

    # Check if there are at least 3 non-zero values
    if length(filtered) < 3
        println("Warning: Less than 3 non-zero lattice values found.")
        return Float32[]  # Return an empty Float32 array if fewer than 3 values
    end

    # Return the first 3 non-zero numbers converted to Float32
    return parse.(Float32, filtered[1:3])
end


"""
    extract_energy(line::AbstractString) -> Float32

Extracts the energy value from a string that contains the substring `"energy="`, followed by a floating-point number. 
The function assumes that the energy value is numeric and is the first value after `"energy="`.

### Arguments
- `line::AbstractString`: A string that contains the energy value in the format `energy=<value>`. 
  For example, `"energy=-28880.2597722246136982"`.

### Returns
- `Float32`: The extracted energy value as a `Float32`. 

### Example
```julia
# Example usage:
line = "energy=-28880.2597722246136982"
energy_value = extract_energy(line)
println(energy_value)  # Output: -28880.26
"""

function extract_energy(line::AbstractString)
    # Check if the string contains "energy="
    if occursin("energy=", line)
        # Split the string at "energy="
        parts = split(line, "energy=")
        # Get the part after "energy="
        energy_str = parts[2]
        
        # Find the first non-numeric character (indicating the end of the number)
        # The condition checks for digits, decimal points, and negative signs
        end_idx = findfirst(c -> !isdigit(c) && c != '.' && c != '-', energy_str) - 1
        
        # Extract the number as a substring and convert it to Float32
        energy_value = parse(Float32, energy_str[1:end_idx])  # Convert to Float32
        
        return energy_value
    else
        # If "energy=" is not found, raise an error
        error("The string does not contain 'energy='")
    end
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
# Assume `dataset` contains atomic data and `all_lattice` contains corresponding lattice vectors
nn_input = create_nn_input(dataset, all_lattice, num_atoms=40)
println(nn_input)

"""

# Function to create the neural network input array
function create_nn_input(dataset, all_lattice, num_atoms::Int)
    num_datasets = length(dataset)

    # Input array shape: (num_datasets, num_atoms, num_atoms) of type Float32
    nn_input = Array{Float32, 3}(undef, num_datasets, num_atoms, num_atoms)

    for i in 1:num_datasets
        current_dataset = dataset[i]
        lattice_vectors = all_lattice[i]

        for j in 1:num_atoms
            atom_j = current_dataset[j]
            charge_j = atom_j[1]  # Ensure charge is a scalar value
            pos_j = atom_j[2:4]

            # Insert the atom charge into the first slot
            nn_input[i, j, 1] = Float32(charge_j)

            # Convert positions to Float32
            pos_j_float = Float32.(pos_j)

            # Calculate the distance to other atoms
            slot_index = 2  # Start from the second slot
            for k in 1:num_atoms
                if j == k
                    continue  # Skip distance to itself
                end
                atom_k = current_dataset[k]
                pos_k = atom_k[2:4]

                # Convert positions to Float32
                pos_k_float = Float32.(pos_k)

                # Calculate the distance with PBC
                distance = distance_with_pbc(pos_j_float, pos_k_float, lattice_vectors)

                # Ensure the distance is of type Float32
                nn_input[i, j, slot_index] = Float32(distance)
                slot_index += 1  # Move to the next slot
            end
        end
    end

    return nn_input
end
