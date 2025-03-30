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
2. `species::Vector{String}`: A vector containing the unique species (elements) present in the system.
3. `all_cells::Array{Float32, 3}`: A 3D array containing the cell matrices (dimensions) for each configuration.
4. `dataset::Array{Float32, 3}`: A 3D array with the following data for each configuration:
   - Row 1: Atomic charges (mapped from species)
   - Rows 2-4: Atomic positions (x, y, z)
   - Rows 5-7: Forces acting on atoms (fx, fy, fz)
5. `all_energies::Vector{Float32}`: A vector containing the total energy for each configuration.

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
    dataset = zeros(Float32, 7, atoms_in_a_cell, n_of_configs)  # 3D dataset to store atomic information for each configuration


    # Extract the unique species (elements) used in the system by removing duplicates from the species list
    unique_species = Set(frame[1]["arrays"]["species"])  # Convert species to a set to remove duplicates
    species = collect(unique_species)  # Convert the set back into an array (optional, if you need it as an array)

    # Extract cell matrices for each configuration
    all_cells = [frame[i]["cell"] for i in 1:n_of_configs]

    # Extract energy values for each configuration
    all_energies = [frame[i]["info"]["energy"] for i in 1:n_of_configs]
    
    
    # Extract atom-specific data (charge, position, and forces) for each configuration
    for i in 1:n_of_configs
        for j in 1:atoms_in_a_cell
  
            # Store charge (mapped from species)
            dataset[1, j, i] = element_to_charge[frame[i]["arrays"]["species"][j]]

            # Store atomic positions (first 3 elements: x, y, z)
            dataset[2:4, j, i] = frame[i]["arrays"]["pos"][1:3, j]

            # Store forces (first 3 components: fx, fy, fz)
            dataset[5:7, j, i] = frame[i]["arrays"]["forces"][1:3, j]
        end
    end

    # Return the extracted data: number of atoms, species, cell matrices, dataset, and energies
    return atoms_in_a_cell, species, all_cells, dataset, all_energies
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
function create_nn_input(dataset, all_lattice, num_atoms::Int32)
    num_datasets = size(dataset)[3]


    # Input array shape: (num_datasets, num_atoms, num_atoms) of type Float32
    nn_input = Array{Float32, 3}(undef, num_datasets, num_atoms, num_atoms)

    for i in 1:num_datasets
        current_dataset = dataset[:,:,i]
        lattice_vectors = all_lattice[i,:]

        for j in 1:num_atoms
            atom_j = current_dataset[:,j]
            charge_j = atom_j[1]  # Ensure charge is a scalar value
            pos_j = atom_j[2:4]
            # Insert the atom charge into the first slot
            nn_input[i, j, 1] = charge_j


            # Calculate the distance to other atoms
            slot_index = 2  # Start from the second slot
            for k in 1:num_atoms
                if j == k
                    continue  # Skip distance to itself
                end
                atom_k = current_dataset[:,k]
                pos_k = atom_k[2:4]

                # Calculate the distance with PBC
                distance = distance_with_pbc(pos_j, pos_k, lattice_vectors[1])

                nn_input[i, j, slot_index] = distance
                slot_index += 1  # Move to the next slot
            end
        end
    end

    return nn_input
end
