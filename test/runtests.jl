using Test, Random, Flux
using SymmLearn
include("tests/test_d_pbc.jl")
include("tests/test_fc.jl")
include("tests/test_extract.jl")
include("tests/test_g1layer.jl")
include("tests/test_workflow.jl")


# ---- Fixtures / Setup ----
function prepare_test_data()
    """
    Prepare common dataset, lattice, NN inputs, and models for testing.

    Returns:
        Dict: all required objects keyed for access in tests
    """
    file_path = "test/reduced_train.xyz"
    N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)
    lattice = all_cells[1]
    nn_input_dataset, all_forces, species_idx = prepare_nn_data(dataset, species, unique_species)
    x_train, y_train, x_val, y_val, _, _, _ = data_preprocess(nn_input_dataset, all_energies, all_forces, split=[0.6,0.2,0.2])
    species_models = build_species_models(unique_species, species_idx, 5, 8.0f0)
    return Dict(
        :N_atoms => N_atoms,
        :species => species,
        :unique_species => unique_species,
        :dataset => dataset,
        :all_energies => all_energies,
        :lattice => lattice,
        :nn_input_dataset => nn_input_dataset,
        :all_forces => all_forces,
        :species_idx => species_idx,
        :x_train => x_train,
        :y_train => y_train,
        :x_val => x_val,
        :y_val => y_val,
        :species_models => species_models
    )
end


"""
SymmLearn expected user workflow tests
- **test_data_extraction()** – Verifies that `extract_data` correctly parses input files and produces consistent, non-empty data structures.  
- **test_nn_preparation()** – Checks that `prepare_nn_data` generates valid neural network inputs and forces for each atom.  
- **test_data_preprocessing()** – Ensures that `data_preprocess` properly splits and structures datasets for training and validation.  
- **test_model_construction()** – Confirms that `build_species_models` constructs models for all unique species without errors.  
- **test_distance_and_forces()** – Validates the consistency of `distance_layer`, `distance_derivatives`, `extract_energies`, and `extract_forces` in terms of output shapes and dimensions.  
- **test_training_updates()** – Ensures that training (`dispatch_train` + loss computation) updates model parameters, demonstrating that learning occurs."""

@testset "SymmLearn expected user workflow tests" begin
    test_data_extraction()
    test_nn_preparation()
    test_data_preprocessing()
    test_model_construction()
    test_distance_and_forces()
    test_training_updates()
end

"""
d_pbc tests
- **test_minimum_image_cubic()** – Checks that the minimum image distance in a cubic periodic cell is correctly computed.  
- **test_fractional_cartesian_consistency()** – Verifies consistency between fractional and Cartesian coordinates for PBC distance calculations.  
- **test_return_image_vector()** – Ensures the function correctly returns the minimal image vector, relative coordinates, and translation indices."""

@testset "d_pbc tests" begin
    test_minimum_image_cubic()
    test_fractional_cartesian_consistency()
    test_return_image_vector()
end

"""
fc tests
- **test_fc_cutoff_zero()** – Confirms that `fc` returns zero when beyond the cutoff.  
- **test_fc_smooth_interior()** – Checks that `fc` produces deterministic and correct values inside the cutoff.  
- **test_fc_near_rc_guard()** – Ensures the EPS guard prevents divergence when `Rij` approaches the cutoff radius."""

@testset "fc tests" begin
    test_fc_cutoff_zero()
    test_fc_smooth_interior()
    test_fc_near_rc_guard()
end

"""
extract_* tests
- **test_extract_energy_single()** – Verifies that the energy extracted from a single sample matches the stored value.  
- **test_extract_forces_single()** – Confirms that the forces extracted from a single sample match the stored forces.  
- **test_extract_forces_ndims()** – Checks that batch force extraction respects the desired shape according to the number of dimensions specified (1D, 2D, 3D)."""

@testset "extract_* tests" begin
    test_extract_energy_single()
    test_extract_forces_single()
    test_extract_forces_ndims()
end

"""
G1Layer tests
- **test_g1layer_initialization()** – Verifies deterministic layer initialization with a fixed random seed.  
- **test_g1layer_forward_pass()** – Confirms that the forward pass produces outputs with correct dimensions and deterministic results for identical inputs.  
- **test_g1layer_positive_output()** – Ensures that small positive inputs produce strictly positive outputs."""

@testset "G1Layer tests" begin
    test_g1layer_initialization()
    test_g1layer_forward_pass()
    test_g1layer_positive_output()
end

