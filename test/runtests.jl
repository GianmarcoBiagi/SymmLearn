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



@testset "G1Layer tests" begin
    test_g1layer_initialization()
    test_g1layer_forward_pass()
    test_g1layer_positive_output()
end




