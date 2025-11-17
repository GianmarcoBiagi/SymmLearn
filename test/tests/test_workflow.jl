include("../../src/Machine_Learning/Data_prep.jl")
include("../../src/Utils.jl")
include("../../src/Machine_Learning/Train.jl")
include("../../src/Machine_Learning/Model.jl")
include("../../src/Machine_Learning/Loss.jl")

# ---- Test Functions with Markdown docstrings ----
function test_data_extraction()
    """
    ## WHAT
    Verify that `extract_data` correctly parses the input file and returns coherent datasets.

    ## GIVEN
    A reduced training file path.

    ## WHEN
    Calling `extract_data(file_path)`.

    ## THEN
    - N_atoms and species are non-empty.
    - Dataset has expected column dimension matching N_atoms.
    """
    data = prepare_test_data()
    @test !isempty(data[:N_atoms])
    @test !isempty(data[:species])
    @test size(data[:dataset], 2) == data[:N_atoms]
end

function test_nn_preparation()
    """
    ## WHAT
    Check that `prepare_nn_data` produces neural network-ready input structures.

    ## GIVEN
    Dataset and species information.

    ## WHEN
    Calling `prepare_nn_data(dataset, species, unique_species)`.

    ## THEN
    - Output `nn_input_dataset` length matches the number of samples.
    """
    data = prepare_test_data()
    @test length(data[:nn_input_dataset]) == size(data[:dataset], 3)
end

function test_data_preprocessing()
    """
    ## WHAT
    Validate dataset splitting for training and validation.

    ## GIVEN
    Neural network inputs, energies, and forces.

    ## WHEN
    Calling `data_preprocess(...)`.

    ## THEN
    - Training and validation datasets are non-empty.
    """
    data = prepare_test_data()
    @test !isempty(data[:x_train]) && !isempty(data[:y_train])
end

function test_model_construction()
    """
    ## WHAT
    Ensure that `build_species_models` constructs models for all species.

    ## GIVEN
    List of unique species and species index mapping.

    ## WHEN
    Building models with `build_species_models(...)`.

    ## THEN
    - Number of models equals number of unique species.
    """
    data = prepare_test_data()
    @test length(data[:species_models]) == length(data[:unique_species])
end

function test_distance_and_forces()
    """
    ## WHAT
    Verify that distance computations and force/energy extraction produce correct shapes.

    ## GIVEN
    A small batch of NN inputs and labels.

    ## WHEN
    Calling `distance_layer`, `distance_derivatives`, `extract_energies`, `extract_forces`.

    ## THEN
    - Distances, derivatives, energies, and forces have consistent expected shapes.
    """
    data = prepare_test_data()
    x_batch = data[:x_train][1:3, :]
    y_batch = data[:y_train][1:3]

    dist = distance_layer(x_batch; lattice=data[:lattice])
    df_matrix = distance_derivatives(x_batch; lattice=data[:lattice])
    e = extract_energies(y_batch)
    f = extract_forces(y_batch)

    @test length(dist) == 15
    @test size(df_matrix,1) == 3
    @test length(e) == 3
    @test size(f,1) == 3
end

function test_training_updates()
    """
    ## WHAT
    Confirm that `train_model!` updates model parameters.

    ## GIVEN
    Initialized species models and training/validation sets.

    ## WHEN
    Performing one epoch of training.

    ## THEN
    - Model parameters after training differ from before, indicating learning occurred.
    """
    data = prepare_test_data()
    params_before, _ = Flux.destructure(data[:species_models])
    trained_model, _, _ = train_model!(data[:species_models], data[:x_train], data[:y_train], data[:x_val], data[:y_val], epochs=1, batch_size=3, lattice=data[:lattice])
    params_after, _ = Flux.destructure(trained_model)
    @test params_after != params_before
end