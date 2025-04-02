using Test
using SymmLearn

include("../src/MLTrain.jl")
include("../src/ReadFile.jl")
include("../src/Utils.jl")

# Funzione di test che simula il processo completo
@testset "Model Training Test" begin
    # Step 1: Extract information from the input file
    file_path = "test/train.xyz"
    
    # Assuming the extract_data function works correctly and returns valid data
    N_atoms, species, all_cells, dataset, all_energies = extract_data(file_path)
    @test !isempty(N_atoms)  # Check if N_atoms is not empty
    @test !isempty(species)  # Check if species list is not empty
    @test size(all_cells, 1) == N_atoms  # Ensure the number of atoms in cells matches

    # Step 2: Create neural network input
    nn_input_dataset = create_nn_input(dataset, all_cells, N_atoms)
    @test size(nn_input_dataset) == (size(dataset, 1), size(dataset, 2), N_atoms)  # Check if input dataset has correct shape
    
    # Step 3: Data preprocessing
    Train, Val, Test_data, y_mean, y_std = data_preprocess(nn_input_dataset, all_energies)
    @test size(Train[1]) == size(Train[2])  # Ensure that the training set input and output match in size
    @test size(Val[1]) == size(Val[2])  # Ensure that the validation set input and output match in size
    
    # Step 4: Create models
    models = create_model(["Cs", "Pb", "I"], 6.5f0, 8, false)
    @test length(models) == 3  # Ensure 3 models are created for the 3 elements
    
    # Step 5: Train the model
    trained_model = train_model!(
        models,
        Train[1], 
        Train[2], 
        Val[1],
        Val[2],
        loss_function;
        initial_lr=0.01, min_lr=1e-5, decay_factor=0.5, patience=25, 
        epochs=2, batch_size=32, verbose=true
    )
    @test !isempty(trained_model)  # Ensure that a trained model is returned
end
