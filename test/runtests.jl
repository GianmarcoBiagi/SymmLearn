using Test
using SymmLearn

include("../src/MLTrain.jl")
include("../src/ReadFile.jl")
include("../src/Utils.jl")

#the create_toy_model function is almost the same as the create_model one, the only difference is that the model created here has far less parameters to make the test quicker
#for further info check the documentation for create_model
function create_toy_model(
    ions::Vector{String}, 
    R_cutoff::Float32, 
    G1_number::Int = 5, 
    verbose::Bool = false
)
    # Get ion charges using element_charge dictionary
    ion_charges = 0.1f0 * getindex.(Ref(element_to_charge), ions)

    # Number of ions
    n_of_ions = length(ions)

    # Create an array to store ModelContainer instances
    models = Vector{ModelContainer}(undef, n_of_ions)

    # Create a neural network model for each ion and assign to the array
    for i in 1:n_of_ions
        ion_name = ions[i]
        
        # Create the model
        model = Chain(
            MyLayer(1, G1_number, R_cutoff, ion_charges[i]),
            Dense(G1_number, 2),
            Dense(2, 1)
        )
        
        # Store the model inside ModelContainer
        models[i] = ModelContainer(ion_name, model)
    end

        # Check if a GPU is available and move models to GPU if possible
    if CUDA.functional()
        device_name = CUDA.name(CUDA.device())  # Get GPU name
        models = [ModelContainer(m.name, m.model |> gpu) for m in models]
        println("Models successfully mounted on GPU: ", device_name)
    else
        println("No GPU available. Models remain on CPU.")
    end

    # Print model details if verbose is enabled
    if verbose
        for container in models
            println("A model has been created called: ", container.name)
            println(container.model)
            println("────────────────────────────────")
        end
    end

    # Convert models array into an immutable Tuple
    models_final = Tuple(models)

    return models_final  # Return the immutable Tuple of models
end

@testset "Model Training Test" begin
    file_path = "test/reduced_train.xyz"
    
    # Step 1: Extract information from the input file
    time_extract = @elapsed N_atoms, species, all_cells, dataset, all_energies = extract_data(file_path)
    println("Time for extract_data: ", time_extract, " seconds")
    @test !isempty(N_atoms)
    @test !isempty(species)
    @test size(dataset, 2) == N_atoms

    # Step 2: Create neural network input
    time_nn_input = @elapsed nn_input_dataset = create_nn_input(dataset, all_cells, N_atoms)
    println("Time for create_nn_input: ", time_nn_input, " seconds")
    @test size(nn_input_dataset) == (size(dataset, 3), size(dataset, 2), N_atoms)

    # Step 3: Data preprocessing
    time_preprocess = @elapsed Train, Val, Test_data, y_mean, y_std = data_preprocess(nn_input_dataset, all_energies)
    println("Time for data_preprocess: ", time_preprocess, " seconds")
    @test size(Train[1],1) == size(Train[2],1)
    @test size(Val[1],1) == size(Val[2],1)

    # Step 4: Create models
    time_create_model = @elapsed models = create_toy_model(species, 1.5f0, 1, false)
    println("Time for create_model: ", time_create_model, " seconds")
    @test length(models) == 3

    # Step 5: Train the model
    time_train = @elapsed trained_model = train_model!(
        models,
        Train[1], 
        Train[2], 
        Val[1],
        Val[2],
        loss_function;
        epochs=1, batch_size=4, verbose=false
    )
    println("Time for train_model!: ", time_train, " seconds")
    @test !isempty(trained_model)
end
