using Test
using SymmLearn

include("../src/Data_prep.jl")
include("../src/Utils.jl")
include("../src/Train.jl")
include("../src/Model.jl")


@testset "Model Training Test" begin
    file_path = "test/reduced_train.xyz"
    
    # Step 1: Extract information from the input file
    time_extract = @elapsed N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)
    println("Time for extract_data: ", time_extract, " seconds")
    @test !isempty(N_atoms)
    @test !isempty(species)
    @test size(dataset, 2) == N_atoms


    lattice = all_cells[1]
    
    # Step 2: Create neural network input
    time_nn_input = @elapsed nn_input_dataset , all_forces, species_idx = prepare_nn_data(dataset, species , unique_species)
    println("Time for prepare_nn_data: ", time_nn_input, " seconds")
    @test size(nn_input_dataset) == (10 ,)
    



    # Step 3: Data preprocessing
    time_preprocess = @elapsed x_train , y_train , x_val , y_val , _ , _ , _ = data_preprocess(nn_input_dataset, all_energies, all_forces ,split=[0.6, 0.2, 0.2])
    println("Time for data_preprocess: ", time_preprocess, " seconds")


    

    # Step 4 build the full model
    time_build_model = @elapsed species_models = build_species_models(unique_species, species_idx, 5, 8.0f0)
    println("Time for build_model: ", time_build_model, " seconds")



    x = x_train[1:3, :]
    y = y_train[1:3]
    dist = distance_matrix_layer(x ; lattice = lattice)
    df_matrix = distance_derivatives(x ; lattice = lattice)
    e = extract_energies(y)
    f = extract_forces(y)

    model_time = @elapsed batch_output = dispatch(dist , species_models)
    println("Time for computing the model output for a batch: ", model_time, " seconds")
    @test size(batch_output) == (3 ,)

    f_loss_time = @elapsed fconst = force_loss( species_models, dist, f , df_matrix)
    println("Time for computing the force loss of a batch: ", f_loss_time, " seconds")

    loss_time = @elapsed sample_loss = loss(species_models, dist, e , fconst)
    println("Time for computing the loss of a a batch: ", loss_time, " seconds")
    @test mean(sample_loss) > 0f0 




   
    #extract params for a later test
    params , _ = Flux.destructure(species_models)

    # Step 7: Train the model
    
    time_train = @elapsed trained_model,train_loss,val_loss = train_model!(
        species_models,
        x_train, y_train, 
        x_val, y_val,
         epochs=1 , batch_size=3 , lattice = lattice)

    println("Time for train_model!: ", time_train, " seconds")

    # Check to see if parameters actually changed after the training

    trained_params , _ = Flux.destructure(trained_model)

    @test trained_params != params
  

    

end
