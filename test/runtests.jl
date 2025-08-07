using Test
using SymmLearn


include("../src/MLTrain.jl")
include("../src/Data_prep.jl")
include("../src/Utils.jl")






@testset "Model Training Test" begin
    file_path = "test/reduced_train.xyz"
    
    # Step 1: Extract information from the input file
    time_extract = @elapsed N_atoms, species, unique_species, all_cells, dataset, all_energies = extract_data(file_path)
    println("Time for extract_data: ", time_extract, " seconds")
    @test !isempty(N_atoms)
    @test !isempty(species)
    @test size(dataset, 2) == N_atoms


    
    # Step 2: Create neural network input
    time_nn_input = @elapsed nn_input_dataset = create_nn_input(dataset, N_atoms)
    println("Time for create_nn_input: ", time_nn_input, " seconds")
    @test size(nn_input_dataset) == (size(dataset, 3), N_atoms * 3)


    # Step 3: Create neural network target
    time_nn_target = @elapsed target = create_nn_target(dataset, all_energies)
    println("Time for create_nn_target: ", time_nn_target, " seconds")


    # Step 4: Data preprocessing
    time_preprocess = @elapsed Train, Val, _, _, _, _ = data_preprocess(nn_input_dataset, target,split=[0.6, 0.2, 0.2])
    println("Time for data_preprocess: ", time_preprocess, " seconds")


    

    # Step 5 & 6: Build the full model in one step
    time_build_model = @elapsed model = build_model(species, 1, 5.0f0)
    println("Time for build_total_model_inline: ", time_build_model, " seconds")



    x_sample = Train[1][1:1, :]
    y_sample = Train[2][1]
    x_batch = Train[1][1:3, :]
    y_batch = Train[2][1:3]

    sample_loss = loss_function(model,x_sample,y_sample)
    @test sample_loss != 0f0 

    sample_output = model(x_sample)
    @test size(sample_output) == (1,)

    batch_output = model(x_batch)
    @test size(batch_output) == (3 , 1)


   
    #extract params for a later test
    params , _ = Flux.destructure(model)

    # Step 7: Train the model
    
    time_train = @elapsed final_model,trained_model,train_loss,val_loss = train_model!(
        model,
        Train[1], 
        Train[2], 
        Val[1],
        Val[2],
        loss_function;
         initial_lr=0.1,epochs=1, batch_size=1, verbose=false
    )
    println("Time for train_model!: ", time_train, " seconds")

    # Check to see if parameters actually changed after the training

    trained_params , _ = Flux.destructure(final_model)

    @test trained_params != params
  

    

end
