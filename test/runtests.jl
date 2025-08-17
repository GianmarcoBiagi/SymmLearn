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


    
    # Step 2: Create neural network input
    time_nn_input = @elapsed nn_input_dataset , all_forces = prepare_nn_data(dataset, N_atoms)
    println("Time for prepare_nn_data: ", time_nn_input, " seconds")
    @test size(nn_input_dataset) == (size(dataset, 3), N_atoms * 3)



    # Step 3: Data preprocessing
    time_preprocess = @elapsed Train, Val, _, _, _, _ = data_preprocess(nn_input_dataset, all_energies, all_forces ,split=[0.6, 0.2, 0.2])
    println("Time for data_preprocess: ", time_preprocess, " seconds")


    

    # Step 4 build the full model
    time_build_model = @elapsed model = build_model(species, 1, 5.0f0)
    println("Time for build_model: ", time_build_model, " seconds")



    x = Train[1][1:3, :]
    y = Train[2][1:3]
    e = extract_energies(y)
    f = extract_forces(y)

    model_time = @elapsed batch_output = model(x)
    println("Time for computing the model output for a batch: ", model_time, " seconds")
    @test size(batch_output) == (3 ,)

    f_loss_time = @elapsed fconst = force_loss(model , x , f)
    println("Time for computing the force loss of a batch: ", f_loss_time, " seconds")

    loss_time = @elapsed sample_loss = loss(model, x, e , fconst)
    println("Time for computing the loss of a a batch: ", loss_time, " seconds")
    @test mean(sample_loss) > 0f0 




   
    #extract params for a later test
    params , _ = Flux.destructure(model)

    # Step 7: Train the model
    
    time_train = @elapsed final_model,trained_model,train_loss,val_loss = train_model!(
        model,
        Train[1], 
        Train[2], 
        Val[1],
        Val[2],
        loss;
         initial_lr=0.1 , epochs=1 , batch_size=1
    )
    println("Time for train_model!: ", time_train, " seconds")

    # Check to see if parameters actually changed after the training

    trained_params , _ = Flux.destructure(final_model)

    @test trained_params != params
  

    

end
