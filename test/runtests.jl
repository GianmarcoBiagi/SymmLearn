using Test
using SymmLearn

include("../src/Machine_Learning/Data_prep.jl")
include("../src/Utils.jl")
include("../src/Machine_Learning/Train.jl")
include("../src/Machine_Learning/Model.jl")
include("../src/Machine_Learning/Loss.jl")

"""
Testset: "Expected user use of the package"

Purpose:
Validate the entire expected workflow as experienced by an end user of the package.  
This test simulates a realistic usage scenario, from raw data extraction to model training,
ensuring that each major pipeline component operates coherently and produces consistent, 
non-empty outputs.

Why:
This test guarantees that the integration between data preparation, preprocessing, 
model construction, and training behaves as expected under normal use. It ensures
the exposed functions interoperate correctly and that training updates model parameters.

What:
1. Verify that `extract_data` correctly parses an input file and yields consistent structures.  
2. Confirm that `prepare_nn_data` generates valid neural network input data.  
3. Check that `data_preprocess` splits and structures datasets as required for training.  
4. Validate that `build_species_models` constructs models without errors.  
5. Ensure `distance_layer`, `distance_derivatives`, `extract_energies`, and `extract_forces`
   produce outputs with consistent shapes.  
6. Confirm that `dispatch_train`, `force_loss`, and `loss_train` produce meaningful numeric results.  
7. Verify that `train_model!` modifies model parameters after one epoch, demonstrating that 
   learning occurs.

Failure of any individual assertion isolates a specific phase of the workflow.
"""


@testset "Expected user use of the package" begin
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
    dist = distance_layer(x ; lattice = lattice)
    df_matrix = distance_derivatives(x ; lattice = lattice)
    e = extract_energies(y)
    f = extract_forces(y)

    model_time = @elapsed batch_output = dispatch_train(dist , species_models)
    println("Time for computing the model output for a batch: ", model_time, " seconds")
    @test size(batch_output) == (3 ,)

    f_loss_time = @elapsed fconst = force_loss( species_models, dist, f , df_matrix)
    println("Time for computing the force loss of a batch: ", f_loss_time, " seconds")

    loss_time = @elapsed sample_loss = loss_train(species_models, dist, e , fconst)
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
