using Test
using SymmLearn

include("../src/MLTrain.jl")
include("../src/Data_prep.jl")
include("../src/Utils.jl")

#the create_toy_model function is almost the same as the create_model one, the only difference is that the model created here has far less parameters to make the test quicker
#for further info check the documentation for create_model


function build_toy_branch(Atom_name::String,G1_number::Int,R_cutoff::Float32)
    ion_charge = 0.1f0 * element_to_charge[Atom_name]
    return Chain(
        MyLayer(1, G1_number, R_cutoff, ion_charge),
        Dense(G1_number, 1)
    )
end


function create_toy_species_models(
    unique_species::Vector{String},
    G1_number::Int,
    R_cutoff::Float32
)
    models = Dict{String, Chain}()
    for sp in unique_species
        # build_branch returns a Flux.Chain for that species
        models[sp] = build_toy_branch(sp, G1_number, R_cutoff)
    end
    return models
end


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
    @test size(nn_input_dataset) == (size(dataset, 3), N_atoms, 3)


    # Step 3: Create neural network target
    time_nn_target = @elapsed target = create_nn_target(dataset, all_energies)
    println("Time for create_nn_target: ", time_nn_target, " seconds")


    # Step 4: Data preprocessing
    time_preprocess = @elapsed Train, Val, _, _, _, _ = data_preprocess(nn_input_dataset, target,split=[0.6, 0.2, 0.2])
    println("Time for data_preprocess: ", time_preprocess, " seconds")


    

    # Step 5: Create branches
    time_create_species_model = @elapsed species_models = create_toy_species_models(species,1,5.0f0)   
    println("Time for create_species_model: ", time_create_species_model, " seconds")
    @test length(species_models) == size(unique_species)[1]

    # Step 6: Create models
    time_assemble_model = @elapsed model = assemble_model(species_models, species,all_cells[1])
    println("Time for assemble_model: ", time_assemble_model, " seconds")

    #extract params for a later test
    params , _ = Flux.destructure(model)

    x_sample= Train[1][1:3,:,:]

    y_sample=Train[2][1:3]


    println("model output: ",model(x_sample))

    println("model loss on the sample: ",loss_function(model,x_sample,y_sample))



    # Step 7: Train the model

    time_train = @elapsed trained_model,train_loss,val_loss = train_model!(
        model,
        Train[1], 
        Train[2], 
        Val[1],
        Val[2],
        loss_function;
         initial_lr=0.1,epochs=1, batch_size=4, verbose=false
    )
    println("Time for train_model!: ", time_train, " seconds")

    # Check to see if parameters actually changed after the training

    _ , trained_params = Flux.destructure(model)

    @test trained_params != params
  

    

end
