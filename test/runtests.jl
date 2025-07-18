using Test
using SymmLearn
using Enzyme

include("../src/MLTrain.jl")
include("../src/Data_prep.jl")
include("../src/Utils.jl")

#the create_toy_model function is almost the same as the create_model one, the only difference is that the model created here has far less parameters to make the test quicker
#for further info check the documentation for create_model


struct ToyModel
    branches::Vector{Chain}
end




(m::ToyModel)(input) = begin
    total = m.branches[1](input)
    for i in 2:length(m.branches)
        total = total + m.branches[i](input)
    end
    return reshape(total, :)
end





function build_toy_model(
    species_order::Vector{String},
    G1_number::Int,
    R_cutoff::Float32;
    lattice::Union{Nothing, Matrix{Float32}} = nothing
)
    N = length(species_order)

    branches = [begin
        atom = species_order[i]
        charge = 0.1f0 * element_to_charge[atom]

        Chain(
            x -> distance_layer(x, i, lattice),
            MyLayer(1, G1_number, R_cutoff, charge),
            Dense(G1_number, 1)
        )
    end for i in 1:N]

    return ToyModel(branches)
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


    

    # Step 5 & 6: Build the full model in one step
    time_build_model = @elapsed model = build_toy_model(species, 1, 5.0f0; lattice=all_cells[1])
    println("Time for build_total_model_inline: ", time_build_model, " seconds")

   


    #extract params for a later test
    params , _ = Flux.destructure(model)

    x_sample= Train[1][1:3,:,:]

    y_sample=Train[2][1:3]


    println("model output with a batch as an input: ",model(x_sample))

    println("model loss on the sample with a batch as an input: ",loss_function(model,x_sample,y_sample))

    x_sample= Train[1][1,:,:]

    y_sample=Train[2][1]


    println("model output with a single configuration as an input: ",model(x_sample))

    println("model loss on the sample with a single configuration as an input: ",loss_function(model,x_sample,y_sample))

    exit(0)

    # Step 7: Train the model

    time_train = @elapsed trained_model,train_loss,val_loss = train_model!(
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

    _ , trained_params = Flux.destructure(model)

    @test trained_params != params
  

    

end
