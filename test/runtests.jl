using Test
using SymmLearn

include("../src/MLTrain.jl")
include("../src/ReadFile.jl")
include("../src/Utils.jl")

#Qui il grande codice

#forse meglio tenere le funzioni come cose separate e non come parte del grande codice che fa tutto


#=input del grande codice: 
1) Percorso file di input
1.5) Percorso file di output ( modello, grafici )
2) raggio di cutoff
3) verbose 
4) se si ha già un modello pronto in questo formato lo si può dare in input ???

=#


# Step 1: extract the informations from the input file
# in this first step the important information from the .xyz file is extracted and converted it into Julia arrays


file_path="test/train.xyz"



N_atoms, species, all_cells, dataset, all_energies = extract_data(file_path)

nn_input_dataset = create_nn_input(dataset, all_cells, N_atoms) #dataset creation for the neural network



# Step 2: data preprocessing 
# in this step normalization and suddivision in test train and validation are done 

Train,Val,Test_data,y_mean,y_std=data_preprocess(nn_input_dataset, all_energies)


#Step 3: Neural network 
# in this step the neural network is chosen/created


models = create_model(["Cs","Pb","I"], 6.5f0, 8, false)


#step 4 Neural Network training
trained_model=train_model!(models,
Train[1], 
Train[2], 
Val[1],
Val[2] ,
loss_function;
initial_lr=0.01, min_lr=1e-5, decay_factor=0.5, patience=25, 
epochs=2, batch_size=32, verbose=true)




