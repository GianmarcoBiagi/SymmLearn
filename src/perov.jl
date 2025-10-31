using Flux
using Enzyme
using Statistics
using Plots

include("../src/Data_prep.jl")
include("../src/Utils.jl")
include("../src/Train.jl")
include("../src/Model.jl")


file_path = "src/dataset.xyz"

Train, Val, Test_data, energy_mean, energy_std, _, _, unique_species, species_idx, _= xyz_to_nn_input(file_path)



println(" Now the data is in the correct format")

#define the model using 5 G1 symmetry functions


model = build_species_models(unique_species, species_idx, 5, Float32(5.0))

#We can check if the model and the loss work as we expected on a small batch 

x_batch = Train[1][1:3, :]
y_batch = Train[2][1:3]
batch_dist = distance_matrix_layer(x_batch)


e = extract_energies(y_batch)
f = Float32(0.0) # we won't need to train the model using the forces



println("Model output with a batch as input: ", dispatch(batch_dist , model))
println("Model loss with batch input: ", loss(model, batch_dist, e , f))

#we train the model setting the parameter forces to false

last_model, train_loss, val_loss = train_model!(
        model,
        Train,
        Val;
         forces = true , initial_lr = 0.1, epochs = 500, batch_size = 8, verbose = true
    )

energies = [Train[2][i].energies for i in 1:length(Train[2])]

predict = dispatch(last_model, Train[1][:,:])

scatter(energies, predict)


# energie vere
energies = [Train[2][i].energy for i in 1:length(Train[2])]

# predizioni con il modello migliore
predict = dispatch(last_model, Train[1][:,:])

# RMSE totale
rmse_total = sqrt(mean((energies .- predict).^2))

# RMSE per atomo (dividendo per 40)
rmse_per_atom = rmse_total / 40

println("RMSE totale: ", rmse_total)
println("RMSE per atomo: ", rmse_per_atom)

# Plot
plt = scatter(
    energies, predict,
    xlabel = "True energy [eV]",
    ylabel = "Predicted energy [eV]",
    title = "Energy prediction vs True energy",
    label = "Predictions",
    legend = :topleft,
    alpha = 0.7,
    markerstrokewidth = 0.5,
    markersize = 6
)

# linea guida y=x
plot!(plt, energies, energies, label="y = x", color=:red, lw=2, ls=:dash)

# salva immagine
savefig(plt, "energy_prediction.png")
println("Plot salvato come energy_prediction.png")