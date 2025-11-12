# Usage

After installing the package, load it with:

```julia
using SymmLearn
```

This section illustrates the **expected workflow** for using `SymmLearn.jl`, from raw data extraction to model construction and training.
The goal is to demonstrate how each core component integrates into a complete machine-learning force field pipeline.

## Data Preparation

The first step is to extract and preprocess data from an `.xyz` file containing all configurations.
This can be done in a single call:

```julia
x_train, y_train, 
x_val,   y_val, 
x_test,  y_test, 
e_mean,  e_std, 
unique_species, species_idx, lattice = 
    xyz_to_nn_input(path_to_your_XYZ_file)

```
This function performs all the following:

- Parses atomic configurations and extracts energy and force labels
- Splits the dataset into training, validation, and test sets
- Normalizes energies (returns `e_mean` and `e_std` for rescaling)
- Identifies atomic species and their indices
- Extracts the lattice matrix 

**Note:** Currently, `SymmLearn.jl` assumes a fixed lattice for all configurations.  
The lattice from the first structure is used for the entire dataset.

## Model Construction

Next, build a model based on the desired level of symmetry and complexity.
You can control:
- The number of radial symmetry functions, `N_G1`
- The number of angular symmetry functions, `N_G2` #TODO
- The cutoff radius, `r_cutoff`
- The network `depth`, which determines model complexity

```julia
N_G1 = 5 
N_G2 = 2
r_cutoff = 2.75f0
depth = 1

model = build_species_models(unique_species, species_idx, N_G1, N_G2, r_cutoff , depth = 1 ) 
```

Each atomic species gets its own subnetwork.
The total energy of the system is computed as the sum of all atomic contributions.

## Model Training

Once the model is built, train it using the prepared datasets.
The training function supports energy-only or energy + force fitting, adaptive learning rate decay, and early stopping. We suggest to check the documentation for the [`SymmLearn.train_model!`](@ref) function for additional information.

```julia
 trained_model, train_loss, val_loss = train_model!(
        model,
        x_train, y_train,
        x_val, y_val;
         forces = true,  epochs = 500 , initial_lr = 1e-3 )
```
This process optimizes the neural network parameters to minimize the energy and (optionally) force prediction errors.

Now that the machine learning potential has been trained you can check it's loss plot, how well it predicts the data, or even use it for molecular dynamics via the Molly interface!