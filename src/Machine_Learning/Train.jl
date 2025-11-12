using ProgressMeter
using Flux
using Enzyme


"""
    train_model!(model, x_train, y_train, x_val, y_val, 
                 λ=1.0f0; forces=true, initial_lr=0.01, min_lr=1e-6, decay_factor=0.1, 
                 patience=50, epochs=1000, batch_size=32, verbose=false, lattice=nothing)

Train a neural network model to predict total energies (and optionally forces) 
for atomic structures using mini-batch gradient descent, adaptive learning rate, 
and early stopping.

# Arguments
- `model::Flux.Chain`  
  Neural network to train. Can combine multiple branches for different atomic species.

- `x_train::Any`  
  Training atomic structures, either batched or compatible with `distance_layer`.

- `y_train::Vector{Sample}`  
  Ground-truth labels containing `.energy` and `.forces` for training.

- `x_val::Any`  
  Validation atomic structures, same format as `x_train`.

- `y_val::Vector{Sample}`  
  Ground-truth labels for validation.

- `λ::Float32` (default = 1.0)  
  Weight applied to the force loss relative to the energy loss.

- `forces::Bool` (default = true)  
  If true, include force loss in addition to energy loss.

- `initial_lr::Float32` (default = 0.01)  
  Initial learning rate for the Adam optimizer.

- `min_lr::Float32` (default = 1e-6)  
  Minimum allowed learning rate; training stops decaying when reached.

- `decay_factor::Float32` (default = 0.1)  
  Factor by which to multiply the learning rate if validation loss plateaus.

- `patience::Int` (default = 50)  
  Number of epochs without improvement before reducing the learning rate.

- `epochs::Int` (default = 1000)  
  Maximum number of training epochs.

- `batch_size::Int` (default = 32)  
  Number of structures per mini-batch.

- `verbose::Bool` (default = false)  
  If true, prints progress, learning rate changes, and final results.

- `lattice::Union{Nothing, Matrix{Float32}}` (default = nothing)  
  Optional 3×3 lattice matrix if distances should be computed under PBC.

# Returns
- `best_model::Flux.Chain`  
  Model achieving the lowest validation loss during training.

- `loss_tr::Vector{Float32}`  
  Training loss per epoch.

- `loss_val::Vector{Float32}`  
  Validation loss per epoch.

# Description
1. Precomputes pairwise distances (`distance_layer`) and derivatives (`distance_derivatives`) for both training and validation sets.
2. Performs mini-batch gradient descent using `Enzyme.gradient` and `Flux.update!`.
3. Optionally computes forces in addition to energies in the loss function.
4. Applies adaptive learning rate: decays learning rate if validation loss does not improve for `patience` epochs.
5. Saves the model achieving the lowest validation loss (`best_model`).
6. Supports early stopping if the learning rate reaches `min_lr`.

# Notes
- Forces are computed using analytical derivatives of distances and backpropagated through the model.
- The `λ` parameter balances energy and force contributions in the total loss.
- Loss values are monitored to prevent NaNs; training will stop if a NaN is detected.

```julia
# Assume models, x_train, y_train, x_val, y_val are defined
best_model, train_loss, val_loss = train_model!(
    model, x_train, y_train, x_val, y_val;
    λ=0.5f0, forces=true, initial_lr=0.01, epochs=500, batch_size=16, verbose=true
)

println("Training finished. Best validation loss: ", minimum(val_loss))```
"""
function train_model!(
    model,
    x_train, y_train,
    x_val, y_val;
    λ::Float32=1.0f0, forces::Bool=true, initial_lr::Float32=0.01f0, min_lr::Float32=0.00001f0,
    decay_factor::Float32=0.1f0, patience::Int=50, epochs::Int=1000, batch_size::Int=32,
    verbose::Bool=false, lattice::Union{Nothing, Matrix{Float32}}=nothing)

  # Input validation
  if model === nothing
    println("Error: 'model' is missing. Provide a valid Flux.Chain model.")
    exit(1)
  end
  if x_train === nothing || isempty(x_train) || y_train === nothing || isempty(y_train)
    println("Error: Training data is missing or empty. Provide valid x_train and y_train.")
    exit(1)
  end
  if x_val === nothing || isempty(x_val) || y_val === nothing || isempty(y_val)
    println("Error: Validation data is missing or empty. Provide valid x_val and y_val.")
    exit(1)
  end
  if initial_lr <= 0f0
    println("Error: 'initial_lr' must be positive.")
    exit(1)
  end
  if min_lr <= 0f0
    println("Error: 'min_lr' must be positive.")
    exit(1)
  end
  if decay_factor <= 0f0 || decay_factor >= 1f0
    println("Error: 'decay_factor' must be between 0 and 1.")
    exit(1)
  end
  if epochs <= 0 || batch_size <= 0
    println("Error: 'epochs' and 'batch_size' must be positive integers.")
    exit(1)
  end
  if lattice !== nothing && !(typeof(lattice) <: Matrix{Float32})
    println("Error: 'lattice' must be a Matrix{Float32} or nothing.")
    exit(1)
  end

  # Optimizer setup
  o = OptimiserChain(ClipNorm(1.0), Adam(initial_lr))
  opt = Flux.setup(o, model)
  N = size(x_train, 1)

  # Precompute energies and forces

  e_t = extract_energies(y_train)
  f_t = extract_forces(y_train)
  e_v = extract_energies(y_val)
  f_v = extract_forces(y_val)


  dist_train = distance_layer(x_train; lattice=lattice)
  dist_val   = distance_layer(x_val; lattice=lattice)
  d_matrix_train = distance_derivatives(x_train; lattice=lattice)
  d_matrix_val   = distance_derivatives(x_val; lattice=lattice)


  current_lr = initial_lr
  best_model = deepcopy(model)
  best_epoch = 1
  best_loss  = Inf
  no_improve_count = 0

  loss_tr = zeros(Float32, epochs)
  loss_val = zeros(Float32, epochs)

    @showprogress for epoch in 1:epochs
        # Mini-batch gradient update
        for batch in batch_indices(N, batch_size)
            xb = dist_train[batch, :]
            yb = y_train[batch]
            x_der = d_matrix_train[batch, :, :, :]
            e = e_t[batch]
            f = f_t[batch, :, :]
            fconst = forces ? force_loss(model, xb, f, x_der) : 0f0


            grad = Enzyme.gradient(set_runtime_activity(Reverse),
                                       (m, x, ee, ff) -> loss_train(m, x, ee, ff; λ),
                                       model, Const(xb), Const(e), Const(fconst))[1]


       

            Flux.update!(opt, model, grad)
        end

        # Full train/val loss computation
        fconst_t = forces ? force_loss(model, dist_train, f_t, d_matrix_train) : 0f0
        fconst_v = forces ? force_loss(model, dist_val, f_v, d_matrix_val) : 0f0


        loss_tr[epoch]  = loss_train(model, dist_train, e_t, fconst_t; λ)
        loss_val[epoch] = loss_train(model, dist_val, e_v, fconst_v; λ)


        if isnan(loss_tr[epoch]) || isnan(loss_val[epoch])
            println("Error: NaN detected in loss at epoch $epoch. Try reducing learning rate.")
            exit(1)
        end

        if verbose && epoch % 50 == 0
            println("------- epoch $epoch -------")
            println("Training loss: ", loss_tr[epoch])
            println("Validation loss: ", loss_val[epoch])
        end

        # Learning rate decay and early stopping

            opt, current_lr, no_improve_count, stop_training =
                maybe_decay_lr!(opt, current_lr, no_improve_count,
                                patience, decay_factor, min_lr, epoch; verbose=verbose)


            best_model, best_loss, best_epoch, no_improve_count =
                maybe_save_best!(model, loss_val[epoch], epoch,
                                 best_model, best_loss, best_epoch, no_improve_count)

 

        if stop_training
            println("Early stopping at epoch $epoch: learning rate reached min_lr")
            break
        end
    end

    if verbose
        println("Final training loss: ", loss_tr[best_epoch])
        println("Final validation loss: ", loss_val[best_epoch])
    end

    return best_model, loss_tr, loss_val
end





"""
    maybe_save_best!(model, loss_val, epoch, best_model, best_loss, best_epoch, no_improve_count; tol=1.0)

Update the record of the best-performing model based on validation loss.

This function checks whether the current validation loss `loss_val` is smaller than `tol * best_loss`.  
If the condition is met, the current model is considered an improvement:
- `best_model` is updated via `deepcopy(model)`,
- `best_loss` is set to `loss_val`,
- `best_epoch` is updated to the current `epoch`,
- `no_improve_count` is reset to 0.

If there is no improvement, only `no_improve_count` is incremented by 1.

# Arguments
- `model`: Current neural network model.
- `loss_val::Float32`: Validation loss for the current epoch.
- `epoch::Int`: Current epoch number.
- `best_model`: Model corresponding to the best observed validation loss.
- `best_loss::Float32`: Best validation loss recorded so far.
- `best_epoch::Int`: Epoch number when `best_model` was saved.
- `no_improve_count::Int`: Number of consecutive epochs without improvement.
- `tol::Float32=1.0`: Tolerance factor. A new model is considered better if `loss_val < tol * best_loss`.

# Returns
Tuple `(best_model, best_loss, best_epoch, no_improve_count)` with updated values.

# Notes
- Setting `tol < 1.0` allows small tolerance before updating the best model.
- This function is useful for implementing early stopping and adaptive learning rate strategies in training loops.
"""
function maybe_save_best!(model, loss_val, epoch, best_model, best_loss, best_epoch, no_improve_count; tol=1.0)
    if loss_val < best_loss * tol
        return deepcopy(model), loss_val, epoch, 0
    else
        return best_model, best_loss, best_epoch, no_improve_count + 1
    end
end


"""
    maybe_decay_lr!(opt, current_lr, no_improve_count, patience, decay_factor, min_lr, epoch; verbose=false)

Check whether the learning rate should be decayed based on validation performance.

If the number of consecutive epochs without improvement (`no_improve_count`) reaches `patience`,
the learning rate is multiplied by `decay_factor`, but not below `min_lr`. The optimizer state
is updated in place to preserve momentum terms.

# Arguments
- `opt`: Optimizer (e.g., `Flux.Optimise.Adam`) whose learning rate will be modified.
- `current_lr::Float32`: Current learning rate.
- `no_improve_count::Int`: Number of consecutive epochs without improvement.
- `patience::Int`: Number of epochs to wait before decaying LR.
- `decay_factor::Float32`: Factor to multiply learning rate when decaying.
- `min_lr::Float32`: Minimum allowed learning rate.
- `epoch::Int`: Current epoch number.
- `verbose::Bool=false`: Print information when learning rate is decayed.

# Returns
Tuple `(opt, current_lr, no_improve_count, stop_training::Bool)`:
- `opt`: Updated optimizer with new learning rate if decayed.
- `current_lr`: Updated learning rate.
- `no_improve_count`: Reset to 0 if LR was decayed, else unchanged.
- `stop_training::Bool`: True if `current_lr` reached `min_lr`, signaling training should stop.

# Notes
- This function allows adaptive learning rate schedules without restarting training.
- The optimizer’s momentum buffers remain intact.
"""
function maybe_decay_lr!(opt, current_lr, no_improve_count, patience, decay_factor, min_lr, epoch; verbose=false)
    if no_improve_count >= patience
        new_lr = max(current_lr * decay_factor, min_lr)
        update_lr!(opt, new_lr)
        if verbose
            println("Reducing learning rate to $new_lr at epoch $epoch")
        end
        stop_training = new_lr < min_lr
        return opt, new_lr, 0, stop_training
    else
        return opt, current_lr, no_improve_count, false
    end
end


"""
    update_lr!(opt, new_lr)

Recursively update the learning rate `eta` for all Adam optimizer instances
contained within a possibly nested optimizer structure.

# Arguments
- `opt`: Optimizer or optimizer tree (e.g., `OptimiserChain`) potentially containing multiple Adam leaves.
- `new_lr::Float32`: New learning rate to assign.

# Returns
- `opt`: The same optimizer object with updated learning rates.

# Notes
- Only Adam optimizers (`Flux.Optimise.Adam`) are modified.
- Other optimizer types or wrappers are left unchanged.
- This function modifies the optimizer in-place.
"""
function update_lr!(opt, new_lr)
    for leaf in opt
        for (k, v) in pairs(leaf)
            if v isa Optimisers.Leaf && v.opt isa Flux.Optimise.Adam
                v.opt.eta = new_lr
            end
        end
    end
    return opt
end


"""
    batch_indices(n::Int, batchsize::Int) -> Vector{Vector{Int}}

Generate mini-batch indices for stochastic gradient descent.

# Arguments
- `n::Int`: Total number of samples in the dataset.
- `batchsize::Int`: Size of each mini-batch.

# Returns
- `Vector{Vector{Int}}`: Shuffled list of index vectors, each representing a mini-batch.  
  The last batch may contain fewer than `batchsize` elements if `n` is not divisible by `batchsize`.

# Notes
- The indices are shuffled randomly each call to ensure stochasticity.
- Useful for iterating over training data in `train_model!`.
"""
function batch_indices(n, batchsize)
    # Restituisce un vettore di vettori con gli indici dei batch
    idx = collect(1:n)
    shuffle!(idx)
    [idx[i:min(i+batchsize-1, n)] for i in 1:batchsize:n]
end



