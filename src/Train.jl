using ProgressMeter
using Flux
using Enzyme


"""
    train_model!(model, x_train, y_train, x_val, y_val, loss_function; 
                 initial_lr=0.01f0, min_lr=1e-5, decay_factor=0.5, patience=25, 
                 epochs=3000, batch_size=32, verbose=true, forces=true)

Trains a neural network model to predict total energies (and optionally forces) of atomic structures
using mini-batch gradient descent with adaptive learning rate and early stopping.

# Arguments
- `model::Flux.Chain`  
  Neural network model to train. Can combine multiple branches for different atom types.

- `x_train::Any`  
  Training dataset. Each entry is a tuple of atomic features.

- `y_train::Vector{Float32}`  
  Ground truth energies (and optionally forces) for the training set.

- `x_val::Any`  
  Validation dataset, same format as `x_train`.

- `y_val::Vector{Float32}`  
  Ground truth energies (and optionally forces) for the validation set.

- `λ::Float32=10.0`  
  Relative weight given to the force loss (default for energy is 1).

- `initial_lr::Float32=0.01`  
  Initial learning rate for the `Adam` optimizer.

- `min_lr::Float32=1e-5`  
  Minimum learning rate. Training will stop decaying when this is reached.

- `decay_factor::Float32=0.5`  
  Factor to multiply the learning rate by when validation loss plateaus.

- `patience::Int=25`  
  Number of epochs without improvement before decaying the learning rate.

- `epochs::Int=3000`  
  Maximum number of training epochs.

- `batch_size::Int=32`  
  Number of structures per mini-batch.

- `verbose::Bool=true`  
  If true, prints training progress, learning rate changes, and final results.

- `forces::Bool=true`  
  Whether to include force loss in addition to energy loss.

- `lattice::Union{Nothing, Matrix{Float32}}=nothing`  
  If the distances must be computed using pbs pass the lattice as an input.

# Returns

- `best_model::Flux.Chain`  
  Model achieving the lowest validation loss during training.

- `loss_tr::Vector{Float32}`  
  Training loss per epoch (length equals number of epochs run).

- `loss_val::Vector{Float32}`  
  Validation loss per epoch (length equals number of epochs run).

# Notes
- The learning rate is reduced if validation loss does not improve for `patience` epochs.
- Early stopping is implicit: the training may stop improving while still running until `epochs`.
- The best-performing model on the validation set is saved and returned separately.
"""



function train_model!(
    model,
    x_train, y_train,
    x_val, y_val;
    λ = 1.0f0 , forces = true, initial_lr=0.01, min_lr=1e-6, decay_factor=0.1, patience=50,
    epochs=1000, batch_size=32, verbose=false, lattice::Union{Nothing, Matrix{Float32}}=nothing
)

    o =  OptimiserChain(ClipNorm(1.0), Adam(initial_lr))

    # Setup dell'ottimizzatore con il modello
    opt = Flux.setup(o, model)
    N = size(x_train , 1)


    # Precompute validation targets
    e_v = extract_energies(y_val)
    f_v = extract_forces(y_val ; ndims = 2)
    e_t = extract_energies(y_train)
    f_t = extract_forces(y_train ; ndims = 2)

    dist_train = distance_layer(x_train ; lattice = lattice )
    dist_val = distance_layer(x_val ; lattice = lattice)

    d_matrix_train = distance_derivatives(x_train ; lattice = lattice)
    d_matrix_val = distance_derivatives(x_val; lattice = lattice)

  

    current_lr   = initial_lr
    best_model   = deepcopy(model)
    best_epoch   = 1
    best_loss    = Inf
    no_improve_count = 0
 

    loss_tr = zeros(Float32, epochs)
    loss_val   = zeros(Float32, epochs)

    @showprogress for epoch in 1:epochs
      # Loop sui batch
      for batch in batch_indices(N, batch_size)
       
        xb = dist_train[batch, :]
        yb = y_train[batch]

        x_der = d_matrix_train[batch, : , : ,:]
    

        e = e_t[batch]
        f = f_t[batch , :]
            

        fconst = forces ? force_loss(model, xb, f , x_der) : 0f0
   
  

        grad = Enzyme.gradient(set_runtime_activity(Reverse),
                              (m, x, ee, ff) -> loss_train(m, x, ee, ff ; λ),
                              model, Const(xb), Const(e), Const(fconst))[1]

      

        Flux.update!(opt, model, grad)
      end

      # Calcolo della loss sull’intero train e val set
    
      fconst_t = forces ? force_loss(model, dist_train, f_t, d_matrix_train) : 0f0
      fconst_v = forces ? force_loss(model, dist_val, f_v, d_matrix_val)   : 0f0


      loss_tr[epoch] = loss_train(model, dist_train, e_t, fconst_t ; λ)
      loss_val[epoch]   = loss_train(model, dist_val,   e_v, fconst_v ; λ)

      if isnan(loss_tr[epoch]) || isnan(loss_val[epoch])
        println("Something is wrong, the computed loss is NaN , getting out from the train function")
        println("try a smaller learning rate!")
        break
      end



      if verbose && epoch%50 == 0
        println("------- epoch  $epoch -------")
        println("The loss on the train dataset is $(loss_tr[epoch])")
        println("The loss on the val dataset is $(loss_val[epoch])")
      end

      # Gestione decay del learning rate ed early stopping
      opt, current_lr, no_improve_count, stop_training =
          maybe_decay_lr!(opt, current_lr, no_improve_count,
                          patience, decay_factor, min_lr, epoch; verbose=verbose)

      # Salvataggio del best model
      best_model, best_loss, best_epoch, no_improve_count =
        maybe_save_best!(model, loss_val[epoch], epoch,
                        best_model, best_loss, best_epoch, no_improve_count)

      if stop_training
        println("Early stopping at epoch $epoch: learning rate reached min_lr")
        break
      end
  end

  if verbose
      println("Final Train Loss: ", loss_tr[best_epoch])
      println("Final Val Loss: ", loss_val[best_epoch])
  end

  return  best_model, loss_tr, loss_val
end




"""
    maybe_save_best!(model, loss_val, epoch, best_model, best_loss, best_epoch, no_improve_count; tol=0.98)

Check if the current validation loss is better than the best recorded loss (by a factor `tol`).
If yes, update `best_model`, `best_loss`, `best_epoch`, and reset `no_improve_count`.

# Arguments
- `model`: the current model.
- `loss_val`: validation loss at the current epoch.
- `epoch`: current epoch number.
- `best_model`: current best model (deepcopy will be done if improved).
- `best_loss`: best validation loss recorded so far.
- `best_epoch`: epoch number of the best model.
- `no_improve_count`: counter for epochs without improvement.
- `tol`: improvement tolerance factor (default: `0.98`).

# Returns
Updated `(best_model, best_loss, best_epoch, no_improve_count)`.
"""
function maybe_save_best!(model, loss_val, epoch, best_model, best_loss, best_epoch, no_improve_count; tol=1.0)
    if loss_val < best_loss * tol
        return deepcopy(model), loss_val, epoch, 0
    else
        return best_model, best_loss, best_epoch, no_improve_count + 1
    end
end


"""
    maybe_decay_lr!(opt::Flux.Optimise.Adam, current_lr, no_improve_count, patience, decay_factor, min_lr, epoch; verbose=false)

Decay the learning rate of the Adam optimizer in place after `patience` epochs without improvement.
Preserves optimizer state (momenta).

# Arguments
- `opt`: Adam optimizer (with internal state).
- `current_lr`: current learning rate (Float).
- `no_improve_count`: epochs without improvement.
- `patience`: patience threshold before decaying LR.
- `decay_factor`: multiplicative factor to reduce LR.
- `min_lr`: minimum allowed learning rate.
- `epoch`: current epoch.
- `verbose`: print info if true.

# Returns
Updated `(opt, current_lr, no_improve_count, stop_training::Bool)`.
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

Update learning rate `eta` for all Adam leaves inside the optimizer state tree.
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

Generates shuffled mini-batch indices for a dataset.

# Arguments
- `n::Int`: Total number of samples in the dataset.
- `batchsize::Int`: Desired size of each mini-batch.

# Returns
- `Vector{Vector{Int}}`: A vector of vectors, where each subvector contains 
  the indices of the samples belonging to a mini-batch.  
  The indices are shuffled before the dataset is split.
"""


function batch_indices(n, batchsize)
    # Restituisce un vettore di vettori con gli indici dei batch
    idx = collect(1:n)
    shuffle!(idx)
    [idx[i:min(i+batchsize-1, n)] for i in 1:batchsize:n]
end



