using Flux
using Enzyme
using ProgressMeter



"""
    f_out(model, x::AbstractVector) -> Float32

Compute the scalar output (first component) of the model for a single input.

# Arguments
- `model::Function`: A callable model (e.g. `Flux.Chain`) that maps input structures to outputs.
- `x::AbstractVector`: A vector representing a single input structure.

# Returns
- `Float32`: The scalar prediction of the model for this input.
"""
function f_out(model, x::AbstractVector)
    # Extract first component as scalar output
    return model(x)[1]
end

"""
    energy_loss(model, x, y) -> AbstractArray

Compute the squared error between predicted energy and reference energy.

# Arguments
- `model::Function`: A callable model.
- `x`: Input structure.
- `y`: Reference scalar energy.

# Returns
- Squared error as an array.
"""
function energy_loss( model, x, y)

    return (dispatch(x , model) .- y).^2
end

"""disp
    calculate_force(model, x::AbstractVector) -> AbstractVector

Compute the negative gradient of the scalar model output w.r.t. input `x`, i.e., the predicted forces.

# Arguments
- `model::Function`: Callable model.
- `x::AbstractVector`: Single input structure.

# Returns
- `AbstractVector`: Predicted forces of same size as `x`.
"""
function calculate_force( x::AbstractVector , model)

    # Enzyme gradient: returns a tuple (grad w.r.t model, grad w.r.t x)
    grad_x , _ = Enzyme.gradient(set_runtime_activity(Reverse) , (x,s) -> dispatch(x,s),  x , Const(model))
    forces = vcat([-g.coord for g in grad_x[1]]...) 
    return forces

end




"""
    force_loss(model, x::AbstractVector, f::AbstractVector) -> Float32

Compute mean squared error (MSE) between predicted and reference forces for a single input.

# Arguments
- `model::Function`: Callable model.
- `x::AbstractVector`: Single input structure.
- `f::AbstractVector`: Reference forces.

# Returns
- `Float32`: MSE loss for this input.
"""
function force_loss(model, x::AbstractVector,  f)
    predicted_forces = calculate_force(x , model)

    return mean((predicted_forces .- f) .^2)
end

"""
    force_loss(model, X::AbstractMatrix, F::AbstractMatrix) -> Vector{Float32}

Compute force losses for a batch of inputs.

# Arguments
- `model::Function`: Callable model.
- `X::AbstractMatrix`: Batch of inputs (rows = examples).
- `F::AbstractMatrix`: Corresponding reference forces.

# Returns
- `Vector{Float32}`: Force loss for each example in the batch.
"""
function force_loss(model, X::AbstractMatrix, F::AbstractMatrix)
    # Map each example in the batch to its force loss
    losses = map(1:size(X, 1)) do i

        force_loss(model, X[i , :] , F[i , :])
    end
    return losses'
end

"""
    loss(model, x, y, fconst; λ=0.5f0) -> Float32

Compute the combined energy + force loss for a single input.

# Arguments
- `model::Function`: Callable model.
- `x`: Input structure.
- `y`: Reference energy.
- `fconst`: Precomputed force term (treated as constant).
- `λ::Float32`: Weight for the force contribution (default 0.5).

# Returns
- `Float32`: Total loss combining energy and force term.
"""
function loss(models, x, y, fconst; λ::Float32=0.5f0)    
    e_loss = energy_loss( models , x , y)
   

    return sum(e_loss .+ λ .* fconst)
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
function maybe_save_best!(model, loss_val, epoch, best_model, best_loss, best_epoch, no_improve_count; tol=0.98)
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
        stop_training = new_lr <= min_lr
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

- `loss_function::Function`  
  Function to compute the loss: `loss_function(model, data, targets)`.

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

# Returns
- `model::Flux.Chain`  
  The model after the final training epoch.

- `best_model::Flux.Chain`  
  Model achieving the lowest validation loss during training.

- `loss_train::Vector{Float32}`  
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
    x_train,
    y_train,
    x_val,
    y_val;
    forces = true, initial_lr=0.01, min_lr=1e-4, decay_factor=0.5, patience=25,
    epochs=1000, batch_size=32, verbose=false
)

    opt = Flux.setup(Adam(initial_lr), model)
    N = size(x_train , 1)


    # Precompute validation targets
    e_v = extract_energies(y_val)
    f_v = extract_forces(y_val)
    e_t = extract_energies(y_train)
    f_t = extract_forces(y_train)
  

    current_lr   = initial_lr
    best_model   = deepcopy(model)
    best_epoch   = 1
    best_loss    = Inf
    no_improve_count = 0
 

    loss_train = zeros(Float32, epochs)
    loss_val   = zeros(Float32, epochs)

    @showprogress for epoch in 1:epochs
      # Loop sui batch
      for batch in batch_indices(N, batch_size)
       
        xb = x_train[batch, :]
        yb = y_train[batch]
    

        e = extract_energies(yb)
        f = extract_forces(yb)
            

        fconst = forces ? force_loss(model, xb, f) : 0f0
   

        grad = Enzyme.gradient(set_runtime_activity(Reverse),
                              (m, x, ee, ff) -> loss(m, x, ee, ff),
                              model, Const(xb), Const(e), Const(fconst))
 

        Flux.update!(opt, model, grad[1])
      end

      # Calcolo della loss sull’intero train e val set
    
      fconst_t = forces ? force_loss(model, x_train, f_t) : 0f0
      fconst_v = forces ? force_loss(model, x_val, f_v)   : 0f0


      loss_train[epoch] = loss(model, x_train, e_t, fconst_t)
      loss_val[epoch]   = loss(model, x_val,   e_v, fconst_v)

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
      println("Final Train Loss: ", loss_train[best_epoch])
      println("Final Val Loss: ", loss_val[best_epoch])
  end

  return model, best_model, loss_train, loss_val
end



function batch_indices(n, batchsize)
    # Restituisce un vettore di vettori con gli indici dei batch
    idx = collect(1:n)
    shuffle!(idx)
    [idx[i:min(i+batchsize-1, n)] for i in 1:batchsize:n]
end

function train_model_small!(
    model,
    x_train,
    y_train;
    forces=true, initial_lr=0.01, epochs=500, batchsize=32
)

    N = size(x_train, 1)  # numero di campioni
    opt = Flux.setup(Adam(initial_lr), model)

    @showprogress for epoch in 1:epochs
        for batch in batch_indices(N, batchsize)
            xb = x_train[batch , :]   # prendi subset del batch
            yb = y_train[batch]

            e = extract_energies(yb)
            f = extract_forces(yb)

            fconst = forces ? force_loss(model, xb, f) : 0f0

            grad = Enzyme.gradient(set_runtime_activity(Reverse),
                                   (m, x, ee, ff) -> loss(m, x, ee, ff),
                                   model, Const(xb), Const(e), Const(fconst))

            Flux.update!(opt, model, grad[1])
        end
    end

    return model
end
