using Flux
using Enzyme
using ProgressMeter



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
  Train,
  Val,
  loss::Function;
  forces = true,
  initial_lr=0.1, min_lr=1e-5, decay_factor=0.5, patience=25,
  epochs=3000, batch_size=32, verbose=false
)

  x_train = Train[1][:,:]
  y_train = Train[2]
  x_val = Val[1][: , :]
  y_val = Val[2]

  # Ottimizzatore
  opt = Flux.setup(Adam(initial_lr), model)
  current_lr = initial_lr

  # Per salvare il best model in modo leggero
  θ, re = Flux.destructure(model)
  best_params = copy(θ)
  best_epoch = 0
  best_loss = Inf
  no_improve_count = 0

  # Storico delle loss
  loss_train = zeros(Float32, epochs)
  loss_val = zeros(Float32, epochs)

  @showprogress for epoch in 1:epochs
    # Shuffle dati senza modificare quelli originali
    idx = randperm(size(x_train, 1))
    x_epoch = x_train[idx, :]
    y_epoch = y_train[idx]

    # Training per mini-batch
    for i in 1:batch_size:size(x_epoch, 1)
      end_idx = min(i + batch_size - 1, size(x_epoch, 1))
      x_batch = x_epoch[i:end_idx, :]
      y_batch = y_epoch[i:end_idx]

      e = extract_energies(y_batch)

      if forces
        f = extract_forces(y_batch)
        fconst = force_loss(model, x_batch, f)
      else
        fconst = 0f0
      end

      grad = Enzyme.gradient(
        set_runtime_activity(Reverse),
        (m, x, ee, ff) -> loss(m, x, ee, ff),
        model, Const(x_batch), Const(e), Const(fconst)
      )

      Flux.update!(opt, model, grad[1])
    end

    # === Loss evaluation ===
    # Per training uso un batch casuale, non tutto il dataset
    idx_sample = rand(1:size(x_train, 1), min(1024, size(x_train, 1)))
    e_t = extract_energies(y_train[idx_sample])
    if forces
      f_t = extract_forces(y_train[idx_sample])
      fconst_t = force_loss(model, x_train[idx_sample, :], f_t)
    else
      fconst_t = 0f0
    end
    loss_train[epoch] = loss(model, x_train[idx_sample, :], e_t, fconst_t)

    # Validazione su tutto il dataset
    e_v = extract_energies(y_val)
    if forces
      f_e = extract_forces(y_val)
      fconst_v = force_loss(model, x_val, f_e)
    else
      fconst_v = 0f0
    end
    loss_val[epoch] = loss(model, x_val, e_v, fconst_v)

    # === Checkpoint migliore modello ===
    if loss_val[epoch] < best_loss
      best_loss = loss_val[epoch]
      best_epoch = epoch
      θ, _ = Flux.destructure(model)
      best_params .= θ   # salva solo i pesi
      no_improve_count = 0
    else
      no_improve_count += 1
    end

    # === Learning rate decay ===
    if no_improve_count >= patience
      new_lr = max(current_lr * decay_factor, min_lr)
      current_lr = new_lr
      # aggiorna direttamente il campo eta nell'ottimizzatore Adam
      for st in opt
        if hasproperty(st, :eta)
          st.eta = new_lr
        end
      end
      no_improve_count = 0
      if verbose
        println("Reducing learning rate to $new_lr at epoch $epoch")
      end
      if current_lr <= min_lr
        println("Early stopping at epoch $epoch: learning rate reached min_lr")
        break
      end
    end
  end

  # Ricostruisci il best model dai pesi salvati
  best_model = re(best_params)

  if verbose
    println("Best Model Found at Epoch $best_epoch with Val Loss: $best_loss")
  end

  return model, best_model, loss_train, loss_val
end




