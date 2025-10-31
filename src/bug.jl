using Flux
using Enzyme
using Random
using Statistics
using LinearAlgebra

#Here i define my "problematic" custom layer

struct G1Layer
    W_eta::Vector{Float32}
    W_Fs::Vector{Float32}
    cutoff::Float32
    charge::Float32
end

Flux.@layer G1Layer trainable = (W_eta,W_Fs,)


function G1Layer(N_G1::Int, cutoff::Float32, charge::Float32; seed::Union{Int,Nothing}=nothing)
    rng = seed === nothing ? Random.GLOBAL_RNG : MersenneTwister(seed)

    # Avoid Fs too close to zero to prevent huge contributions
    r_min = 0.1f0
    W_Fs = range(r_min, cutoff, length=N_G1) .+ 0.01f0 .* rand(rng, Float32, N_G1)

    # Compute average spacing and set eta proportional to 1/(spacing^2)
    delta = diff(W_Fs)
    avg_spacing = mean(delta)
    eta_base = 1.0f0 / (avg_spacing^2)
    W_eta = eta_base .* (0.8f0 .+ 0.4f0 .* rand(rng, Float32, N_G1))

    return G1Layer(W_eta, W_Fs, cutoff, charge)
end


function (layer::G1Layer)(x::AbstractMatrix{Float32})
    n_batch, n_neighbors = size(x)
    n_features = size(layer.W_eta, 1)

    output = zeros(Float32, n_features, n_batch)

    @inbounds for b in 1:n_batch
        for f in 1:n_features
            s = 0f0
            for n in 1:n_neighbors
                dx = x[b, n] - layer.W_Fs[f]
                s += fc(x[b, n], layer.cutoff) * exp(-layer.W_eta[f] * dx * dx)
            end
            output[f, b] = 0.1f0 * layer.charge * s
        end
    end

    return output
end

function fc(
    Rij::Float32, Rc::Float32
    ) 
    if Rij >= Rc
        return zero(Float32)
    end

    ε = eps(Float32)  
    denom = 1 - (Rij / Rc)^2
    if denom < ε
        return zero(Float32)
    end

    arg = 1 - 1 / denom
    return (exp(arg))
end

#This is the loss containing a gradient

function losss(m , x , y)

    e_L = mean((m(x) .- y) .^2)
    f_L = Enzyme.gradient(Reverse , (mm,xx) -> mm(xx)[1], Const(m) , x)[2]

    return e_L + norm(f_L)
end

x = rand(Float32 , 1 , 5)

y = sum(rand(Float32 , 5) .* x')

model = Chain(
            G1Layer(2 , 5.0f0 , 2.0f0),
            Dense(2 , 1))
println("The output of the model is: ",model(x))

o =  OptimiserChain(ClipNorm(1.0), Adam(0.1))

opt = Flux.setup(o, model)
grad = Enzyme.gradient(set_runtime_activity(Reverse) , (m , xx , yy)-> losss(m , xx ,yy) , model , Const(x) , Const(y))[1]

Flux.update!(opt, model, grad)