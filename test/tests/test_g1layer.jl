include("../../src/Machine_Learning/Model.jl")

# ---- G1Layer tests ----
function test_g1layer_initialization()
    """
    ## WHAT
    Confirm deterministic initialization with seed.

    ## GIVEN
    Fixed N_G1, cutoff, charge, and RNG seed.

    ## WHEN
    Constructing G1Layer.

    ## THEN
    W_eta and W_Fs are reproducible across calls.
    """
    seed = 42
    layer1 = G1Layer(5, 2.5f0, 1.0f0; seed=seed)
    layer2 = G1Layer(5, 2.5f0, 1.0f0; seed=seed)
    @test layer1.W_eta == layer2.W_eta
    @test layer1.W_Fs  == layer2.W_Fs
end

function test_g1layer_forward_pass()
    """
    ## WHAT
    Forward pass outputs correct dimensions and deterministic results.

    ## GIVEN
    Small input batch and initialized layer.

    ## WHEN
    Applying layer to input twice.

    ## THEN
    Outputs have correct shape and are identical for same input.
    """
    layer = G1Layer(3, 2.0f0, 1.0f0; seed=1)
    x = rand(Float32, 4, 6)
    output = layer(x)
    @test size(output) == (3, 4)

    layer2 = G1Layer(2, 1.0f0, 0.5f0; seed=123)
    x2 = Float32[0.2 0.5; 0.3 0.7]
    out1 = layer2(x2)
    out2 = layer2(x2)
    @test out1 == out2
end

function test_g1layer_positive_output()
    """
    ## WHAT
    Small inputs produce strictly positive outputs.

    ## GIVEN
    Positive distances < cutoff.

    ## WHEN
    Computing G1 output.

    ## THEN
    All output elements are >0.
    """
    layer = G1Layer(2, 2.0f0, 1.0f0; seed=1)
    x = Float32[0.1 0.5; 0.3 1.0]
    output = layer(x)
    @test all(output .> 0f0)
end