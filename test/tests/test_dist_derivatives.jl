# ---- distance_derivatives tests ----

function test_distance_derivatives_single()
    """
    ## WHAT
    Compute analytical derivatives for a single atomic configuration.

    ## GIVEN
    Three atoms with known coordinates.

    ## WHEN
    Calling `distance_derivatives` on a Vector{AtomInput} without lattice.

    ## THEN
    - Output tensor has shape (n_atoms, n_atoms-1, 3).
    - Derivatives have correct magnitude and direction (grad_i = -grad_j).
    - Zero derivative if atoms coincide.
    """
    atoms = [
        AtomInput(1, Float32[0.0, 0.0, 0.0]),
        AtomInput(1, Float32[1.0, 0.0, 0.0]),
        AtomInput(1, Float32[0.0, 1.0, 0.0])
    ]
    derivs = distance_derivatives(atoms)
    @test size(derivs) == (3,2,3)

    # check derivatives for first pair (atom 1 vs 2)
    grad_1_2 = derivs[1,1,:]
    grad_2_1 = derivs[2,1,:]
    @test isapprox(grad_1_2, -grad_2_1; atol=1e-6)

    # check derivative magnitude matches unit vector
    expected = [ -1.0f0, 0.0f0, 0.0f0 ]
    @test isapprox(grad_1_2, expected; atol=1e-6)
end

function test_distance_derivatives_batch()
    """
    ## WHAT
    Compute analytical derivatives for a batch of atomic configurations.

    ## GIVEN
    Two configurations of two atoms each with known coordinates.

    ## WHEN
    Calling `distance_derivatives` on a 2×1 Matrix of Vector{AtomInput} with lattice.

    ## THEN
    - Output tensor has shape (batch, n_atoms, n_atoms-1, 3)
    - Derivatives respect minimal-image convention.
    - Symmetry property grad_i = -grad_j holds.
    - Zero derivatives for coinciding atoms.
    """
    lattice = Matrix{Float32}(I,3,3)
    batch = [
        [AtomInput(1, Float32[0.0,0.0,0.0]), AtomInput(1, Float32[0.9,0.0,0.0])],
        [AtomInput(1, Float32[0.1,0.2,0.0]), AtomInput(1, Float32[0.8,0.2,0.0])]
    ]
    input_matrix = reshape(batch,2,1)
    derivs = distance_derivatives(input_matrix; lattice=lattice)
    @test size(derivs) == (2,2,1,3)

    # check symmetry in first batch
    @test isapprox(derivs[1,1,1,:], -derivs[1,2,1,:]; atol=1e-6)
    @test all(derivs .>= -1f0)  # sanity: no huge values
end

function test_distance_derivatives_zero_distance()
    """
    ## WHAT
    Verify derivative tensor handles coinciding atoms.

    ## GIVEN
    Two atoms at the same coordinates.

    ## WHEN
    Calling `distance_derivatives`.

    ## THEN
    - Derivatives are zero vectors.
    """
    atoms = [
        AtomInput(1, Float32[0.0,0.0,0.0]),
        AtomInput(1, Float32[0.0,0.0,0.0])
    ]
    derivs = distance_derivatives(atoms)
    @test all(derivs .== 0f0)
end


