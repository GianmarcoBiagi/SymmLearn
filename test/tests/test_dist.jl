# ---- distance_layer tests ----

function test_distance_layer_single()
    """
    ## WHAT
    Compute distances for a single atomic configuration.

    ## GIVEN
    Three atoms with known coordinates.

    ## WHEN
    Calling `distance_layer` on Vector{AtomInput} without lattice.

    ## THEN
    Distances match Euclidean distances, distances are symmetric,
    and output is Vector{G1Input} with correct shape.
    """
    atoms = [
        AtomInput(1, Float32[0.0, 0.0, 0.0]),
        AtomInput(1, Float32[1.0, 0.0, 0.0]),
        AtomInput(1, Float32[0.0, 1.0, 0.0])
    ]

    # call single-configuration method
    out = distance_layer(atoms)
    @test length(out) == 3
    @test all(g -> g isa G1Input, out)

    # distances from atom 1
    d01 = out[1].distances[1,1]
    d02 = out[1].distances[1,2]
    @test isapprox(d01, 1.0; atol=1e-6)
    @test isapprox(d02, 1.0; atol=1e-6)

    # symmetry: distance from 2 to 1
    @test isapprox(out[2].distances[1,1], d01; atol=1e-6)
    # symmetry: distance from 3 to 1
    @test isapprox(out[3].distances[1,1], d02; atol=1e-6)
end

function test_distance_layer_batch()
    """
    ## WHAT
    Compute distances for a batch of atomic configurations.

    ## GIVEN
    Two configurations of two atoms each with known coordinates.

    ## WHEN
    Calling `distance_layer` on a 2×1 Matrix of Vector{AtomInput} with lattice.

    ## THEN
    Output is 2×1 Matrix of G1Input, distances are correct under PBC,
    and distances are symmetric and non-negative.
    """
    lattice = Matrix{Float32}(I, 3, 3)
    batch = [
        [AtomInput(1, Float32[0.0,0.0,0.0]), AtomInput(1, Float32[0.9,0.0,0.0])],
        [AtomInput(1, Float32[0.1,0.2,0.0]), AtomInput(1, Float32[0.8,0.2,0.0])]
    ]
    input_matrix = reshape(batch, 2, 1)  # batch shape: 2×1
    out = distance_layer(input_matrix; lattice=lattice)

    @test size(out) == (2,1)
    @test all(g -> g isa G1Input, vec(out))

    # distances in first configuration
    d0 = out[1,1][1].distances[1,1]
    d1 = out[1,1][2].distances[1,1]
    @test isapprox(d0, 0.1; atol=1e-6)
    @test isapprox(d1, d0; atol=1e-6)  # symmetry
    @test d0 ≥ 0f0 && d1 ≥ 0f0

    # distances in second configuration
    d2 = out[2,1][1].distances[1,1]
    d3 = out[2,1][2].distances[1,1]
    @test isapprox(d2, 0.3; atol=1e-6)
    @test isapprox(d3, d2; atol=1e-6)  # symmetry
    @test d2 ≥ 0f0 && d3 ≥ 0f0
end

function test_distance_layer_batch_multiple_atoms()
    """
    ## WHAT
    Compute distances for a batch with more than two atoms.

    ## GIVEN
    A batch of two configurations with three atoms each.

    ## WHEN
    Calling `distance_layer`.

    ## THEN
    All distance vectors have length N_atoms - 1, distances are positive,
    and symmetric.
    """
    batch = [
        [
            AtomInput(1, Float32[0.0,0.0,0.0]),
            AtomInput(1, Float32[1.0,0.0,0.0]),
            AtomInput(1, Float32[0.0,1.0,0.0])
        ],
        [
            AtomInput(1, Float32[0.1,0.2,0.0]),
            AtomInput(1, Float32[0.8,0.2,0.0]),
            AtomInput(1, Float32[0.5,0.5,0.0])
        ]
    ]
    input_matrix = reshape(batch, 2, 1)
    out = distance_layer(input_matrix)

    for conf in 1:2, i in 1:3
        g = out[conf,1][i]
        @test size(g.distances, 2) == 2  # N_atoms - 1
        @test all(g.distances .>= 0f0)
    end

    # check symmetry: dist(i,j) = dist(j,i)
    @test isapprox(out[1,1][1].distances[1,1], out[1,1][2].distances[1,1]; atol=1e-6)
    @test isapprox(out[2,1][1].distances[1,1], out[2,1][2].distances[1,1]; atol=1e-6)
end
