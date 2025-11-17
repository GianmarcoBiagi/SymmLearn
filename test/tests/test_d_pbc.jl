include("../../src/Machine_Learning/Data_prep.jl")
include("../../src/Utils.jl")

# ---- d_pbc tests ----
function test_minimum_image_cubic()
    """
    ## WHAT
    Validate shortest periodic distance using minimum image convention.

    ## GIVEN
    Cubic lattice and two atoms separated by >0.5.

    ## WHEN
    Computing PBC distance with `d_pbc`.

    ## THEN
    Distance equals minimal wrapped value.
    """
    atom1 = [0.1, 0.1, 0.1]
    atom2 = [0.9, 0.1, 0.1]
    lattice = Matrix{Float32}(I, 3, 3)
    expected = 0.2f0
    result = d_pbc(atom1, atom2, lattice)
    @test result ≈ expected
end

function test_fractional_cartesian_consistency()
    """
    ## WHAT
    Verify identical distance from two coordinate conventions.

    ## GIVEN
    Non-orthogonal lattice; same points in fractional and Cartesian coordinates.

    ## WHEN
    Computing PBC distance in both modes.

    ## THEN
    Distances match exactly.
    """
    lattice = Float32[1 0.2 0; 0 1 0.3; 0 0 1]
    atom1_frac = [0.2, 0.3, 0.4]
    atom2_frac = [0.9, 0.3, 0.4]
    atom1_cart = lattice * atom1_frac
    atom2_cart = lattice * atom2_frac
    expected = d_pbc(atom1_cart, atom2_cart, lattice; coords=:cartesian)
    result = d_pbc(atom1_frac, atom2_frac, lattice; coords=:fractional)
    @test result ≈ expected
end

function test_return_image_vector()
    """
    ## WHAT
    Verify minimal-image vector and translation indices are correct.

    ## GIVEN
    Unit lattice; displacement >1 cell.

    ## WHEN
    Requesting full periodic image data.

    ## THEN
    Minimal vector and indices match theoretical values.
    """
    atom1 = [0.0, 0.0, 0.0]
    atom2 = [1.1, 0.0, 0.0]
    lattice = Matrix{Float32}(I, 3, 3)
    expected_d = 0.1f0
    expected_r = Float32[0.1, 0, 0]
    expected_n = [1, 0, 0]
    d, rvec, n = d_pbc(atom1, atom2, lattice; return_image=true)
    @test d ≈ expected_d
    @test rvec ≈ expected_r
    @test n == expected_n
end