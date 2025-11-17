include("../../src/Machine_Learning/Data_prep.jl")
include("../../src/Utils.jl")


# ---- extract_* tests ----
function test_extract_energy_single()
    """
    ## WHAT
    Verify direct energy extraction from a sample.

    ## GIVEN
    Sample with known energy vector.

    ## WHEN
    Calling `extract_energies`.

    ## THEN
    Returned object equals stored energy.
    """
    x = Sample(1.5f0, rand(Float32, 2, 3))
    expected = x.energy
    result = extract_energies(x)
    @test result === expected
end

function test_extract_forces_single()
    """
    ## WHAT
    Verify direct forces extraction from a sample.

    ## GIVEN
    Sample with known forces matrix.

    ## WHEN
    Calling `extract_forces`.

    ## THEN
    Returned object equals stored forces.
    """
    F = rand(Float32, 3, 3)
    x = Sample(0f0, F)
    expected = F
    result = extract_forces(x)
    @test result === expected
end

function test_extract_forces_ndims()
    """
    ## WHAT
    Verify forces extraction shapes for ndims=3,2,1.

    ## GIVEN
    Two samples with known 2x3 forces matrices.

    ## WHEN
    Calling `extract_forces` with ndims=3,2,1.

    ## THEN
    Outputs match expected reshaped arrays.
    """
    # 3D
    s1 = Sample(0f0, Float32[1 2 3; 4 5 6])
    s2 = Sample(0f0, Float32[7 8 9; 10 11 12])
    Y = [s1, s2]
    expected3 = [1.0 4.0; 7.0 10.0;;; 2.0 5.0; 8.0 11.0;;; 3.0 6.0; 9.0 12.0]
    result3 = extract_forces(Y; ndims=3)
    @test result3 == expected3

    # 2D
    s1 = Sample(0f0, Float32[1 3 5; 2 4 6])
    s2 = Sample(0f0, Float32[7 9 11; 8 10 12])
    Y = [s1, s2]
    expected2 = Float32[1 2 3 4 5 6; 7 8 9 10 11 12]
    result2 = extract_forces(Y; ndims=2)
    @test result2 == expected2

    # 1D
    s1 = Sample(0f0, Float32[1 5 9; 3 7 11])
    s2 = Sample(0f0, Float32[2 6 10; 4 8 12])
    Y = [s1, s2]
    expected1 = Float32[1,2,3,4,5,6,7,8,9,10,11,12]
    result1 = extract_forces(Y; ndims=1)
    @test result1 == expected1
end