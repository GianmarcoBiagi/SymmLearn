include("../../src/Utils.jl")


# ---- fc tests ----
function test_fc_cutoff_zero()
    """
    ## WHAT
    Confirm hard cutoff behavior.

    ## GIVEN
    Rij >= Rc.

    ## WHEN
    Evaluating `fc`.

    ## THEN
    Returned value is exactly 0.
    """
    expected = 0f0
    result = fc(1.0f0, 1.0f0)
    @test result == expected
end

function test_fc_smooth_interior()
    """
    ## WHAT
    Verify deterministic exponential inside cutoff.

    ## GIVEN
    Rij < Rc.

    ## WHEN
    Computing `fc`.

    ## THEN
    Output equals analytic expression.
    """
    Rij = 0.5f0
    Rc  = 1.0f0
    denom = 1 - (Rij/Rc)^2
    arg = 1 - 1/denom
    expected = exp(arg)
    result = fc(Rij, Rc)
    @test result ≈ expected
end

function test_fc_near_rc_guard()
    """
    ## WHAT
    Verify eps-based guard prevents divergence.

    ## GIVEN
    Rij approaching Rc.

    ## WHEN
    Denominator < eps(Float32).

    ## THEN
    fc returns 0.
    """
    Rc  = 1.0f0
    Rij = sqrt(1 - eps(Float32)/2) * Rc
    expected = 0f0
    result = fc(Rij, Rc)
    @test result == expected
end