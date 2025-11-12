using LinearAlgebra

"""
    element_to_charge :: Dict{String, Int}

Dictionary mapping chemical element symbols (as `String`s) to their atomic numbers (as `Int`s).  
This is used to assign a charge or atomic number to an element for neural network input preprocessing.

# Example
```julia
# Retrieve atomic number of Carbon
carbon_atomic_number = element_to_charge["C"]
println(carbon_atomic_number)  # Output: 6
```
""" 
element_to_charge = Dict(
    "H" => 1,
    "He" => 2,
    "Li" => 3,
    "Be" => 4,
    "B" => 5,
    "C" => 6,
    "N" => 7,
    "O" => 8,
    "F" => 9,
    "Ne" => 10,
    "Na" => 11,
    "Mg" => 12,
    "Al" => 13,
    "Si" => 14,
    "P" => 15,
    "S" => 16,
    "Cl" => 17,
    "Ar" => 18,
    "K" => 19,
    "Ca" => 20,
    "Sc" => 21,
    "Ti" => 22,
    "V" => 23,
    "Cr" => 24,
    "Mn" => 25,
    "Fe" => 26,
    "Co" => 27,
    "Ni" => 28,
    "Cu" => 29,
    "Zn" => 30,
    "Ga" => 31,
    "Ge" => 32,
    "As" => 33,
    "Se" => 34,
    "Br" => 35,
    "Kr" => 36,
    "Rb" => 37,
    "Sr" => 38,
    "Y" => 39,
    "Zr" => 40,
    "Nb" => 41,
    "Mo" => 42,
    "Tc" => 43,
    "Ru" => 44,
    "Rh" => 45,
    "Pd" => 46,
    "Ag" => 47,
    "Cd" => 48,
    "In" => 49,
    "Sn" => 50,
    "Sb" => 51,
    "Te" => 52,
    "I" => 53,
    "Xe" => 54,
    "Cs" => 55,
    "Ba" => 56,
    "La" => 57,
    "Ce" => 58,
    "Pr" => 59,
    "Nd" => 60,
    "Pm" => 61,
    "Sm" => 62,
    "Eu" => 63,
    "Gd" => 64,
    "Tb" => 65,
    "Dy" => 66,
    "Ho" => 67,
    "Er" => 68,
    "Tm" => 69,
    "Yb" => 70,
    "Lu" => 71,
    "Hf" => 72,
    "Ta" => 73,
    "W" => 74,
    "Re" => 75,
    "Os" => 76,
    "Ir" => 77,
    "Pt" => 78,
    "Au" => 79,
    "Hg" => 80,
    "Tl" => 81,
    "Pb" => 82,
    "Bi" => 83,
    "Po" => 84,
    "At" => 85,
    "Rn" => 86,
    "Fr" => 87,
    "Ra" => 88,
    "Ac" => 89,
    "Th" => 90,
    "Pa" => 91,
    "U" => 92,
    "Np" => 93,
    "Pu" => 94,
    "Am" => 95,
    "Cm" => 96,
    "Bk" => 97,
    "Cf" => 98,
    "Es" => 99,
    "Fm" => 100,
    "Md" => 101,
    "No" => 102,
    "Lr" => 103,
    "Rf" => 104,
    "Db" => 105,
    "Sg" => 106,
    "Bh" => 107,
    "Hs" => 108,
    "Mt" => 109,
    "Ds" => 110,
    "Rg" => 111,
    "Cn" => 112,
    "Nh" => 113,
    "Fl" => 114,
    "Mc" => 115,
    "Lv" => 116,
    "Ts" => 117,
    "Og" => 118,
)
"""
    charge_to_element :: Dict{Int, String}

Inverse mapping of `element_to_charge`. Maps atomic numbers (charges) to their element symbols.

# Example
```julia
charge_to_element[6]  # Returns "C"
charge_to_element[79] # Returns "Au"
```
"""
charge_to_element = Dict(v => k for (k, v) in element_to_charge)

"""
    d_pbc(atom1, atom2, lattice; coords=:cartesian, return_image=false)

Compute the minimum-image distance between two atoms under periodic boundary conditions (PBC).

# Arguments
- `atom1::AbstractVector{<:Real}`: Coordinates of the first atom.
- `atom2::AbstractVector{<:Real}`: Coordinates of the second atom.
- `lattice::AbstractMatrix{<:Real}`: 3×3 lattice matrix where columns are lattice vectors.
- `coords::Symbol`: Either `:cartesian` (default) or `:fractional` to specify the input coordinate type.
- `return_image::Bool`: If true, also return the minimum-image vector in Cartesian coordinates and the integer lattice translation vector applied.

# Returns
- `d::Float32`: Minimum distance under PBC.
- Optionally `(d, rvec, n)` if `return_image=true`:
  - `rvec::Vector{Float32}`: Cartesian vector along the minimum-image direction.
  - `n::Vector{Int}`: Integer lattice translation indices applied to obtain the minimum image.

# Notes
- Works for both orthogonal and non-orthogonal lattices.
- Implements the minimum-image convention in fractional coordinates.
- Supports input coordinates in either Cartesian or fractional form.
"""
function d_pbc(atom1::AbstractVector{<:Real},
                        atom2::AbstractVector{<:Real},
                        lattice::AbstractMatrix{<:Real};
                        coords::Symbol = :cartesian,
                        return_image::Bool = false)

    # --- input checks and conversion to Float32 arrays ---
    @assert length(atom1) == 3 "atom1 must be length-3"
    @assert length(atom2) == 3 "atom2 must be length-3"
    @assert size(lattice) == (3,3) "lattice must be 3x3"

    r1 = Float32.(atom1)
    r2 = Float32.(atom2)
    L  = Float32.(lattice)

    # ensure lattice is invertible
    detL = det(L)
    @assert abs(detL) > 1e-12 "lattice matrix is (nearly) singular"

    if coords == :fractional
        # fractional -> compute fractional delta directly
        delta_frac = r2 .- r1
    elseif coords == :cartesian
        # Cartesian -> convert delta to fractional
        delta_cart = r2 .- r1
        delta_frac = inv(L) * delta_cart
    else
        error("coords must be :cartesian or :fractional")
    end

    # Apply minimum-image convention in fractional coordinates
    # n = round(delta_frac) gives the integer translation such that frac_min = delta_frac - n is in [-0.5,0.5)
    n = round.(Int, delta_frac)            # integer translation vector (same behavior as round(.))
    frac_min = delta_frac .- Float32.(n)   # fractional minimal-image vector

    # Convert back to Cartesian to get the actual vector
    rvec = L * frac_min
    d = norm(rvec)

    if return_image
        return d, rvec, n
    else
        return d
    end
end




"""
    fc(Rij, Rc)

Compute a smooth cutoff function for distances between particles.

# Arguments
- `Rij::Float32`: Distance between two particles.
- `Rc::Float32`: Cutoff distance. Returns 0 if `Rij >= Rc`.

# Returns
- `Float32`: Value of the smooth cutoff function. 

# Behavior
- Returns 0 if `Rij >= Rc`.
- For `Rij < Rc`, computes a smooth exponential decay using:
    fc(Rij) = exp(1 - 1 / (1 - (Rij / Rc)^2))
A small tolerance (`eps(Float32)`) is used to avoid numerical issues when `Rij` approaches `Rc`.


"""
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


"""
    extract_energies(x::Sample)

Return the energy of a single `Sample`.

# Arguments
- `x::Sample`: a single sample containing energy and forces.

# Returns
- `energy::AbstractVector`: the energy stored in the sample.
"""
function extract_energies(x::Sample)
    return x.energy
end


"""
    extract_forces(x::Sample)

Return the forces of a single `Sample`.

# Arguments
- `x::Sample`: a single sample containing energy and forces.

# Returns
- `forces::AbstractVector`: the forces stored in the sample.
"""
function extract_forces(x::Sample)
    return x.forces
end


"""
    extract_energies(X::Vector{Sample})

Return the energies of a batch of samples.

# Arguments
- `X::Vector{Sample}`: A collection of `Sample` objects, each containing an energy vector.

# Returns
- `energies_batch::Vector{Float32}`: 1D vector of length `n_batch`, where each element contains the first component of the energy of a sample.

# Notes
- The function extracts only the first element of each sample's `energy` field.
- Output is always of type `Float32`.
"""
function extract_energies(X::Vector{Sample})
    n_batch = size(X, 1)

    energies_batch = zeros(Float32, n_batch)
    for i in 1:n_batch
        energies_batch[i] = X[i].energy[1]
    end

    return energies_batch
end


"""
    extract_forces(y::Vector{Sample}; ndims::Int=3)

Extract and reshape atomic force vectors from a batch of samples.

# Arguments
- `y::Vector{Sample}`: A batch of `Sample` objects, each containing a `forces` field as a 2D array of shape `(n_atoms, 3)`.
- `ndims::Int=3`: Desired dimensionality of the returned array:
  - `3`: Returns a 3D array `(n_batch, n_atoms, 3)`  
  - `2`: Returns a 2D array `(n_batch, n_atoms*3)`  
  - `1`: Returns a 1D array `(n_batch*n_atoms*3)`

# Returns
- Array of `Float32` forces in the requested shape according to `ndims`.

# Behavior
- If `ndims == 3`, output shape is `(n_batch, n_atoms, 3)`.
- If `ndims == 2`, output shape is `(n_batch, n_atoms*3)`.
- If `ndims == 1`, output shape is `(n_batch*n_atoms*3)`.
- Prints an error message if `ndims` is not 1, 2, or 3.

# Notes
- Assumes all samples have the same number of atoms.
- Forces are extracted directly from the `forces` field of each sample.
"""
function extract_forces(y::Vector{Sample}; ndims::Int=3)
 
    n_batch = size(y , 1)
    n_atoms = div(length(y[1].forces) , 3)

    forces = zeros(Float32 , (n_batch , n_atoms , 3))

    for b in 1:n_batch

        for i in 1:n_atoms

            forces[b , i , :] = y[b].forces[i , 1:3]
        end
    end

    if ndims == 3

        return (forces)
    elseif ndims == 2

        return (reshape(forces , n_batch , n_atoms*3))
    elseif ndims == 1

        return (vcat(forces...))
    end
    println("Errore , output format can only be 1,2 or 3 dimensional please set ndims accordingly")
end






