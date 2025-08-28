using LinearAlgebra

"""
    element_charge

A dictionary that maps chemical element symbols (as String`s) to their respective atomic numbers (as Int`s).
This dictionary is used to retrieve the atomic number of an element by its symbol, which is typically needed when processing atomic datasets.

### Structure
The dictionary contains mappings for a range of chemical elements from Hydrogen (H) to Oganesson (Og), with each element symbol as the key and its atomic number as the value.

### Example Usage
```julia
# Example of retrieving the atomic number of Carbon (C)
carbon_atomic_number = element_charge["C"]
println("The atomic number of Carbon is: ", carbon_atomic_number)
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
    charge_to_element = Dict(v => k for (k, v) in element_charge)

Creates a dictionary that maps atomic charges to their corresponding element names.

# Arguments
- `element_charge::Dict{String, Float32}`: A dictionary where keys are element names (e.g., `"Cs"`, `"Pb"`, `"I"`) and values are their respective atomic charges.

# Returns
- `Dict{Float32, String}`: A dictionary where keys are atomic charges and values are element names.

# Example
```julia
element_charge = Dict("Cs" => 55.0, "Pb" => 85.0, "I" => 53.0)

charge_to_element = Dict(v => k for (k, v) in element_charge)

println(charge_to_element)  
# Output: Dict(55.0 => "Cs", 85.0 => "Pb", 53.0 => "I")
"""

charge_to_element = Dict(v => k for (k, v) in element_to_charge)

"""
    distance_atoms_pbc(atom1, atom2, lattice)

Compute the minimum-image distance between two atoms under periodic boundary conditions (PBC),
valid for any (possibly non-orthogonal) lattice.

# Arguments
- `atom1::Vector{Float32}`: Cartesian coordinates of the first atom
- `atom2::Vector{Float32}`: Cartesian coordinates of the second atom
- `lattice::Matrix{Float32}`: 3x3 lattice matrix (columns are lattice vectors)

# Returns
- `d::Float32`: Minimum distance considering PBC
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
    fc(Rij::T, Rc::T) :: T where T

Computes a smooth cutoff function for distances between particles. The function returns 0 if the distance `Rij` exceeds the cutoff `Rc`. 
If `Rij` is less than `Rc`, the function computes a smooth transition based on the ratio between `Rij` and `Rc`, using an exponential function. The cutoff is made smoother using a small tolerance (`ε`) to handle edge cases where the value becomes too small.

### Arguments
- `Rij::T`: The distance between two particles (numeric type `T`, e.g., `Float32` or `Float32`).
- `Rc::T`: The cutoff distance, beyond which the function returns 0.

### Returns
- A value of type `T`, which is the result of the cutoff function based on the given `Rij` and `Rc`.

### Behavior
- If `Rij >= Rc`, the function returns 0.
- If `Rij < Rc`, the function applies a smooth exponential transition, with the transition becoming increasingly sharp as `Rij` approaches `Rc`.

### Example
```julia
Rij = 1.0
Rc = 2.5
result = fc(Rij, Rc)
println(result)  # This will output the result of the cutoff function for Rij=1.0 and Rc=2.5.

"""

function fc(
    #Rij::T, Rc::T
    Rij, Rc
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
- `X::Vector{Sample}`: a collection of samples.

# Returns
- `energies_batch::RowVector{Float32}`: row vector of size `(1, n_batch)`,  
  where each column contains the energy of one sample.
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
    extract_forces(y; ndims::Int=3)

Extracts and reshapes atomic force vectors from a batch of data.

# Arguments
- `y`: A collection (e.g., vector or array) of objects, where each element 
  contains a field `forces` representing flattened atomic force components 
  as a 1D array of length `3 * n_atoms`.
- `ndims::Int=3`: Desired dimensionality of the returned tensor.  
  - `3`: Returns a 3D tensor `(n_batch, n_atoms, 3)`  
  - `2`: Returns a 2D matrix `(n_batch, n_atoms * 3)`  
  - `1`: Returns a 1D vector `(n_batch * n_atoms * 3)`  

# Returns
- If `ndims == 3`: `Array{Float32, 3}` of shape `(n_batch, n_atoms, 3)`,  
  where `forces[b, i, :]` contains the 3D force vector for atom `i` in batch `b`.  
- If `ndims == 2`: `Array{Float32, 2}` of shape `(n_batch, n_atoms*3)`.  
- If `ndims == 1`: `Array{Float32, 1}` of shape `(n_batch*n_atoms*3)`.  

# Errors
- Prints an error message if `ndims` is not 1, 2, or 3.  

# Example
```julia
# Suppose y is a vector of structs, each with a field `.forces` containing 
# forces like [fx1, fy1, fz1, fx2, fy2, fz2, ...].

forces3d = extract_forces(y; ndims=3)  # shape: (n_batch, n_atoms, 3)
forces2d = extract_forces(y; ndims=2)  # shape: (n_batch, n_atoms*3)
forces1d = extract_forces(y; ndims=1)  # shape: (n_batch*n_atoms*3)
"""

function extract_forces(y::Vector{Sample}; ndims::Int=3)
 
    n_batch = size(y , 1)
    n_atoms = div(length(y[1].forces) , 3)

    forces = zeros(Float32 , (n_batch , n_atoms , 3))

    for b in 1:n_batch

        for i in 1:n_atoms

            forces[b , i , :] = y[b].forces[1+3(i-1) : 3i]
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






