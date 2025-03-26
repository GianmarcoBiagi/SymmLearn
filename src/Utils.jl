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

element_charge = Dict(
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
    distance_with_pbc(pos1, pos2, lattice_vectors)

Calculates the distance between two positions `pos1` and `pos2`, considering periodic boundary conditions (PBC).
This function is useful when working with periodic systems such as crystalline solids, where particles on one side of the system interact with those on the opposite side.

#### Parameters:
- `pos1::Vector{Float64}`: Position of the first point (3D vector).
- `pos2::Vector{Float64}`: Position of the second point (3D vector).
- `lattice_vectors::Matrix{Float64}`: Matrix of the lattice vectors of the system (3x3 matrix for a 3D system).

#### Returns:
- `Float64`: The distance between the two points considering periodic boundary conditions.

#### Example:
```julia
# Define two points (positions) in 3D space
pos1 = [1.0, 2.0, 3.0]
pos2 = [3.0, 4.0, 5.0]

# Define the lattice vectors of the system (3x3 matrix)
lattice_vectors = [
    10.0 0.0 0.0;
    0.0 10.0 0.0;
    0.0 0.0 10.0
]

# Calculate the distance between the two points with PBC
distance = distance_with_pbc(pos1, pos2, lattice_vectors)
println(distance)  # Output: The distance considering PBC
"""

function distance_with_pbc(pos1, pos2, lattice_vectors)
    # Calculate the difference between the two positions
    delta = pos2 .- pos1

    # Apply periodic boundary conditions (PBC)
    delta_pbc = delta .- lattice_vectors .* round.(delta ./ lattice_vectors)

    # Return the norm (Euclidean distance) of the adjusted difference
    return norm(delta_pbc)
end

"""
    fc(Rij::T, Rc::T) :: T where T

Computes a smooth cutoff function for distances between particles. The function returns 0 if the distance `Rij` exceeds the cutoff `Rc`. Otherwise, it computes the function value using a specific formula, which is based on the ratio between `Rij` and `Rc`.

# Arguments
- `Rij::T`: The distance between two particles, where `T` is a numeric type (e.g., `Float32` or `Float64`).
- `Rc::T`: The cutoff distance, beyond which the function returns 0.

# Returns
- A value of type `T`, which is the result of the function based on the given input.

# Example
```julia
Rij = 1.0
Rc = 2.5
result = fc(Rij, Rc)
println(result)  # This will output the result of the cutoff function for Rij=1.0 and Rc=2.5.
"""

function fc(Rij::T, Rc::T) :: T where T
    # If the distance Rij is beyond the cutoff Rc, return 0
    if Rij >= Rc
        return 0
    else 
        # Otherwise, compute the value based on the formula
        arg = 1 - 1 / (1 - (Rij / Rc) * (Rij / Rc))

        return exp(arg)
    end
end

