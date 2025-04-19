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
- `element_charge::Dict{String, Float64}`: A dictionary where keys are element names (e.g., `"Cs"`, `"Pb"`, `"I"`) and values are their respective atomic charges.

# Returns
- `Dict{Float64, String}`: A dictionary where keys are atomic charges and values are element names.

# Example
```julia
element_charge = Dict("Cs" => 55.0, "Pb" => 85.0, "I" => 53.0)

charge_to_element = Dict(v => k for (k, v) in element_charge)

println(charge_to_element)  
# Output: Dict(55.0 => "Cs", 85.0 => "Pb", 53.0 => "I")
"""

charge_to_element = Dict(v => k for (k, v) in element_to_charge)

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

function distance_with_pbc(pos1::Vector{Float32}, pos2::Vector{Float32}, lattice_vectors::Matrix{Float32})
    # Convert cartesian positions to fractional coordinates
    frac1 = lattice_vectors \ pos1
    frac2 = lattice_vectors \ pos2
    
    # Compute displacement in fractional coordinates and wrap into [-0.5, 0.5)
    delta_frac = frac2 .- frac1
    delta_frac = delta_frac .- round.(delta_frac)
    
    # Convert back to cartesian coordinates
    delta_cart = lattice_vectors * delta_frac
    
    # Return the Euclidean norm
    return norm(delta_cart)
end




"""
    fc(Rij::T, Rc::T) :: T where T

Computes a smooth cutoff function for distances between particles. The function returns 0 if the distance `Rij` exceeds the cutoff `Rc`. 
If `Rij` is less than `Rc`, the function computes a smooth transition based on the ratio between `Rij` and `Rc`, using an exponential function. The cutoff is made smoother using a small tolerance (`ε`) to handle edge cases where the value becomes too small.

### Arguments
- `Rij::T`: The distance between two particles (numeric type `T`, e.g., `Float32` or `Float64`).
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

function fc(Rij::T, Rc::T) :: T where T
    if Rij >= Rc
        return zero(T)
    end

    ε = eps(T)  
    denom = 1 - (Rij / Rc)^2
    if denom < ε
        return zero(T)
    end

    arg = 1 - 1 / denom
    return exp(arg)
end

