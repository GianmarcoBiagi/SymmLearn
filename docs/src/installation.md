# Installation



`SymmLearn.jl` is available for Julia v1.10.2 and later
releases, and can be installed with [Julia built-in package
manager](https://julialang.github.io/Pkg.jl/stable/).  In a Julia session, after
entering the package manager mode with `]`, run the command

```julia
pkg> add Measurements
```

or else you can install the package directly from its source repository using Julia's package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/SymmLearn.jl")
```
Alternatively, if you have cloned the repository locally, you can use:
```julia
using Pkg
Pkg.develop(path="/path/to/SymmLearn")
```
After installation, you can load the package with:
```julia
using SymmLearn
```