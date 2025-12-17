module ILUK

using SparseArrays
using LinearAlgebra

# All symbolic factorization routines (general + symmetric + initialization)
include("symbolic_routines.jl")

# All numerical factorization routines (general + symmetric with adaptive shifting)
include("numeric_routines.jl")

# Public API interface
include("api.jl")

# Factorization types
export LDLFactorization   # Symmetric factorization struct
export LDUFactorization   # General factorization struct

# Factorization API
export symbolic_ilu_k     # General symbolic factorization
export ilu_k              # General numerical factorization with adaptive shifting
export symbolic_cholesky  # Symmetric symbolic factorization
export ldlt_k             # Symmetric numerical factorization with adaptive shifting

# Solve API
export ldl_solve!, ldl_solve   # Symmetric solve
export ldu_solve!, ldu_solve   # General solve

end # module ILUK
