module ILUK

using SparseArrays
using LinearAlgebra

# Core symbolic factorization
include("symbolic_iluk.jl")        # General ILU(k) symbolic patterns
include("fill_symbolic.jl")        # Initialize L,U with A values

# Numerical factorization
include("ilu_k_basic.jl")          # Standard Crout-based ILU(k)
include("ilu_k_adaptive.jl")       # Adaptive shifting (LimitedLDL-inspired)

# Symmetric factorization
include("ldlt_k_symbolic.jl")      # Specialized symmetric symbolic patterns
include("ldlt_k_basic.jl")         # Standard LDL^T factorization

# Public API
include("ilu_k.jl")               # Main interface functions

# Clean API - only 4 functions
export symbolic_ilu_k  # General symbolic factorization
export ilu_k          # General numerical factorization with adaptive shifting
export symbolic_ldlt_k # Symmetric symbolic factorization
export ldlt_k         # Symmetric numerical factorization with adaptive shifting

end # module ILUK
