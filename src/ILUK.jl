module ILUK

using SparseArrays
using LinearAlgebra

# Core symbolic factorization
include("symbolic_iluk.jl")
include("fill_symbolic.jl")

# Numerical factorization implementations
include("numerical_iluk.jl")  # Standard implementation (corrected)

# Symmetric factorization
include("symmetric_iluk.jl")

# Robust factorization with adaptive shifts
include("numerical_iluk_lldl_inspired.jl")

# Main interface
include("ilu_k.jl")

# Public API exports
export symbolic_ilu_k
export ilu_k
export fill_symbolic!

# Symmetric matrix support
export symbolic_ilu_k_symmetric
export ilu_k_symmetric

# Advanced options (for users who need control)
export numerical_ilu_k!  # In-place version
export numerical_ilu_k_lldl!  # Robust version with shift options
export symmetric_ilu_k_lldl!  # Robust symmetric version

end # module ILUK
