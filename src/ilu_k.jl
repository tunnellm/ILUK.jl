"""
    ilu_k(A::SparseMatrixCSC, k::Integer; kwargs...)

Compute the incomplete LU factorization with fill level k.

# Arguments
- `A`: Input sparse matrix
- `k`: Fill level (k ≥ 0)

# Keyword Arguments
- `shift::Real=0.0`: Fixed diagonal shift for stability
- `adaptive::Bool=false`: Use adaptive shifting (robust for difficult matrices)
- `min_pivot::Real=1e-10`: Minimum acceptable pivot (for adaptive mode)

# Returns
- `L`: Lower triangular factor with unit diagonal
- `U`: Upper triangular factor

# Example
```julia
using SparseArrays
A = spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))

# Standard factorization
L, U = ilu_k(A, 1)

# Robust factorization with adaptive shifting
L, U = ilu_k(A, 1, adaptive=true)
```

# Algorithm
Combines symbolic factorization (BFS-based pattern computation) with
numerical factorization (Crout method) following Hysom & Pothen (1999).
When `adaptive=true`, uses LimitedLDL-inspired shift strategy for robustness.
"""
function ilu_k(A::SparseMatrixCSC{T}, k::Integer;
              shift::Real=T(0),
              adaptive::Bool=false,
              min_pivot::Real=T(1e-10)) where {T}

    # Step 1: Compute symbolic pattern
    L, U = symbolic_ilu_k(A, k)

    # Step 2: Fill with values from A
    fill_symbolic!(A, L, U)

    # Step 3: Perform numerical factorization
    if adaptive
        # Use robust version with adaptive shifting
        result = numerical_ilu_k_lldl!(L, U;
                                       min_pivot=min_pivot,
                                       ensure_positive=false)
        if !result.success
            @warn "Adaptive factorization failed after $(result.attempts) attempts"
        end
    else
        # Use standard version with fixed shift
        numerical_ilu_k!(L, U, shift)
    end

    return L, U
end

"""
    ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, A::SparseMatrixCSC, k::Integer; shift=0.0)

In-place version of ilu_k that reuses pre-allocated L and U matrices.

Useful when solving multiple systems with the same sparsity pattern.
"""
function ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    A::SparseMatrixCSC{T},
    k::Integer;
    shift::Real=T(0)
) where {T}

    # Fill with values from A
    fill_symbolic!(A, L, U)

    # Perform numerical factorization
    numerical_ilu_k!(L, U, shift)

    return L, U
end