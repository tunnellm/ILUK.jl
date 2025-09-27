"""
    ilu_k(A::SparseMatrixCSC, k::Integer; shift=0.0)

Compute the incomplete LU factorization with fill level k.

# Arguments
- `A`: Input sparse matrix
- `k`: Fill level (k ≥ 0)
- `shift`: Optional diagonal shift for stability (default 0.0)

# Returns
- `L`: Lower triangular factor with unit diagonal
- `U`: Upper triangular factor

# Example
```julia
using SparseArrays
A = spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))
L, U = ilu_k(A, 1)
```

# Algorithm
Combines symbolic factorization (BFS-based pattern computation) with
numerical factorization (Crout method) following Hysom & Pothen (1999).
"""
function ilu_k(A::SparseMatrixCSC{T}, k::Integer; shift::Real=T(0)) where {T}
    # Step 1: Compute symbolic pattern
    L, U = symbolic_ilu_k(A, k)

    # Step 2: Fill with values from A
    fill_symbolic!(A, L, U)

    # Step 3: Perform numerical factorization
    numerical_ilu_k!(L, U, shift)

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