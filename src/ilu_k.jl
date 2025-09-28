"""
    ilu_k(A::SparseMatrixCSC, k::Integer; kwargs...)

Compute the incomplete LU factorization with fill level k using adaptive shifting.

# Arguments
- `A`: Input sparse matrix
- `k`: Fill level (k ≥ 0)

# Keyword Arguments
- `min_pivot::Real=1e-10`: Minimum acceptable pivot magnitude
- `α_min::Real=1e-10`: Initial shift value when failure occurs
- `α_increase_factor::Real=10.0`: Multiplicative factor for shift increase
- `max_attempts::Int=3`: Maximum number of shift increases

# Returns
- `L`: Lower triangular factor with unit diagonal
- `U`: Upper triangular factor

# Example
```julia
using SparseArrays
A = spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))
L, U = ilu_k(A, 1)
```

Uses LimitedLDL-inspired adaptive shift strategy for robustness.
"""
function ilu_k(A::SparseMatrixCSC{T}, k::Integer;
              min_pivot::Real=T(1e-10),
              α_min::Real=T(1e-10),
              α_increase_factor::Real=10.0,
              max_attempts::Int=3) where {T}

    # Step 1: Compute symbolic pattern
    L, U = symbolic_ilu_k(A, k)

    # Step 2: Fill with values from A
    fill_symbolic!(A, L, U)

    # Step 3: Perform numerical factorization with adaptive shifting
    result = numerical_ilu_k_lldl!(L, U;
                                   min_pivot=min_pivot,
                                   α_min=α_min,
                                   α_increase_factor=α_increase_factor,
                                   max_attempts=max_attempts,
                                   ensure_positive=false)
    if !result.success
        @warn "Adaptive factorization failed after $(result.attempts) attempts with final shift $(result.final_shift)"
    end

    return L, U
end

"""
    symbolic_ldlt_k(A::SparseMatrixCSC, k::Integer)

Compute symbolic LDL^T factorization pattern for symmetric matrices.
Only computes L (lower triangular) since U = L^T for symmetric factorization.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level (k ≥ 0)

# Returns
- `L`: Sparse lower triangular matrix with symbolic pattern

# Example
```julia
using SparseArrays
A = sparse(Symmetric(spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))))
L = symbolic_ldlt_k(A, 1)
```

More efficient than symbolic_ilu_k for symmetric matrices.
"""
function symbolic_ldlt_k(A::SparseMatrixCSC{T}, k::Integer) where {T}
    return symbolic_ilu_k_symmetric_new(A, k)
end

"""
    ldlt_k(A::SparseMatrixCSC, k::Integer; kwargs...)

Compute incomplete LDL^T factorization for symmetric matrices with adaptive shifting.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level (k ≥ 0)

# Keyword Arguments
- `min_pivot::Real=1e-10`: Minimum acceptable pivot (positive value for SPD systems)
- `α_min::Real=1e-10`: Initial shift value when failure occurs
- `α_increase_factor::Real=10.0`: Multiplicative factor for shift increase
- `max_attempts::Int=3`: Maximum number of shift increases
- `ensure_positive::Bool=false`: Force positive pivots (set true for SPD systems)

# Returns
- `L`: Lower triangular factor with unit diagonal
- `D`: Diagonal factor

# Example
```julia
using SparseArrays
A = sparse(Symmetric(spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))))

# General symmetric matrix
L, D = ldlt_k(A, 1)

# SPD matrix (ensures positive pivots)
L, D = ldlt_k(A, 1, ensure_positive=true)
```

More efficient than ilu_k for symmetric matrices. Uses LimitedLDL-inspired shifting.
"""
function ldlt_k(A::SparseMatrixCSC{T}, k::Integer;
               min_pivot::Real=T(1e-10),
               α_min::Real=T(1e-10),
               α_increase_factor::Real=10.0,
               max_attempts::Int=3,
               ensure_positive::Bool=false) where {T}

    # Step 1: Compute symbolic pattern (only L)
    L = symbolic_ldlt_k(A, k)

    # Step 2: Initialize D and fill with values from A
    n = size(A, 1)
    D = zeros(T, n)
    fill_symbolic_symmetric_new!(A, L, D)

    # Step 3: Perform numerical factorization with adaptive shifting
    result = symmetric_ilu_k_lldl!(L, D;
                                   min_pivot=min_pivot,
                                   α_min=α_min,
                                   α_increase_factor=α_increase_factor,
                                   max_attempts=max_attempts,
                                   ensure_positive=ensure_positive)
    if !result.success
        @warn "Adaptive symmetric factorization failed after $(result.attempts) attempts with final shift $(result.final_shift)"
    end

    return L, D
end