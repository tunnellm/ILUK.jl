"""
    Public API for ILUK Package

Factorization structs and main interface functions.
"""

# =============================================================================
# FACTORIZATION STRUCTS
# =============================================================================

"""
    LDLFactorization{T}

Incomplete LDL^T factorization for symmetric matrices.
Represents A ≈ (I + L) * D * (I + L)' on the sparsity pattern.

Fields:
- `L`: Strictly lower triangular factor (SparseMatrixCSC)
- `D`: Diagonal vector
- `shift`: Diagonal shift applied (0 if none needed)
- `success`: Whether factorization succeeded
- `flops`: Total floating-point operations (assuming FMA)
- `graph_ops`: Total graph operations (index loads/stores)
"""
struct LDLFactorization{T}
    L::SparseMatrixCSC{T,Int}
    D::Vector{T}
    shift::T
    success::Bool
    flops::Int
    graph_ops::Int
end

"""
    LDUFactorization{T}

Incomplete LDU factorization for general matrices.
Represents A ≈ (I + L) * D * (I + U) on the sparsity pattern.

Fields:
- `L`: Strictly lower triangular factor (SparseMatrixCSC)
- `D`: Diagonal vector
- `U`: Strictly upper triangular factor (SparseMatrixCSC)
- `shift`: Diagonal shift applied (0 if none needed)
- `success`: Whether factorization succeeded
- `flops`: Total floating-point operations (assuming FMA)
- `graph_ops`: Total graph operations (index loads/stores)
"""
struct LDUFactorization{T}
    L::SparseMatrixCSC{T,Int}
    D::Vector{T}
    U::SparseMatrixCSC{T,Int}
    shift::T
    success::Bool
    flops::Int
    graph_ops::Int
end

# =============================================================================
# FACTORIZATION FUNCTIONS
# =============================================================================

"""
    ilu_k(A::SparseMatrixCSC, k::Integer; kwargs...) -> LDUFactorization

Compute the incomplete LU factorization with fill level k using adaptive shifting.

# Arguments
- `A`: Input sparse matrix
- `k`: Fill level (k ≥ 0)

# Keyword Arguments
- `min_pivot::Real=1e-10`: Minimum acceptable pivot magnitude
- `α::Real=0`: Initial diagonal shift
- `α_increase_factor::Real=10.0`: Multiplicative factor for shift increase
- `max_attempts::Int=3`: Maximum number of shift increases

# Returns
- `LDUFactorization` struct containing L, D, U factors

# Example
```julia
using SparseArrays, LinearAlgebra
A = spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))
F = ilu_k(A, 1)
x = F \\ b  # Solve using the factorization
```
"""
function ilu_k(A::SparseMatrixCSC{T}, k::Integer;
              min_pivot::Real=T(1e-10),
              α::Real=T(0),
              α_increase_factor::Real=10.0,
              max_attempts::Int=3) where {T}

    n = size(A, 1)

    # Step 1: Compute symbolic pattern
    L, U, graph_ops = symbolic_ilu_k(A, k)

    # Step 2: Initialize D and fill with values from A
    D = zeros(T, n)
    graph_ops += fill_symbolic!(A, L, U, D)

    # Step 3: Perform numerical factorization with adaptive shifting
    result = numeric_ilu_k!(L, U, D;
                                   min_pivot=min_pivot,
                                   α=α,
                                   α_increase_factor=α_increase_factor,
                                   max_attempts=max_attempts,
                                   graph_ops=graph_ops)
    if !result.success
        @warn "Adaptive factorization failed after $(result.attempts) attempts with final shift $(result.shift)"
    end

    return LDUFactorization(L, D, U, result.shift, result.success, result.flops, result.graph_ops)
end

"""
    ldlt_k(A::SparseMatrixCSC, k::Integer; kwargs...) -> LDLFactorization

Compute incomplete LDL^T factorization for symmetric matrices with adaptive shifting.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level (k ≥ 0)

# Keyword Arguments
- `min_pivot::Real=1e-10`: Minimum acceptable pivot magnitude
- `α::Real=0`: Initial diagonal shift
- `α_increase_factor::Real=10.0`: Multiplicative factor for shift increase
- `max_attempts::Int=3`: Maximum number of shift increases
- `ensure_positive::Bool=false`: Require positive pivots (fails on negative)

# Returns
- `LDLFactorization` struct containing L, D factors

# Example
```julia
using SparseArrays, LinearAlgebra
A = sparse(Symmetric(spdiagm(-1 => -ones(9), 0 => 2*ones(10), 1 => -ones(9))))
F = ldlt_k(A, 1)
x = F \\ b  # Solve using the factorization
```
"""
function ldlt_k(A::SparseMatrixCSC{T}, k::Integer;
               min_pivot::Real=T(1e-10),
               α::Real=T(0),
               α_increase_factor::Real=10.0,
               max_attempts::Int=3,
               ensure_positive::Bool=false) where {T}

    # Step 1: Compute symbolic pattern (only L)
    L, graph_ops = symbolic_cholesky(A, k)

    # Step 2: Initialize D and fill with values from A
    n = size(A, 1)
    D = zeros(T, n)
    graph_ops += fill_symbolic_symmetric!(A, L, D)

    # Step 3: Perform numerical factorization with adaptive shifting
    result = numeric_ldlt_k!(L, D;
                                   min_pivot=min_pivot,
                                   α=α,
                                   α_increase_factor=α_increase_factor,
                                   max_attempts=max_attempts,
                                   ensure_positive=ensure_positive,
                                   graph_ops=graph_ops)
    if !result.success
        @warn "Adaptive symmetric factorization failed after $(result.attempts) attempts with final shift $(result.shift)"
    end

    return LDLFactorization(L, D, result.shift, result.success, result.flops, result.graph_ops)
end

# =============================================================================
# SOLVE INTERFACE
# =============================================================================

import Base: \
import LinearAlgebra: ldiv!

# Backslash operator
function (\)(F::LDLFactorization{T}, b::AbstractVector{T}) where {T}
    ldl_solve(F.L, F.D, b)
end

function (\)(F::LDUFactorization{T}, b::AbstractVector{T}) where {T}
    ldu_solve(F.L, F.D, F.U, b)
end

function (\)(F::LDLFactorization{T}, b::AbstractVector) where {T}
    ldl_solve(F.L, F.D, convert(Vector{T}, b))
end

function (\)(F::LDUFactorization{T}, b::AbstractVector) where {T}
    ldu_solve(F.L, F.D, F.U, convert(Vector{T}, b))
end

# ldiv! - in-place left division

"""
    ldiv!(x, F::LDLFactorization, b)

Compute `F \\ b` and store the result in `x`.
"""
function ldiv!(x::AbstractVector{T}, F::LDLFactorization{T}, b::AbstractVector{T}) where {T}
    ldl_solve!(x, F.L, F.D, b)
end

"""
    ldiv!(x, F::LDUFactorization, b)

Compute `F \\ b` and store the result in `x`.
"""
function ldiv!(x::AbstractVector{T}, F::LDUFactorization{T}, b::AbstractVector{T}) where {T}
    ldu_solve!(x, F.L, F.D, F.U, b)
end

"""
    ldiv!(F::LDLFactorization, b)

Compute `F \\ b` in-place, storing the result in `b`.
"""
function ldiv!(F::LDLFactorization{T}, b::AbstractVector{T}) where {T}
    ldl_solve!(b, F.L, F.D, b)
end

"""
    ldiv!(F::LDUFactorization, b)

Compute `F \\ b` in-place, storing the result in `b`.
"""
function ldiv!(F::LDUFactorization{T}, b::AbstractVector{T}) where {T}
    ldu_solve!(b, F.L, F.D, F.U, b)
end

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    nnz(F::LDLFactorization)

Return the number of nonzeros in the factorization (L + D).
"""
function SparseArrays.nnz(F::LDLFactorization)
    nnz(F.L) + length(F.D)
end

"""
    nnz(F::LDUFactorization)

Return the number of nonzeros in the factorization (L + D + U).
"""
function SparseArrays.nnz(F::LDUFactorization)
    nnz(F.L) + length(F.D) + nnz(F.U)
end

"""
    size(F::LDLFactorization)

Return the size of the factorized matrix.
"""
Base.size(F::LDLFactorization) = size(F.L)
Base.size(F::LDLFactorization, i::Integer) = size(F.L, i)

"""
    size(F::LDUFactorization)

Return the size of the factorized matrix.
"""
Base.size(F::LDUFactorization) = size(F.L)
Base.size(F::LDUFactorization, i::Integer) = size(F.L, i)
