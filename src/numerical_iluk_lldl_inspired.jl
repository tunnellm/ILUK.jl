"""
    Numerical ILU(k) with LimitedLDL-Inspired Shift Strategy

Implements a shift strategy inspired by LimitedLDLFactorizations.jl:
- Sign-aware shifting for indefinite systems
- Exponential shift increase on failure
- Complete restart with shifted matrix
- For SPD systems, ensures positive pivots
"""

using LinearAlgebra
using SparseArrays

"""
    numerical_ilu_k_lldl!(L::SparseMatrixCSC, U::SparseMatrixCSC;
                         min_pivot::Real=1e-10,
                         α_min::Real=1e-10,
                         α_increase_factor::Real=10.0,
                         max_attempts::Int=3,
                         ensure_positive::Bool=false)

Perform ILU(k) factorization with LimitedLDL-inspired shift strategy.

Arguments:
- `min_pivot`: Minimum acceptable pivot magnitude (or minimum positive value for SPD)
- `α_min`: Initial shift value when first failure occurs
- `α_increase_factor`: Multiplicative factor for shift increase
- `max_attempts`: Maximum number of shift increases
- `ensure_positive`: If true, ensure all pivots are positive (for SPD systems)

Returns:
- NamedTuple with success status, final shift, and number of attempts
"""
function numerical_ilu_k_lldl!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T};
    min_pivot::Real=T(1e-10),
    α_min::Real=T(1e-10),
    α_increase_factor::Real=10.0,
    max_attempts::Int=3,
    ensure_positive::Bool=false
) where {T}

    n = size(L, 1)

    # Save original matrix values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    # Track which diagonals are positive/negative in original matrix
    diag_signs = zeros(Int8, n)
    for j = 1:n
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                diag_signs[j] = sign(U_orig[idx])
                break
            end
        end
    end

    # Shift parameters
    α = T(0)
    attempt = 0

    while attempt <= max_attempts
        # Reset to original values
        L.nzval .= L_orig
        U.nzval .= U_orig

        # Apply shift to diagonal elements
        if α > 0
            for j = 1:n
                for idx in U.colptr[j]:(U.colptr[j+1]-1)
                    if U.rowval[idx] == j
                        if ensure_positive
                            # For SPD systems, always add shift
                            U.nzval[idx] += α
                        else
                            # Sign-aware shifting for indefinite systems
                            if diag_signs[j] >= 0
                                U.nzval[idx] += α
                            else
                                U.nzval[idx] -= α
                            end
                        end
                        break
                    end
                end
            end
        end

        # Attempt factorization
        success, failed_column = attempt_factorization!(L, U, min_pivot, ensure_positive)

        if success
            return (success=true, shift=α, attempts=attempt, failed_column=0)
        end

        # Factorization failed, increase shift
        if attempt == 0
            α = α_min
            println("Factorization failed at column $failed_column, applying initial shift α = $α")
        else
            α *= α_increase_factor
            println("Factorization failed at column $failed_column, increasing shift to α = $α")
        end

        attempt += 1
    end

    return (success=false, shift=α, attempts=attempt, failed_column=-1)
end

"""
    attempt_factorization!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                          min_pivot::Real, ensure_positive::Bool)

Attempt ILU factorization with current matrix values.
Returns (success, failed_column).
"""
function attempt_factorization!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    min_pivot::Real,
    ensure_positive::Bool
) where {T}

    n = size(L, 1)

    # Work vectors
    work_L = zeros(T, n)
    work_U = zeros(T, n)
    L_row_filled = Int[]

    for j = 1:n
        # Gather row j of L
        empty!(L_row_filled)
        for k = 1:j-1
            for idx in L.colptr[k]:(L.colptr[k+1]-1)
                if L.rowval[idx] == j
                    work_L[k] = L.nzval[idx]
                    push!(L_row_filled, k)
                    break
                end
            end
        end

        # Find A[j,j] (already includes any shift)
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                a_jj = U.nzval[idx]
                u_jj_idx = idx
                break
            end
        end

        # Compute U[j,j]
        sum_diag = T(0)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j && work_L[k] != 0
                sum_diag += work_L[k] * U.nzval[idx]
            end
        end

        u_jj = a_jj - sum_diag

        # Check pivot
        if ensure_positive
            # For SPD systems, pivot must be positive
            if u_jj <= min_pivot
                return false, j
            end
        else
            # For general systems, pivot must be non-zero (or above threshold)
            if abs(u_jj) < min_pivot
                return false, j
            end
        end

        U.nzval[u_jj_idx] = u_jj

        # Gather column j of U for L computation
        fill!(work_U, 0)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j
                work_U[k] = U.nzval[idx]
            end
        end

        # Compute column j of L
        for idx_l in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_l]

            if i == j
                L.nzval[idx_l] = T(1)
            elseif i > j
                sum_l = T(0)
                for k = 1:j-1
                    if work_U[k] != 0
                        for idx in L.colptr[k]:(L.colptr[k+1]-1)
                            if L.rowval[idx] == i
                                sum_l += L.nzval[idx] * work_U[k]
                                break
                            end
                        end
                    end
                end

                L.nzval[idx_l] = (L.nzval[idx_l] - sum_l) / u_jj
            end
        end

        # Compute row j of U
        for col = j+1:n
            for idx_u in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx_u] == j
                    sum_u = T(0)
                    for idx in U.colptr[col]:(U.colptr[col+1]-1)
                        k = U.rowval[idx]
                        if k < j && work_L[k] != 0
                            sum_u += work_L[k] * U.nzval[idx]
                        end
                    end

                    U.nzval[idx_u] = U.nzval[idx_u] - sum_u
                    break
                end
            end
        end

        # Clear work vectors
        for k in L_row_filled
            work_L[k] = T(0)
        end
    end

    return true, 0
end

"""
    symmetric_ilu_k_lldl!(L::SparseMatrixCSC, D::Vector;
                         min_pivot::Real=1e-10,
                         α_min::Real=1e-10,
                         α_increase_factor::Real=10.0,
                         max_attempts::Int=3)

Symmetric ILU(k) with LimitedLDL-inspired shift strategy.
For symmetric matrices using LDL^T decomposition.
"""
function symmetric_ilu_k_lldl!(
    L::SparseMatrixCSC{T},
    D::Vector{T};
    min_pivot::Real=T(1e-10),
    α_min::Real=T(1e-10),
    α_increase_factor::Real=10.0,
    max_attempts::Int=3
) where {T}

    n = size(L, 1)

    # Save original values
    L_orig = copy(L.nzval)
    D_orig = copy(D)

    # Track original diagonal signs (for quasi-definite systems)
    diag_signs = sign.(D_orig)

    # Shift parameters
    α = T(0)
    attempt = 0

    while attempt <= max_attempts
        # Reset to original values
        L.nzval .= L_orig
        D .= D_orig

        # Apply shift to diagonal elements
        if α > 0
            for j = 1:n
                if diag_signs[j] >= 0
                    D[j] += α
                else
                    D[j] -= α  # Negative diagonals shift down
                end
            end
        end

        # Attempt factorization
        success, failed_column = attempt_symmetric_factorization!(L, D, min_pivot)

        if success
            return (success=true, shift=α, attempts=attempt,
                   failed_column=0, all_positive=all(D .> 0))
        end

        # Factorization failed, increase shift
        if attempt == 0
            α = α_min
            println("Symmetric factorization failed at column $failed_column, applying initial shift α = $α")
        else
            α *= α_increase_factor
            println("Symmetric factorization failed at column $failed_column, increasing shift to α = $α")
        end

        attempt += 1
    end

    return (success=false, shift=α, attempts=attempt,
           failed_column=-1, all_positive=false)
end

"""
    attempt_symmetric_factorization!(L::SparseMatrixCSC, D::Vector, min_pivot::Real)

Attempt symmetric LDL^T factorization with current values.
Returns (success, failed_column).
"""
function attempt_symmetric_factorization!(
    L::SparseMatrixCSC{T},
    D::Vector{T},
    min_pivot::Real
) where {T}

    n = size(L, 1)

    # Work vector
    work_L = zeros(T, n)
    L_row_filled = Int[]

    for j = 1:n
        # Gather row j of L
        empty!(L_row_filled)
        for k = 1:j-1
            for idx in L.colptr[k]:(L.colptr[k+1]-1)
                if L.rowval[idx] == j
                    work_L[k] = L.nzval[idx]
                    push!(L_row_filled, k)
                    break
                end
            end
        end

        # Compute D[j]
        sum_diag = T(0)
        for k in L_row_filled
            sum_diag += work_L[k]^2 * D[k]
        end

        d_jj = D[j] - sum_diag

        # Check pivot - for SPD we want positive
        if d_jj <= min_pivot
            return false, j
        end

        D[j] = d_jj

        # Compute column j of L
        for idx_l in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_l]

            if i == j
                L.nzval[idx_l] = T(1)
            elseif i > j
                sum_l = T(0)
                for k in L_row_filled
                    if work_L[k] != 0
                        for idx in L.colptr[k]:(L.colptr[k+1]-1)
                            if L.rowval[idx] == i
                                sum_l += L.nzval[idx] * work_L[k] * D[k]
                                break
                            end
                        end
                    end
                end

                L.nzval[idx_l] = (L.nzval[idx_l] - sum_l) / d_jj
            end
        end

        # Clear work vector
        for k in L_row_filled
            work_L[k] = T(0)
        end
    end

    return true, 0
end