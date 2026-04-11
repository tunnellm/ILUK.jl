"""
    Numerical Factorization Routines

All numerical phase computations for ILU(k) and LDL^T(k) factorizations.
Includes robust adaptive shifting strategy inspired by LimitedLDLFactorizations.jl:
- Sign-aware shifting for indefinite systems
- Exponential shift increase on failure
- Complete restart with shifted matrix
- For SPD systems, ensures positive pivots
"""

using LinearAlgebra
using SparseArrays

"""
    numeric_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, D::Vector;
                         min_pivot::Real=1e-10,
                         α::Real=0,
                         α_increase_factor::Real=10.0,
                         max_attempts::Int=3)

Perform ILU(k) factorization with LimitedLDL-inspired shift strategy.

L is strictly lower triangular, U is strictly upper triangular, D holds diagonal.
Allows indefinite pivots. Only fails on near-zero pivots.

Arguments:
- `min_pivot`: Minimum acceptable pivot magnitude
- `α`: Initial diagonal shift (default 0)
- `α_increase_factor`: Multiplicative factor for shift increase on failure
- `max_attempts`: Maximum number of shift increases

Returns:
- NamedTuple with success status, final shift, and number of attempts
"""
function numeric_ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    D::Vector{T};
    min_pivot::Real=T(1e-10),
    α::Real=T(0),
    α_increase_factor::Real=10.0,
    max_attempts::Int=3
) where {T}

    n = size(L, 1)

    # Minimum shift floor (from LimitedLDLFactorizations)
    α_min = sqrt(eps(T))

    # Save original matrix values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)
    D_orig = copy(D)

    # Track which diagonals are positive/negative in original matrix
    diag_signs = Vector{Int8}(undef, n)
    @inbounds for j = 1:n
        diag_signs[j] = sign(D_orig[j])
    end

    # Pre-allocate work arrays (once, reused across retries)
    work = zeros(T, n)

    # Shift parameters
    α_current = T(α)
    attempt = 0

    while attempt <= max_attempts
        # Reset to original values
        L.nzval .= L_orig
        U.nzval .= U_orig
        D .= D_orig

        # Apply sign-aware shift to diagonal elements
        if α_current > 0
            @inbounds for j = 1:n
                if diag_signs[j] >= 0
                    D[j] += α_current
                else
                    D[j] -= α_current
                end
            end
        end

        # Attempt factorization (zero allocations inside)
        success, failed_column = attempt_factorization!(L, U, D, min_pivot, work)

        if success
            return (success=true, shift=α_current, attempts=attempt, failed_column=0)
        end

        # Factorization failed, increase shift
        α_current = (α_current == 0) ? α_min : α_increase_factor * α_current
        attempt += 1
    end

    return (success=false, shift=α_current, attempts=attempt, failed_column=-1)
end

"""
    attempt_factorization!(L, U, D, min_pivot, work)

Attempt ILU factorization with LDU form. Zero allocations in main loop.
L is strictly lower triangular, U is strictly upper triangular, D is diagonal.

Computes A ≈ (I + L) * diag(D) * (I + U) on the sparsity pattern.

For column j, the update formula is:
  work[i] -= L[i,k] * work[k]  for all i > k in column k of L
where work[k] = D[k] * U[k,j] after processing rows < k.
"""
function attempt_factorization!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    D::Vector{T},
    min_pivot::Real,
    work::Vector{T}
) where {T}

    n = size(L, 1)
    Lcolptr = L.colptr
    Lrowval = L.rowval
    Lval = L.nzval
    Ucolptr = U.colptr
    Urowval = U.rowval
    Uval = U.nzval

    @inbounds for j = 1:n
        # Scatter column j into work
        # U column j: rows < j (strictly upper), contains original A[k,j] values
        for p = Ucolptr[j]:(Ucolptr[j+1]-1)
            work[Urowval[p]] = Uval[p]
        end
        # L column j: rows > j (strictly lower), contains original A[i,j] values
        for p = Lcolptr[j]:(Lcolptr[j+1]-1)
            work[Lrowval[p]] = Lval[p]
        end
        # Diagonal: D[j] contains original A[j,j]
        work[j] = D[j]

        # Apply updates from columns k < j
        # After this loop: work[k] = D[k] * U[k,j], work[j] = D[j], work[i>j] = L[i,j] * D[j]
        for p = Ucolptr[j]:(Ucolptr[j+1]-1)
            k = Urowval[p]
            # work[k] has been updated by all m < k, so it equals D[k] * U[k,j]
            work_k = work[k]

            # Apply update FROM column k to all rows i > k in L column k
            for q = Lcolptr[k]:(Lcolptr[k+1]-1)
                i = Lrowval[q]
                work[i] -= Lval[q] * work_k
            end
        end

        # Get diagonal pivot
        d_jj = work[j]

        # Check pivot (allow indefinite, only fail on near-zero)
        if abs(d_jj) < min_pivot
            return false, j
        end

        # Store diagonal
        D[j] = d_jj

        # Gather U column j: U[k,j] = work[k] / D[k]
        for p = Ucolptr[j]:(Ucolptr[j+1]-1)
            k = Urowval[p]
            Uval[p] = work[k] / D[k]
            work[k] = zero(T)
        end

        # Gather L column j: L[i,j] = work[i] / D[j]
        for p = Lcolptr[j]:(Lcolptr[j+1]-1)
            row = Lrowval[p]
            Lval[p] = work[row] / d_jj
            work[row] = zero(T)
        end

        # Clear diagonal from work
        work[j] = zero(T)
    end

    return true, 0
end

"""
    numeric_ldlt_k!(L::SparseMatrixCSC, D::Vector;
                         min_pivot::Real=1e-10,
                         α::Real=0,
                         α_increase_factor::Real=10.0,
                         max_attempts::Int=3,
                         ensure_positive::Bool=false)

Symmetric LDL^T factorization with LimitedLDL-inspired shift strategy.
"""
function numeric_ldlt_k!(
    L::SparseMatrixCSC{T},
    D::Vector{T};
    min_pivot::Real=sqrt(eps(T))/2,
    α::Real=T(0),
    α_increase_factor::Real=10.0,
    max_attempts::Int=3,
    ensure_positive::Bool=false,
    gmw_beta::Real=T(0),
    lookahead::Bool=false
) where {T}

    n = size(L, 1)

    # Minimum shift floor (from LimitedLDLFactorizations)
    α_min = sqrt(eps(T))

    # Save original values
    L_orig = copy(L.nzval)
    D_orig = copy(D)

    # Track original diagonal signs (for quasi-definite systems)
    # If ensure_positive, treat all as positive for shifting
    diag_signs = Vector{Int8}(undef, n)
    @inbounds for j = 1:n
        diag_signs[j] = ensure_positive ? Int8(1) : sign(D_orig[j])
    end

    # Pre-allocate work arrays (once, reused across retries)
    work = zeros(T, n)
    list = zeros(Int, n)
    indf = zeros(Int, n)

    # Shift parameters
    α_current = T(α)
    attempt = 0

    while attempt <= max_attempts
        # Reset to original values
        L.nzval .= L_orig
        D .= D_orig

        # Apply sign-aware shift to diagonal elements
        if α_current > 0
            @inbounds for j = 1:n
                if diag_signs[j] >= 0
                    D[j] += α_current
                else
                    D[j] -= α_current
                end
            end
        end

        # Attempt factorization (zero allocations inside)
        success, failed_column = attempt_symmetric_factorization!(L, D, min_pivot, ensure_positive, work, list, indf;
                                                                   gmw_beta=gmw_beta, lookahead=lookahead)

        if success
            return (success=true, shift=α_current, attempts=attempt,
                   failed_column=0, all_positive=all(D .> 0))
        end

        # Factorization failed, increase shift
        α_current = (α_current == 0) ? α_min : α_increase_factor * α_current
        attempt += 1
    end

    return (success=false, shift=α_current, attempts=attempt,
           failed_column=-1, all_positive=false)
end

"""
    attempt_symmetric_factorization!(L, D, min_pivot, ensure_positive, work, list, indf)

Attempt symmetric LDL^T factorization. Zero allocations.
L is strictly lower triangular (no diagonal), D is the diagonal.
Uses linked-list approach from LimitedLDLFactorizations.

Work arrays (pre-allocated, size n):
- work: dense accumulator
- list: linked list of columns
- indf: current position in each column
"""
function attempt_symmetric_factorization!(
    L::SparseMatrixCSC{T},
    D::Vector{T},
    min_pivot::Real,
    ensure_positive::Bool,
    work::Vector{T},
    list::Vector{Int},
    indf::Vector{Int};
    gmw_beta::Real=T(0),
    lookahead::Bool=false
) where {T}

    n = size(L, 1)
    colptr = L.colptr
    rowval = L.rowval
    lval = L.nzval

    # Initialize
    @inbounds for j = 1:n
        list[j] = 0
        indf[j] = colptr[j]
    end

    @inbounds for col = 1:n
        # Scatter column into work
        for p = colptr[col]:(colptr[col+1]-1)
            work[rowval[p]] = lval[p]
        end

        # Apply updates from linked list
        d_col = D[col]
        k = list[col]
        while k != 0
            k_pos = indf[k]
            L_col_k = lval[k_pos]  # L[col, k]
            D_k = D[k]

            # Update diagonal
            d_col -= L_col_k * L_col_k * D_k

            # Update column: work[i] -= L[i,k] * D[k] * L[col,k]
            L_col_k_D_k = L_col_k * D_k
            for p = (k_pos+1):(colptr[k+1]-1)
                work[rowval[p]] -= lval[p] * L_col_k_D_k
            end

            # Advance and relink
            next_k = list[k]
            indf[k] += 1
            if indf[k] < colptr[k+1]
                next_row = rowval[indf[k]]
                list[k] = list[next_row]
                list[next_row] = k
            end
            k = next_k
        end

        # GMW(β) local modification: inflate pivot if too small relative to off-diagonals
        if gmw_beta > 0
            # Find max absolute off-diagonal in this column (from work array)
            lmax = zero(T)
            for p = colptr[col]:(colptr[col+1]-1)
                v = abs(work[rowval[p]])
                v > lmax && (lmax = v)
            end
            gmw_threshold = (lmax / T(gmw_beta))^2
            if d_col < gmw_threshold
                d_col = gmw_threshold
            end
        end

        # Check pivot (also catches NaN from LDM overflow)
        if !isfinite(d_col)
            return false, col
        end
        if ensure_positive
            if d_col <= min_pivot
                return false, col
            end
        else
            if abs(d_col) < min_pivot
                return false, col
            end
        end
        D[col] = d_col

        # Gather and normalize
        for p = colptr[col]:(colptr[col+1]-1)
            row = rowval[p]
            lval[p] = work[row] / d_col
            work[row] = zero(T)
        end

        # Look-ahead: after normalizing column col, check future diagonals
        # that will be affected by this column's entries. If any would go
        # below min_pivot, flag B1 breakdown early.
        if lookahead
            for p = colptr[col]:(colptr[col+1]-1)
                row = rowval[p]
                future_d = D[row] - lval[p] * lval[p] * d_col
                if ensure_positive
                    if future_d <= min_pivot
                        return false, col
                    end
                else
                    if abs(future_d) < min_pivot
                        return false, col
                    end
                end
            end
        end

        # Link this column to its first row
        if colptr[col] < colptr[col+1]
            first_row = rowval[colptr[col]]
            list[col] = list[first_row]
            list[first_row] = col
        end
    end

    return true, 0
end

# =============================================================================
# SOLVE ROUTINES
# =============================================================================

"""
    ldl_solve!(x, L, D, b)

Solve (I+L) * D * (I+L)' * x = b in-place.
L is strictly lower triangular (CSC), D is diagonal vector.
Result is stored in x. b is not modified.

Zero allocations.
"""
function ldl_solve!(
    x::Vector{T},
    L::SparseMatrixCSC{T},
    D::Vector{T},
    b::Vector{T}
) where {T}
    n = length(D)
    colptr = L.colptr
    rowval = L.rowval
    lval = L.nzval

    # Copy b to x
    @inbounds for i = 1:n
        x[i] = b[i]
    end

    # Forward solve: (I + L) * y = b
    # Column-oriented: for each column j, update rows i > j
    @inbounds for j = 1:n
        xj = x[j]
        for p = colptr[j]:(colptr[j+1]-1)
            i = rowval[p]
            x[i] -= lval[p] * xj
        end
    end

    # Diagonal solve: z = y / D
    @inbounds for i = 1:n
        x[i] /= D[i]
    end

    # Backward solve: (I + L)' * x = z
    # Column-oriented from right to left
    @inbounds for j = n:-1:1
        for p = colptr[j]:(colptr[j+1]-1)
            i = rowval[p]
            x[j] -= lval[p] * x[i]
        end
    end

    return x
end

"""
    ldu_solve!(x, L, D, U, b)

Solve (I+L) * D * (I+U) * x = b in-place.
L is strictly lower triangular, U is strictly upper triangular (CSC).
D is diagonal vector. Result is stored in x. b is not modified.

Zero allocations.
"""
function ldu_solve!(
    x::Vector{T},
    L::SparseMatrixCSC{T},
    D::Vector{T},
    U::SparseMatrixCSC{T},
    b::Vector{T}
) where {T}
    n = length(D)
    Lcolptr = L.colptr
    Lrowval = L.rowval
    Lval = L.nzval
    Ucolptr = U.colptr
    Urowval = U.rowval
    Uval = U.nzval

    # Copy b to x
    @inbounds for i = 1:n
        x[i] = b[i]
    end

    # Forward solve: (I + L) * y = b
    @inbounds for j = 1:n
        xj = x[j]
        for p = Lcolptr[j]:(Lcolptr[j+1]-1)
            i = Lrowval[p]
            x[i] -= Lval[p] * xj
        end
    end

    # Diagonal solve: z = y / D
    @inbounds for i = 1:n
        x[i] /= D[i]
    end

    # Backward solve: (I + U) * x = z
    # Column-oriented from right to left
    @inbounds for k = n:-1:1
        xk = x[k]
        for p = Ucolptr[k]:(Ucolptr[k+1]-1)
            i = Urowval[p]  # i < k since U is strictly upper
            x[i] -= Uval[p] * xk
        end
    end

    return x
end

"""
    ldl_solve(L, D, b) -> x

Allocating version of ldl_solve!.
"""
function ldl_solve(L::SparseMatrixCSC{T}, D::Vector{T}, b::Vector{T}) where {T}
    x = Vector{T}(undef, length(b))
    ldl_solve!(x, L, D, b)
    return x
end

"""
    ldu_solve(L, D, U, b) -> x

Allocating version of ldu_solve!.
"""
function ldu_solve(L::SparseMatrixCSC{T}, D::Vector{T}, U::SparseMatrixCSC{T}, b::Vector{T}) where {T}
    x = Vector{T}(undef, length(b))
    ldu_solve!(x, L, D, U, b)
    return x
end