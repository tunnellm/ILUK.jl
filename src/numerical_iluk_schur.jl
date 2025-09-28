"""
    numerical_ilu_k_schur_adaptive!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                                    min_pivot::Real=1e-10)

Perform numerical ILU(k) factorization with Schur complement shift recovery.

Key insight: When we encounter a zero pivot at column j, we've already factored
columns 1:j-1. The remaining work is to factor the Schur complement:
    S = A[j:n,j:n] - L[j:n,1:j-1] * U[1:j-1,j:n]

Instead of restarting, we can shift just the diagonal of S and continue.
This preserves all work done so far and only affects the unfactored portion.
"""
function numerical_ilu_k_schur_adaptive!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    min_pivot::Real=T(1e-10)
) where {T}

    n = size(L, 1)

    # Save original matrix values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    # Track shifts applied at each column
    diagonal_shifts = zeros(T, n)
    failed_columns = Int[]

    # Work vectors
    work_L = zeros(T, n)
    work_U = zeros(T, n)
    L_row_filled = Int[]

    # Process each column j using Crout method
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

        # Find original A[j,j] value with accumulated shifts
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                a_jj = U_orig[idx] + diagonal_shifts[j]
                u_jj_idx = idx
                break
            end
        end

        # Compute the updated diagonal element
        sum_diag = T(0)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j && work_L[k] != 0
                sum_diag += work_L[k] * U.nzval[idx]
            end
        end

        u_jj = a_jj - sum_diag

        # Check for near-zero pivot and apply Schur complement shift if needed
        if abs(u_jj) < min_pivot
            push!(failed_columns, j)

            # Compute shift needed to make pivot safe
            if u_jj >= 0
                shift_needed = min_pivot - u_jj + T(1e-14)
            else
                shift_needed = min_pivot - u_jj
            end

            println("Column $j: pivot = $u_jj, applying shift = $shift_needed")

            # Apply shift to diagonal of Schur complement
            # This affects diagonal elements j:n
            for k = j:n
                diagonal_shifts[k] += shift_needed
            end

            # Recompute the pivot with the shift
            u_jj = u_jj + shift_needed
        end

        U.nzval[u_jj_idx] = u_jj

        # Gather column j of U for L computation
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
                # Original A[i,j] value
                a_ij = L_orig[idx_l]

                # Add shift if this is a future diagonal element
                # (This would only happen if i==j, but we're in i>j branch)

                # Compute sum
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

                L.nzval[idx_l] = (a_ij - sum_l) / u_jj
            end
        end

        # Clear work_U
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j
                work_U[k] = T(0)
            end
        end

        # Compute row j of U
        for col = j+1:n
            for idx_u in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx_u] == j
                    # Get original value
                    a_jcol = U_orig[idx_u]

                    # Note: off-diagonal elements don't get shifted
                    # Only diagonal elements get shifted

                    # Compute sum
                    sum_u = T(0)
                    for idx in U.colptr[col]:(U.colptr[col+1]-1)
                        k = U.rowval[idx]
                        if k < j && work_L[k] != 0
                            sum_u += work_L[k] * U.nzval[idx]
                        end
                    end

                    U.nzval[idx_u] = a_jcol - sum_u
                    break
                end
            end
        end

        # Clear work_L
        for k in L_row_filled
            work_L[k] = T(0)
        end
    end

    # Return detailed information
    return (
        success = true,
        diagonal_shifts = diagonal_shifts,
        total_shift = maximum(diagonal_shifts),
        failed_columns = failed_columns,
        num_shifts = length(failed_columns)
    )
end

"""
    detect_pivot_failure(u_jj::T, j::Int, strategy::Symbol=:relative) where T

Detect if a pivot is too small and needs shifting.

Strategies:
- :absolute - pivot is bad if |u_jj| < threshold
- :relative - pivot is bad if |u_jj| < threshold * |A|
- :growth - pivot is bad if it would cause excessive growth in L/U factors
"""
function detect_pivot_failure(u_jj::T, j::Int, A_norm::T=T(1),
                             strategy::Symbol=:relative,
                             threshold::T=T(1e-10)) where T
    if strategy == :absolute
        return abs(u_jj) < threshold
    elseif strategy == :relative
        return abs(u_jj) < threshold * A_norm
    elseif strategy == :growth
        # Would cause entries > 1/threshold in factors
        return abs(u_jj) < threshold
    else
        error("Unknown pivot detection strategy: $strategy")
    end
end

"""
    compute_shift_amount(u_jj::T, strategy::Symbol=:minimal) where T

Compute how much shift to apply when a bad pivot is detected.

Strategies:
- :minimal - smallest shift to make pivot safe
- :conservative - shift to make pivot reasonably sized
- :aggressive - larger shift to avoid future failures
"""
function compute_shift_amount(u_jj::T, min_pivot::T=T(1e-10),
                             strategy::Symbol=:minimal) where T
    if strategy == :minimal
        # Just enough to make pivot = min_pivot
        if u_jj >= 0
            return min_pivot - u_jj + eps(T)
        else
            return min_pivot - u_jj
        end
    elseif strategy == :conservative
        # Make pivot at least 10 * min_pivot
        target = 10 * min_pivot
        if u_jj >= 0
            return max(T(0), target - u_jj)
        else
            return target - u_jj
        end
    elseif strategy == :aggressive
        # Make pivot at least 100 * min_pivot
        target = 100 * min_pivot
        return max(T(0), target - abs(u_jj))
    else
        error("Unknown shift strategy: $strategy")
    end
end