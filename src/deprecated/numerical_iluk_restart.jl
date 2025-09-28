"""
    Correct Approach for Handling Zero Pivots in ILU(k)

When we encounter a zero pivot at column j, we have two main options:

1. RESTART: Apply a global shift and restart from column 1
2. PARTIAL RESTART: Apply shift and restart from column j

Both produce a factorization of (A + σE) where E has 1's on some diagonal positions.
"""

using LinearAlgebra
using SparseArrays

"""
    numerical_ilu_k_with_partial_restart!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                                          min_pivot::Real=1e-10)

Perform ILU(k) with partial restart on pivot failure.

When a zero pivot is detected at column j:
1. Determine required shift σ
2. Restart factorization from column j with shifted diagonal
3. The result is a factorization of A + σ*diag(0,...,0,1,...,1)
                                              j-1 zeros

This is mathematically valid as a preconditioner for (A + σE).
"""
function numerical_ilu_k_with_partial_restart!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    min_pivot::Real=T(1e-10),
    max_restarts::Int=3
) where {T}

    n = size(L, 1)

    # Save original matrix values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    # Track where shifts are applied
    shift_start_column = n + 1  # Column where shift starts (n+1 means no shift)
    shift_amount = T(0)
    restart_count = 0

    # Work vectors
    work_L = zeros(T, n)
    work_U = zeros(T, n)
    L_row_filled = Int[]

    j = 1  # Current column
    while j <= n && restart_count < max_restarts
        # Clear work vectors
        empty!(L_row_filled)
        fill!(work_L, 0)

        # Gather row j of L
        for k = 1:j-1
            for idx in L.colptr[k]:(L.colptr[k+1]-1)
                if L.rowval[idx] == j
                    work_L[k] = L.nzval[idx]
                    push!(L_row_filled, k)
                    break
                end
            end
        end

        # Find A[j,j] value (with shift if j >= shift_start_column)
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                a_jj = U_orig[idx]
                if j >= shift_start_column
                    a_jj += shift_amount
                end
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

        # Check for pivot failure
        if abs(u_jj) < min_pivot
            println("Pivot failure at column $j: |U[$j,$j]| = $(abs(u_jj)) < $min_pivot")

            # Determine shift needed
            if u_jj >= 0
                shift_amount = min_pivot - u_jj + T(1e-14)
            else
                shift_amount = 2 * min_pivot  # Make it safely positive
            end

            println("Applying shift $shift_amount starting from column $j")

            # Mark where shift starts
            if shift_start_column > n
                shift_start_column = j
            end

            # RESTART from column shift_start_column with new shift
            # Reset factorization from this column onward
            j = shift_start_column

            # Restore original values for columns >= j
            for col = j:n
                # Restore L column
                for idx in L.colptr[col]:(L.colptr[col+1]-1)
                    L.nzval[idx] = L_orig[idx]
                end
                # Restore U column
                for idx in U.colptr[col]:(U.colptr[col+1]-1)
                    U.nzval[idx] = U_orig[idx]
                end
            end

            restart_count += 1
            continue
        end

        # Pivot is OK, continue factorization
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
                # Original A[i,j] value (no shift for off-diagonal)
                a_ij = L_orig[idx_l]

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

        # Compute row j of U
        for col = j+1:n
            for idx_u in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx_u] == j
                    # Original value (shifts only affect diagonal)
                    a_jcol = U_orig[idx_u]

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

        j += 1  # Move to next column
    end

    # Return information about the factorization
    return (
        success = (restart_count < max_restarts),
        shift_amount = shift_amount,
        shift_start_column = shift_start_column,
        restart_count = restart_count,
        factorizes_matrix = shift_start_column <= n ?
            "A + $(shift_amount)*diag([0 for i<$shift_start_column], [1 for i>=$shift_start_column])" :
            "A (no shift needed)"
    )
end

"""
    numerical_ilu_k_with_global_shift!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                                       initial_shift::Real=0.0)

Perform ILU(k) with automatic global shift adjustment.

If factorization fails, applies a global shift and restarts from scratch.
This produces a factorization of (A + σI).
"""
function numerical_ilu_k_with_global_shift!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    initial_shift::Real=T(0),
    min_pivot::Real=T(1e-10)
) where {T}

    n = size(L, 1)
    shift = initial_shift
    max_attempts = 5

    # Save original values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    for attempt = 1:max_attempts
        # Reset to original values
        L.nzval .= L_orig
        U.nzval .= U_orig

        # Apply current shift to diagonal
        if shift > 0
            for j = 1:n
                for idx in U.colptr[j]:(U.colptr[j+1]-1)
                    if U.rowval[idx] == j
                        U.nzval[idx] += shift
                        break
                    end
                end
            end
        end

        # Try factorization
        success = true
        try
            # Use standard factorization (we could use the one with shift already applied)
            include("numerical_iluk.jl")
            numerical_ilu_k!(L, U, 0.0)  # Shift already applied to U

            # Check all pivots
            for j = 1:n
                for idx in U.colptr[j]:(U.colptr[j+1]-1)
                    if U.rowval[idx] == j
                        if abs(U.nzval[idx]) < min_pivot
                            success = false
                            break
                        end
                    end
                end
                if !success
                    break
                end
            end

        catch e
            if occursin("Zero diagonal", string(e))
                success = false
            else
                rethrow(e)
            end
        end

        if success
            println("Factorization successful with global shift = $shift")
            return (success=true, shift=shift, attempts=attempt)
        else
            # Increase shift for next attempt
            if shift == 0
                shift = T(1e-6)
            else
                shift *= 10
            end
            println("Attempt $attempt failed, trying with shift = $shift")
        end
    end

    return (success=false, shift=shift, attempts=max_attempts)
end