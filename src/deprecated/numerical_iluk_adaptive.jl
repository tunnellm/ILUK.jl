"""
    numerical_ilu_k_adaptive!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                              shift_strategy=:adaptive, min_pivot=1e-10)

Perform numerical ILU(k) factorization with adaptive shift recovery.

When a near-zero pivot is encountered, this function can:
1. Detect the failure (pivot < min_pivot)
2. Apply a shift to the remaining unfactored portion
3. Continue factorization without restarting

The key insight: At column j, we've computed L[:,1:j-1] and U[1:j-1,:].
The Schur complement S = A[j:n,j:n] - L[j:n,1:j-1] * U[1:j-1,j:n]
is what remains to be factored. We can shift just this part.
"""
function numerical_ilu_k_adaptive!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    shift_strategy::Symbol=:adaptive,
    min_pivot::T=T(1e-10)
) where {T}

    n = size(L, 1)

    # Save original matrix values
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    # Track cumulative shift applied
    total_shift = T(0)
    shifts_applied = Int[]  # Track which columns needed shifts

    # Work vectors for efficient accumulation
    work_L = zeros(T, n)
    work_U = zeros(T, n)
    L_row_filled = Int[]

    # Process each column j using Crout method
    for j = 1:n

        # Gather row j of L into work vector
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

        # Find original A[j,j] value
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                # Apply any accumulated shift to diagonal
                a_jj = U_orig[idx] + total_shift
                u_jj_idx = idx
                break
            end
        end

        # Compute sum(L[j,k] * U[k,j] for k=1:j-1)
        sum_diag = T(0)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j && work_L[k] != 0
                sum_diag += work_L[k] * U.nzval[idx]
            end
        end

        u_jj = a_jj - sum_diag

        # Check for near-zero pivot
        if abs(u_jj) < min_pivot
            # Determine shift needed
            if shift_strategy == :adaptive
                # Adaptive shift: make pivot at least min_pivot
                needed_shift = min_pivot - u_jj
                if u_jj < 0
                    needed_shift = min_pivot - u_jj  # Make it positive min_pivot
                else
                    needed_shift = min_pivot - u_jj
                end
            elseif shift_strategy == :fixed
                # Fixed shift
                needed_shift = T(1e-6)
            else
                error("Unknown shift strategy: $shift_strategy")
            end

            println("Warning: Near-zero pivot $u_jj at position ($j,$j)")
            println("Applying shift of $needed_shift to remaining matrix")

            # Update the shift
            total_shift += needed_shift
            push!(shifts_applied, j)

            # Recompute diagonal with shift
            u_jj = u_jj + needed_shift
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
                # Original A[i,j] with accumulated shift if on diagonal
                a_ij = L_orig[idx_l]

                # Compute sum(L[i,k] * U[k,j] for k=1:j-1)
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
                    # Original value with shift if diagonal
                    a_jcol = U_orig[idx_u]
                    if col == j  # This would be diagonal, but we handle it above
                        a_jcol += total_shift
                    end

                    # Compute sum(L[j,k] * U[k,col] for k=1:j-1)
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

    # Return information about the adaptive process
    return (success=true, total_shift=total_shift,
            shifts_at_columns=shifts_applied)
end

"""
    numerical_ilu_k_with_recovery!(L::SparseMatrixCSC, U::SparseMatrixCSC,
                                   initial_shift::Real=0.0,
                                   max_attempts::Int=5)

Attempt ILU(k) factorization with automatic recovery on failure.
First tries with initial_shift, then increases shift if needed.
"""
function numerical_ilu_k_with_recovery!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    initial_shift::Real=T(0),
    max_attempts::Int=5
) where {T}

    # Save initial state
    L_backup = copy(L)
    U_backup = copy(U)

    shift = initial_shift
    for attempt in 1:max_attempts
        # Reset to initial values
        L.nzval .= L_backup.nzval
        U.nzval .= U_backup.nzval

        println("\nAttempt $attempt with shift=$shift")

        try
            if shift == 0
                # First attempt without shift
                result = numerical_ilu_k_adaptive!(L, U, :adaptive, T(1e-10))
            else
                # Apply initial shift to diagonal
                for j = 1:size(U, 1)
                    for idx in U.colptr[j]:(U.colptr[j+1]-1)
                        if U.rowval[idx] == j
                            U.nzval[idx] += shift
                            break
                        end
                    end
                end
                result = numerical_ilu_k_adaptive!(L, U, :adaptive, T(1e-10))
            end

            if result.success
                println("Success! Total shift applied: $(shift + result.total_shift)")
                if !isempty(result.shifts_at_columns)
                    println("Adaptive shifts applied at columns: $(result.shifts_at_columns)")
                end
                return (success=true, total_shift=shift + result.total_shift)
            end

        catch e
            if occursin("Zero diagonal", string(e))
                # Increase shift for next attempt
                shift = shift == 0 ? T(1e-6) : shift * 10
                println("Failed with zero diagonal, increasing shift to $shift")
            else
                rethrow(e)
            end
        end
    end

    return (success=false, total_shift=shift)
end