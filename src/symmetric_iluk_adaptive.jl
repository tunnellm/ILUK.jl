"""
    Adaptive Symmetric ILU(k) with Positive Definiteness Recovery

For symmetric matrices using LDL^T factorization, we need to ensure D remains positive.
When D[j] becomes negative or too small, we apply a shift to maintain SPD property.
"""

using LinearAlgebra
using SparseArrays

"""
    numerical_ilu_k_symmetric_adaptive!(L::SparseMatrixCSC, D::Vector,
                                       min_pivot::Real=1e-10,
                                       ensure_positive::Bool=true)

Perform symmetric ILU(k) with adaptive shifting to maintain positive definiteness.

When D[j] ≤ min_pivot:
1. Determine required shift to make D[j] positive
2. Apply shift to remaining diagonal elements j:n
3. Continue factorization with shifted Schur complement

Returns factorization of A + σ*diag(0,...,0,1,...,1) where shift starts at failure point.
"""
function numerical_ilu_k_symmetric_adaptive!(
    L::SparseMatrixCSC{T},
    D::Vector{T},
    min_pivot::Real=T(1e-10),
    ensure_positive::Bool=true
) where {T}

    n = size(L, 1)

    # Save original matrix values
    L_orig = copy(L.nzval)
    D_orig = copy(D)

    # Track shifts applied
    diagonal_shifts = zeros(T, n)
    failed_columns = Int[]
    negative_pivots = Int[]

    # Work vector for row j of L
    work_L = zeros(T, n)
    L_row_filled = Int[]

    # Process each column j
    for j = 1:n

        # Gather row j of L into work vector (L[j,k] for k < j)
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

        # Compute D[j] = A[j,j] - sum(L[j,k]^2 * D[k] for k < j)
        a_jj = D_orig[j] + diagonal_shifts[j]  # Original diagonal plus any shift

        sum_diag = T(0)
        for k in L_row_filled
            sum_diag += work_L[k]^2 * D[k]
        end

        d_jj = a_jj - sum_diag

        # Check for negative or near-zero pivot
        needs_shift = false
        shift_amount = T(0)

        if ensure_positive && d_jj <= min_pivot
            needs_shift = true
            if d_jj <= 0
                # Negative pivot - need larger shift
                shift_amount = min_pivot - d_jj + T(1e-14)
                push!(negative_pivots, j)
                println("Column $j: D[$j] = $d_jj ≤ 0, applying shift = $shift_amount")
            else
                # Small positive pivot
                shift_amount = min_pivot - d_jj
                println("Column $j: D[$j] = $d_jj < min_pivot, applying shift = $shift_amount")
            end
            push!(failed_columns, j)
        elseif !ensure_positive && abs(d_jj) < min_pivot
            # For indefinite systems, just avoid zero
            needs_shift = true
            shift_amount = min_pivot * sign(d_jj) - d_jj
            if d_jj < 0
                shift_amount = abs(shift_amount)
            end
            push!(failed_columns, j)
            println("Column $j: |D[$j]| = $(abs(d_jj)) < min_pivot, applying shift = $shift_amount")
        end

        if needs_shift
            # Apply shift to diagonal of Schur complement (elements j:n)
            for k = j:n
                diagonal_shifts[k] += shift_amount
            end

            # Recompute D[j] with the shift
            d_jj = d_jj + shift_amount
        end

        D[j] = d_jj

        # Compute column j of L (below diagonal)
        # L[i,j] = (A[i,j] - sum(L[i,k] * L[j,k] * D[k] for k < j)) / D[j]
        for idx_L in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]

            if i == j
                # Diagonal of L is always 1
                L.nzval[idx_L] = T(1)
            elseif i > j
                # Get original A[i,j] value
                a_ij = L_orig[idx_L]

                # Compute sum(L[i,k] * L[j,k] * D[k] for k < j)
                sum_l = T(0)
                for k in L_row_filled
                    if work_L[k] != 0
                        # Find L[i,k] in column k
                        for idx in L.colptr[k]:(L.colptr[k+1]-1)
                            if L.rowval[idx] == i
                                sum_l += L.nzval[idx] * work_L[k] * D[k]
                                break
                            end
                        end
                    end
                end

                L.nzval[idx_L] = (a_ij - sum_l) / D[j]
            end
        end

        # Clear work vector
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
        negative_pivots = negative_pivots,
        all_positive = ensure_positive ? all(D .> 0) : true,
        min_D = minimum(D),
        max_D = maximum(D)
    )
end

"""
    symmetric_ilu_k_with_restart!(L::SparseMatrixCSC, D::Vector,
                                  min_pivot::Real=1e-10)

Symmetric ILU(k) with restart strategy for maintaining positive definiteness.

When D[j] becomes negative:
1. Apply shift to make it positive
2. Restart factorization from column j with shifted diagonal
"""
function symmetric_ilu_k_with_restart!(
    L::SparseMatrixCSC{T},
    D::Vector{T},
    min_pivot::Real=T(1e-10),
    max_restarts::Int=3
) where {T}

    n = size(L, 1)

    # Save original values
    L_orig = copy(L.nzval)
    D_orig = copy(D)

    shift_start_column = n + 1
    shift_amount = T(0)
    restart_count = 0

    # Work vector
    work_L = zeros(T, n)
    L_row_filled = Int[]

    j = 1  # Current column
    while j <= n && restart_count < max_restarts
        # Clear work
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

        # Compute D[j]
        a_jj = D_orig[j]
        if j >= shift_start_column
            a_jj += shift_amount
        end

        sum_diag = T(0)
        for k in L_row_filled
            sum_diag += work_L[k]^2 * D[k]
        end

        d_jj = a_jj - sum_diag

        # Check for failure
        if d_jj <= min_pivot
            println("Pivot failure at column $j: D[$j] = $d_jj")

            # Determine shift
            shift_amount = max(min_pivot - d_jj + T(1e-14), T(1e-6))
            println("Applying shift $shift_amount starting from column $j")

            if shift_start_column > n
                shift_start_column = j
            end

            # Restart from shift_start_column
            j = shift_start_column

            # Restore original values for columns >= j
            for col = j:n
                for idx in L.colptr[col]:(L.colptr[col+1]-1)
                    L.nzval[idx] = L_orig[idx]
                end
                D[col] = D_orig[col]
            end

            restart_count += 1
            continue
        end

        D[j] = d_jj

        # Compute column j of L
        for idx_L in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]

            if i == j
                L.nzval[idx_L] = T(1)
            elseif i > j
                a_ij = L_orig[idx_L]

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

                L.nzval[idx_L] = (a_ij - sum_l) / D[j]
            end
        end

        j += 1
    end

    return (
        success = (restart_count < max_restarts),
        shift_amount = shift_amount,
        shift_start_column = shift_start_column,
        restart_count = restart_count,
        all_positive = all(D .> 0),
        min_D = minimum(D),
        max_D = maximum(D)
    )
end

"""
    construct_spd_shifted_matrix(A::SparseMatrixCSC, shift_start::Int, shift_amount::Real)

Construct A + shift*E where E is zero except for 1's on diagonal from shift_start:n.
"""
function construct_spd_shifted_matrix(A::SparseMatrixCSC{T}, shift_start::Int,
                                     shift_amount::Real) where T
    n = size(A, 1)
    A_shifted = copy(A)

    if shift_start <= n && shift_amount != 0
        for j = shift_start:n
            # Find diagonal element
            for idx in A_shifted.colptr[j]:(A_shifted.colptr[j+1]-1)
                if A_shifted.rowval[idx] == j
                    A_shifted.nzval[idx] += shift_amount
                    break
                end
            end
        end
    end

    return A_shifted
end