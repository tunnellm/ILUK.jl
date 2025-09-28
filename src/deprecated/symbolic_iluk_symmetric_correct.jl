"""
    symbolic_ilu_k_symmetric_correct(A::SparseMatrixCSC, k::Integer)

Compute symbolic ILU(k) pattern for symmetric matrices.
This version correctly computes the same pattern as the general version
but only returns L (since U = L^T for symmetric factorization).

# Algorithm
For symmetric matrices, we compute the full ILU(k) pattern but only
return the lower triangular part, as U would be the transpose of L.
"""
function symbolic_ilu_k_symmetric_correct(A::SparseMatrixCSC{T}, k::Integer) where {T}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    @assert k >= 0 "Fill level k must be non-negative"

    # For a symmetric matrix, we can use the general symbolic factorization
    # and just extract L
    L_gen, U_gen = symbolic_ilu_k(A, k)

    # For symmetric matrices, the pattern of L from general factorization
    # is what we want
    return L_gen
end

"""
    symbolic_ilu_k_symmetric_efficient(A::SparseMatrixCSC, k::Integer)

Compute symbolic ILU(k) pattern for symmetric matrices more efficiently.
Only computes the lower triangular pattern.

This is the corrected version that properly handles the level-of-fill computation.
"""
function symbolic_ilu_k_symmetric_efficient(A::SparseMatrixCSC{T}, k::Integer) where {T}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    @assert k >= 0 "Fill level k must be non-negative"

    # Level matrices for L and U (we need both to compute fill correctly)
    L_levels = Dict{Tuple{Int,Int}, Int}()
    U_levels = Dict{Tuple{Int,Int}, Int}()

    # Initialize with pattern of A
    # For symmetric A, we initialize both L and U patterns
    for j = 1:n
        for idx in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[idx]
            if i >= j  # Lower or diagonal
                L_levels[(i, j)] = 0
            end
            if i <= j  # Upper or diagonal
                U_levels[(i, j)] = 0
            end
        end
    end

    # Also add transposed entries for symmetry
    for j = 1:n
        for idx in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[idx]
            if i > j  # Transpose of lower entries go to upper
                if !haskey(U_levels, (j, i))
                    U_levels[(j, i)] = 0
                end
            elseif i < j  # Transpose of upper entries go to lower
                if !haskey(L_levels, (j, i))
                    L_levels[(j, i)] = 0
                end
            end
        end
    end

    # Process columns in order (like general ILU(k))
    for j = 1:n
        # Update L and U based on fill rules

        # For each i > j (column j of L)
        rows_in_L_col_j = Int[]
        for (row, col) in keys(L_levels)
            if col == j && row > j
                push!(rows_in_L_col_j, row)
            end
        end

        for i in rows_in_L_col_j
            # Check for fill: L[i,j] gets fill from L[i,k] * U[k,j] for k < j
            for k = 1:j-1
                if haskey(L_levels, (i, k)) && haskey(U_levels, (k, j))
                    level_ik = L_levels[(i, k)]
                    level_kj = U_levels[(k, j)]
                    new_level = max(level_ik, level_kj) + 1

                    if new_level <= k
                        if haskey(L_levels, (i, j))
                            L_levels[(i, j)] = min(L_levels[(i, j)], new_level)
                        else
                            L_levels[(i, j)] = new_level
                        end
                    end
                end
            end
        end

        # For each i < j (row j of U)
        cols_in_U_row_j = Int[]
        for (row, col) in keys(U_levels)
            if row == j && col > j
                push!(cols_in_U_row_j, col)
            end
        end

        for i in cols_in_U_row_j
            # Check for fill: U[j,i] gets fill from L[j,k] * U[k,i] for k < j
            for k = 1:j-1
                if haskey(L_levels, (j, k)) && haskey(U_levels, (k, i))
                    level_jk = L_levels[(j, k)]
                    level_ki = U_levels[(k, i)]
                    new_level = max(level_jk, level_ki) + 1

                    if new_level <= k
                        if haskey(U_levels, (j, i))
                            U_levels[(j, i)] = min(U_levels[(j, i)], new_level)
                        else
                            U_levels[(j, i)] = new_level
                        end
                    end
                end
            end
        end
    end

    # Build L from L_levels (including diagonal)
    L_rowval = Int[]
    L_colptr = zeros(Int, n + 1)
    L_colptr[1] = 1

    for j = 1:n
        col_indices = Int[]

        # Add diagonal if it exists
        if haskey(L_levels, (j, j)) || haskey(U_levels, (j, j))
            push!(col_indices, j)
        end

        # Add lower triangular entries
        for (i, col) in keys(L_levels)
            if col == j && i > j
                push!(col_indices, i)
            end
        end

        sort!(col_indices)

        for i in col_indices
            push!(L_rowval, i)
        end
        L_colptr[j+1] = length(L_rowval) + 1
    end

    # Create sparse matrix
    L_nzval = ones(T, length(L_rowval))
    L = SparseMatrixCSC(n, n, L_colptr, L_rowval, L_nzval)

    return L
end