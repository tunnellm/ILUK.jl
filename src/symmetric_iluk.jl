"""
    symbolic_ilu_k_symmetric(A::SparseMatrixCSC, k::Integer)

Compute symbolic ILU(k) factorization for symmetric matrices.
Returns L and D where A ≈ LDL^T.

For symmetric matrices, we only need to compute and store L (lower triangular)
and D (diagonal), since U = DL^T.
"""
function symbolic_ilu_k_symmetric(A::SparseMatrixCSC{T}, k::Integer) where {T}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    @assert k >= 0 "Fill level k must be non-negative"

    # For symmetric factorization, we only need L (with unit diagonal) and D
    # Start with lower triangular part of A
    L_rowval = Int[]
    L_colptr = zeros(Int, n + 1)
    L_colptr[1] = 1

    # Level matrix to track fill levels
    level_matrix = Dict{Tuple{Int,Int}, Int}()

    # Initialize with pattern of lower triangular part of A (including diagonal)
    for j = 1:n
        for idx in A.colptr[j]:A.colptr[j+1]-1
            i = A.rowval[idx]
            if i >= j  # Lower triangular part including diagonal
                level_matrix[(i, j)] = 0
            end
        end
    end

    # Process each column using symbolic factorization
    for j = 1:n
        # Get all entries in column j with their levels
        col_entries = [(i, level_matrix[(i, j)]) for (i, col) in keys(level_matrix) if col == j]
        sort!(col_entries)  # Sort by row index

        # Process each entry in column j
        for (i, level_ij) in col_entries
            if i <= j
                continue  # Skip upper triangular part
            end

            # For each k < j where L[i,k] exists
            for (row, k) in keys(level_matrix)
                if row == i && k < j
                    level_ik = level_matrix[(i, k)]

                    # Check if L[k,j] exists (which would be U[j,k] in non-symmetric)
                    # In symmetric case, we need L[j,k] if j > k, or L[k,j] if k > j
                    if j > k && haskey(level_matrix, (j, k))
                        level_kj = level_matrix[(j, k)]
                        new_level = max(level_ik, level_kj) + 1

                        if new_level <= k
                            # Add or update fill entry
                            if haskey(level_matrix, (i, j))
                                level_matrix[(i, j)] = min(level_matrix[(i, j)], new_level)
                            else
                                level_matrix[(i, j)] = new_level
                            end
                        end
                    end
                end
            end
        end
    end

    # Build L from level_matrix
    for j = 1:n
        col_indices = Int[]
        for (i, col) in keys(level_matrix)
            if col == j && i >= j  # Include diagonal
                push!(col_indices, i)
            end
        end
        sort!(col_indices)

        for i in col_indices
            push!(L_rowval, i)
        end
        L_colptr[j+1] = length(L_rowval) + 1
    end

    # Create sparse matrices
    L_nzval = ones(T, length(L_rowval))  # Will be filled with actual values later
    L = SparseMatrixCSC(n, n, L_colptr, L_rowval, L_nzval)

    # D is just a diagonal matrix - we'll store it as a vector
    D = ones(T, n)  # Will be filled with actual values later

    return L, D
end

"""
    fill_symbolic_symmetric!(A::SparseMatrixCSC, L::SparseMatrixCSC, D::Vector)

Fill L with values from the lower triangular part of A where patterns overlap.
D is initialized with diagonal values from A.
"""
function fill_symbolic_symmetric!(A::SparseMatrixCSC{T}, L::SparseMatrixCSC{T}, D::Vector{T}) where {T}
    n = size(L, 1)

    # Fill L with values from A where patterns overlap
    for j = 1:n
        for idx_L in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]

            # Find corresponding entry in A
            found = false
            for idx_A in A.colptr[j]:(A.colptr[j+1]-1)
                if A.rowval[idx_A] == i
                    if i == j
                        # Diagonal entry goes to D
                        D[j] = A.nzval[idx_A]
                        L.nzval[idx_L] = T(1)  # L has unit diagonal
                    else
                        L.nzval[idx_L] = A.nzval[idx_A]
                    end
                    found = true
                    break
                end
            end

            # Also check upper triangular part of A (since A is symmetric)
            if !found && i > j
                for idx_A in A.colptr[i]:(A.colptr[i+1]-1)
                    if A.rowval[idx_A] == j
                        L.nzval[idx_L] = A.nzval[idx_A]
                        found = true
                        break
                    end
                end
            end

            if !found
                if i == j
                    L.nzval[idx_L] = T(1)  # Unit diagonal
                else
                    L.nzval[idx_L] = T(0)  # Fill entry
                end
            end
        end
    end

    # Ensure diagonal of L is 1
    for j = 1:n
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            if L.rowval[idx] == j
                L.nzval[idx] = T(1)
                break
            end
        end
    end

    return nothing
end

"""
    numerical_ilu_k_symmetric!(L::SparseMatrixCSC, D::Vector, shift::Real=0.0)

Perform numerical ILU(k) factorization for symmetric matrices using LDL^T decomposition.
Works directly with sparse CSC format.

After fill_symbolic_symmetric!, L contains values from lower triangle of A.
This function performs the numerical factorization in-place.

Algorithm:
For j = 1 to n:
  1. Compute D[j] = A[j,j] - sum(L[j,k]^2 * D[k] for k < j)
  2. Compute L[i,j] = (A[i,j] - sum(L[i,k] * L[j,k] * D[k] for k < j)) / D[j]
"""
function numerical_ilu_k_symmetric!(
    L::SparseMatrixCSC{T},
    D::Vector{T},
    shift::Real=T(0)
) where {T}

    n = size(L, 1)

    # Save original matrix values before we start modifying
    L_orig = copy(L.nzval)
    D_orig = copy(D)

    # Apply diagonal shift
    if shift != 0
        D_orig .+= shift
    end

    # Work vector for accumulating row j of L
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

        # Step 1: Compute D[j]
        # D[j] = A[j,j] - sum(L[j,k]^2 * D[k] for k < j)
        sum_diag = T(0)
        for k in L_row_filled
            sum_diag += work_L[k]^2 * D[k]
        end

        D[j] = D_orig[j] - sum_diag

        if abs(D[j]) < eps(T) * 100
            error("Zero diagonal at position ($j,$j)")
        end

        # Step 2: Compute column j of L (below diagonal)
        # L[i,j] = (A[i,j] - sum(L[i,k] * L[j,k] * D[k] for k < j)) / D[j]
        for idx_L in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]

            if i == j
                # Diagonal of L is always 1
                L.nzval[idx_L] = T(1)
            elseif i > j
                # Get original A[i,j] value (stored in L_orig)
                a_ij = L_orig[idx_L]

                # Compute sum(L[i,k] * L[j,k] * D[k] for k < j)
                # We have L[j,k] in work_L
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

    return nothing
end

"""
    ilu_k_symmetric(A::SparseMatrixCSC, k::Integer; shift::Real=0.0)

Compute incomplete LU factorization with level k fill for symmetric matrices.
Returns L and D such that A ≈ LDL^T.

This is more efficient than general ILU(k) for symmetric matrices as it only
computes and stores the lower triangular factor L and diagonal D.
"""
function ilu_k_symmetric(A::SparseMatrixCSC{T}, k::Integer; shift::Real=T(0)) where {T}
    L, D = symbolic_ilu_k_symmetric(A, k)
    fill_symbolic_symmetric!(A, L, D)
    numerical_ilu_k_symmetric!(L, D, shift)
    return L, D
end