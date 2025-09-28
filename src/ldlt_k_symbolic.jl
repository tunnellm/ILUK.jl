"""
    symbolic_ilu_k_symmetric_new(A::SparseMatrixCSC, k::Integer)

Compute symbolic ILU(k) pattern for symmetric matrices.
Only computes L (lower triangular) since U = DL^T for symmetric factorization.

This is more efficient than computing both L and U when working with symmetric matrices.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level (0 ≤ k)

# Returns
- `L`: Sparse lower triangular matrix with symbolic pattern

# Algorithm
Adapted BFS approach for symmetric matrices where U = L^T.
For each row i, we propagate fill through the transpose relationship.
"""
function symbolic_ilu_k_symmetric_new(A::SparseMatrixCSC{T}, k::Integer) where {T}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    @assert k >= 0 "Fill level k must be non-negative"

    if k == 0
        # For ILU(0), use lower triangular pattern of A with guaranteed diagonal
        L = tril(A)

        # Add missing diagonal entries
        for j = 1:n
            has_diag = false
            for idx in L.colptr[j]:(L.colptr[j+1]-1)
                if L.rowval[idx] == j
                    has_diag = true
                    break
                end
            end
            if !has_diag
                L = L + sparse([j], [j], [T(1)], n, n)
            end
        end

        L.nzval .= 1
        return L
    end

    # Track L pattern for each column (sparse representation)
    L_col_idx = [Int[] for _ in 1:n]
    L_col_lvl = [Int[] for _ in 1:n]

    # Work arrays
    level = fill(k + 1, n)  # Level/distance array
    queue = Vector{Int}(undef, n)  # BFS queue
    touched = Vector{Int}(undef, n)  # Track modified entries

    for i in 1:n
        # Reset work arrays
        touched_t = 0

        # Initialize with pattern from A (symmetric)
        for j_idx in A.colptr[i]:(A.colptr[i+1]-1)
            j = A.rowval[j_idx]
            level[j] = 0
            touched_t += 1
            touched[touched_t] = j
        end

        # Also check transpose entries (column i of A)
        for j in 1:n
            if j != i
                for idx in A.colptr[j]:(A.colptr[j+1]-1)
                    if A.rowval[idx] == i
                        if level[j] > 0  # Not already added
                            level[j] = 0
                            touched_t += 1
                            touched[touched_t] = j
                        end
                        break
                    end
                end
            end
        end

        # Initialize BFS queue with L-part entries (j < i)
        head = 1
        tail = 0
        for tt in 1:touched_t
            j = touched[tt]
            if j < i  # L-part
                tail += 1
                queue[tail] = j
            end
        end

        # BFS propagation
        # For symmetric case, U[j,:] is L[:,j]^T
        while head <= tail
            j = queue[head]
            head += 1
            lvlj = level[j]

            # Propagate through U[j,:] which is L[:,j]^T
            # So we need entries in column j of L with row > j
            for (w_idx, (w, wlvl)) in enumerate(zip(L_col_idx[j], L_col_lvl[j]))
                if w > j  # This is U[j,w] = L[w,j]
                    new_lvl = lvlj + wlvl + 1
                    if new_lvl <= k && new_lvl < level[w]
                        if level[w] == k + 1  # First time touching w
                            touched_t += 1
                            touched[touched_t] = w
                        end
                        level[w] = new_lvl
                        if w < i  # Add to queue if in L-part
                            tail += 1
                            queue[tail] = w
                        end
                    end
                end
            end
        end

        # Sort and store results for row i
        L_row = Int[]
        L_row_lvl = Int[]

        for tt in 1:touched_t
            j = touched[tt]
            if level[j] <= k
                if j <= i  # Include diagonal
                    push!(L_row, j)
                    push!(L_row_lvl, level[j])
                end
            end
            level[j] = k + 1  # Reset for next iteration
        end

        # Sort and store in column format
        perm = sortperm(L_row)
        L_row = L_row[perm]
        L_row_lvl = L_row_lvl[perm]

        # Add to column structures
        for (j, lvl) in zip(L_row, L_row_lvl)
            push!(L_col_idx[j], i)
            push!(L_col_lvl[j], lvl)
        end
    end

    # Build CSC structure
    L_colptr = zeros(Int, n + 1)
    L_colptr[1] = 1
    L_rowval = Int[]

    for j in 1:n
        # Ensure diagonal is always included
        has_diag = false
        for i in L_col_idx[j]
            if i == j
                has_diag = true
                break
            end
        end
        if !has_diag
            # Insert diagonal in sorted position
            inserted = false
            for (idx, i) in enumerate(L_col_idx[j])
                if i > j
                    insert!(L_col_idx[j], idx, j)
                    inserted = true
                    break
                end
            end
            if !inserted
                push!(L_col_idx[j], j)
            end
        end

        for i in L_col_idx[j]
            push!(L_rowval, i)
        end
        L_colptr[j+1] = L_colptr[j] + length(L_col_idx[j])
    end

    # Create sparse matrix with ones
    L_nzval = ones(T, length(L_rowval))
    L = SparseMatrixCSC(n, n, L_colptr, L_rowval, L_nzval)

    return L
end

"""
    fill_symbolic_symmetric_new!(A::SparseMatrixCSC, L::SparseMatrixCSC, D::Vector)

Fill L with values from symmetric matrix A where patterns overlap.
Initialize diagonal vector D with diagonal values from A.

# Arguments
- `A`: Symmetric sparse matrix
- `L`: Lower triangular symbolic pattern (modified in-place)
- `D`: Diagonal vector (modified in-place)
"""
function fill_symbolic_symmetric_new!(A::SparseMatrixCSC{T}, L::SparseMatrixCSC{T}, D::Vector{T}) where {T}
    n = size(L, 1)

    # Initialize D with diagonal of A
    for j = 1:n
        D[j] = T(0)  # Default value
        # Look for A[j,j]
        for idx in A.colptr[j]:(A.colptr[j+1]-1)
            if A.rowval[idx] == j
                D[j] = A.nzval[idx]
                break
            end
        end
    end

    # Fill L with values from A where patterns overlap
    for j = 1:n
        for idx_L in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]

            if i == j
                # L has unit diagonal
                L.nzval[idx_L] = T(1)
            else
                # Look for A[i,j]
                found = false

                # Check column j of A for entry (i,j)
                for idx_A in A.colptr[j]:(A.colptr[j+1]-1)
                    if A.rowval[idx_A] == i
                        L.nzval[idx_L] = A.nzval[idx_A]
                        found = true
                        break
                    end
                end

                # For symmetric matrix, also check column i for entry (j,i)
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
                    L.nzval[idx_L] = T(0)  # Fill entry
                end
            end
        end
    end

    return nothing
end

"""
    ilu_k_symmetric_new(A::SparseMatrixCSC, k::Integer; shift::Real=0.0)

Compute incomplete LDL^T factorization for symmetric matrices.
More efficient than general ILU(k) as it only computes L.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level
- `shift`: Optional diagonal shift for stability

# Returns
- `L`: Lower triangular factor with unit diagonal
- `D`: Diagonal factor

# Example
```julia
A = sparse(Symmetric(randn(10,10)))
L, D = ilu_k_symmetric_new(A, 1)
# Factorization: A ≈ L * Diagonal(D) * L'
```
"""
function ilu_k_symmetric_new(A::SparseMatrixCSC{T}, k::Integer; shift::Real=T(0)) where {T}
    # Get symbolic pattern (only L)
    L = symbolic_ilu_k_symmetric_new(A, k)

    # Initialize D
    n = size(A, 1)
    D = zeros(T, n)

    # Fill with values from A
    fill_symbolic_symmetric_new!(A, L, D)

    # Apply shift if requested
    if shift != 0
        D .+= shift
    end

    # Perform numerical factorization
    numerical_ilu_k_symmetric!(L, D, shift)

    return L, D
end