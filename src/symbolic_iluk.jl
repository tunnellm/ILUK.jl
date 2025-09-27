"""
    symbolic_ilu_k(A::SparseMatrixCSC{<:Number}, k::Integer)

Compute the symbolic ILU(k) factorization of a sparse matrix A.

This implementation follows the algorithm by Pothen & Fan for symbolic factorization
with level-of-fill k. It uses a BFS-based approach to track fill levels through
the elimination graph.

# Arguments
- `A`: Input sparse matrix in CSC format
- `k`: Level of fill (k ≥ 0)

# Returns
- `L`: Lower triangular factor (symbolic pattern with ones)
- `U`: Upper triangular factor (symbolic pattern with ones)

# Algorithm
The algorithm determines fill-in based on paths in the elimination graph.
A fill entry (i,j) is included if there exists a path from i to j through
intermediate vertices that are all less than min(i,j), and the path length
(level) is at most k.

# References
- Pothen, A., & Fan, C. J. (1990). Computing the block triangular form of a
  sparse matrix. ACM Transactions on Mathematical Software, 16(4), 303-324.
"""
function symbolic_ilu_k(A::SparseMatrixCSC{<:Number}, k::Integer)

    if k == 0
        L = tril(A)
        U = triu(A)
        L.nzval .= 1.0
        U.nzval .= 1.0
        return L, U
    end

    n = size(A, 1)

    # First pass: count nonzeros
    L_counts = zeros(Int, n)
    U_counts = zeros(Int, n)

    fill_symbolic!(L_counts, U_counts, A, k, true)

    # Allocate CSC structures
    L_colptr = Vector{Int}(undef, n + 1)
    L_colptr[1] = 1
    for j in 1:n
        L_colptr[j+1] = L_colptr[j] + L_counts[j]
    end
    L_rowidx = Vector{Int}(undef, L_colptr[end] - 1)
    L_nzval = ones(Float64, L_colptr[end] - 1)

    U_colptr = Vector{Int}(undef, n + 1)
    U_colptr[1] = 1
    for j in 1:n
        U_colptr[j+1] = U_colptr[j] + U_counts[j]
    end
    U_rowidx = Vector{Int}(undef, U_colptr[end] - 1)
    U_nzval = ones(Float64, U_colptr[end] - 1)

    # Second pass: fill structures
    fill!(L_counts, 0)
    fill!(U_counts, 0)

    fill_symbolic!(L_counts, U_counts, A, k, false,
                   L_colptr, L_rowidx, U_colptr, U_rowidx)

    L = SparseMatrixCSC(n, n, L_colptr, L_rowidx, L_nzval)
    U = SparseMatrixCSC(n, n, U_colptr, U_rowidx, U_nzval)

    return L, U
end

"""
    fill_symbolic!(L_counts, U_counts, A, k, count_only,
                   L_colptr=nothing, L_rowidx=nothing,
                   U_colptr=nothing, U_rowidx=nothing)

Core symbolic factorization routine using BFS level propagation.
Can run in counting mode (count_only=true) or filling mode (count_only=false).
"""
function fill_symbolic!(L_counts, U_counts, A, k, count_only,
                        L_colptr=nothing, L_rowidx=nothing,
                        U_colptr=nothing, U_rowidx=nothing)

    n = size(A, 1)

    # Work arrays
    level = fill(k + 1, n)  # Level/distance array
    queue = Vector{Int}(undef, n)  # BFS queue
    touched = Vector{Int}(undef, n)  # Track modified entries

    # Track U pattern for each row (sparse representation)
    U_idx = [Int[] for _ in 1:n]
    U_lvl = [Int[] for _ in 1:n]

    for i in 1:n
        # Reset work arrays
        touched_t = 0

        # Initialize with pattern from A
        for j_idx in A.colptr[i]:(A.colptr[i+1]-1)
            j = A.rowval[j_idx]
            if j < i
                # L part: j is in L[i,:]
                level[j] = 0
                touched_t += 1
                touched[touched_t] = j
            elseif j > i
                # U part: j is in U[i,:]
                level[j] = 0
                touched_t += 1
                touched[touched_t] = j
            end
        end

        # Initialize BFS queue with L-part entries
        head = 1
        tail = 0
        for tt in 1:touched_t
            j = touched[tt]
            if j < i  # L-part
                tail += 1
                queue[tail] = j
            end
        end

        # BFS over U-patterns of previous rows
        while head <= tail
            j = queue[head]
            head += 1
            lvlj = level[j]

            # Propagate through U[j,:]
            for (w, wlvl) in zip(U_idx[j], U_lvl[j])
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

        # Sort and store results
        L_row = Int[]
        U_row = Int[]
        U_row_lvl = Int[]

        for tt in 1:touched_t
            j = touched[tt]
            if level[j] <= k
                if j < i
                    push!(L_row, j)
                elseif j > i
                    push!(U_row, j)
                    push!(U_row_lvl, level[j])
                end
            end
            level[j] = k + 1  # Reset for next iteration
        end

        # Add diagonal
        push!(L_row, i)
        push!(U_row, i)
        push!(U_row_lvl, 0)

        sort!(L_row)
        perm = sortperm(U_row)
        U_row = U_row[perm]
        U_row_lvl = U_row_lvl[perm]

        # Store U pattern for this row
        U_idx[i] = U_row
        U_lvl[i] = U_row_lvl

        if count_only
            # Count nonzeros
            for j in L_row
                L_counts[j] += 1
            end
            for j in U_row
                U_counts[j] += 1
            end
        else
            # Fill CSC structures
            for j in L_row
                L_rowidx[L_colptr[j] + L_counts[j]] = i
                L_counts[j] += 1
            end
            for j in U_row
                U_rowidx[U_colptr[j] + U_counts[j]] = i
                U_counts[j] += 1
            end
        end
    end
end

export symbolic_ilu_k