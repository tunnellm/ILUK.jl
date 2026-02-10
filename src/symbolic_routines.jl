"""
    Symbolic Factorization Routines

All symbolic phase computations for ILU(k) and LDL^T(k) factorizations.
Includes pattern generation and initialization with matrix values.
"""

# =============================================================================
# GENERAL SYMBOLIC FACTORIZATION
# =============================================================================

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
- `L`: Strictly lower triangular sparsity pattern (no diagonal)
- `U`: Strictly upper triangular sparsity pattern (no diagonal)

Diagonal is handled separately by D vector in numerical phase.

# Algorithm
The algorithm determines fill-in based on paths in the elimination graph.
A fill entry (i,j) is included if there exists a path from i to j through
intermediate vertices that are all less than min(i,j), and the path length
(level) is at most k.
"""
function symbolic_ilu_k(A::SparseMatrixCSC{<:Number}, k::Integer)

    if k == 0
        # For ILU(0), use pattern of A excluding diagonal (D vector handles diagonal)
        L = tril(A, -1)  # Strictly lower triangular
        U = triu(A, 1)   # Strictly upper triangular

        L.nzval .= 1.0
        U.nzval .= 1.0
        # k=0: single pass through A to partition into L and U
        graph_ops = nnz(A)
        return L, U, graph_ops
    end

    n = size(A, 1)
    nnzA = nnz(A)
    graph_ops = 0

    # 1) Build row-wise pattern of A (CSC -> CSR-like structure)
    # This is critical: we need to iterate over rows, not columns
    row_counts = zeros(Int, n)
    for col in 1:n
        for p in A.colptr[col]:(A.colptr[col+1]-1)
            row_counts[A.rowval[p]] += 1
        end
    end
    graph_ops += nnzA  # 1 index read per nonzero

    row_ptr = Vector{Int}(undef, n + 1)
    row_ptr[1] = 1
    for i in 1:n
        row_ptr[i+1] = row_ptr[i] + row_counts[i]
    end
    graph_ops += n  # prefix sum over n rows

    row_pattern = Vector{Int}(undef, nnzA)
    temp = zeros(Int, n)
    for col in 1:n
        for p in A.colptr[col]:(A.colptr[col+1]-1)
            r = A.rowval[p]
            idx = row_ptr[r] + temp[r]
            row_pattern[idx] = col
            temp[r] += 1
        end
    end
    graph_ops += nnzA  # 1 index store per nonzero

    # 2) Workspace and counters
    countsL = zeros(Int, n)
    countsU = zeros(Int, n)
    level = fill(k + 1, n)
    queue = Vector{Int}(undef, n)
    touched = Vector{Int}(undef, n)
    U_idx = Vector{Vector{Int}}(undef, n)
    U_lvl = Vector{Vector{Int}}(undef, n)

    # First pass: record per-row U-patterns and tally counts
    for i in 1:n
        touched_t = 0
        tail = 0

        # Seed BFS from original row pattern (ROW i of A)
        for idx in row_ptr[i]:(row_ptr[i+1]-1)
            c = row_pattern[idx]  # c is column index, A[i,c] is nonzero
            graph_ops += 1  # read from row_pattern
            if level[c] == k + 1
                touched_t += 1
                touched[touched_t] = c
                level[c] = 0
                if c < i
                    tail += 1
                    queue[tail] = c
                end
            end
        end

        # BFS over U-patterns of previous rows
        head = 1
        while head <= tail
            j = queue[head]
            head += 1
            lvlj = level[j]
            for (w, wlvl) in zip(U_idx[j], U_lvl[j])
                graph_ops += 1  # edge traversal: read from U_idx/U_lvl
                new_lvl = lvlj + wlvl + 1
                if new_lvl <= k && new_lvl < level[w]
                    if level[w] == k + 1
                        # First touch - add to queue only once
                        touched_t += 1
                        touched[touched_t] = w
                        if w < i
                            tail += 1
                            queue[tail] = w
                        end
                    end
                    level[w] = new_lvl
                end
            end
        end

        # Count L/U entries and record U-pattern
        # L is strictly lower triangular (c < i), U is strictly upper (c > i)
        # Diagonal is handled separately by D vector
        ucount = 0
        for t in 1:touched_t
            c = touched[t]
            lvl = level[c]
            graph_ops += 1  # read touched entry
            if lvl <= k
                if c < i
                    countsL[c] += 1
                end
                if c > i
                    countsU[c] += 1
                end
                if c >= i
                    ucount += 1  # Still track for BFS (includes diagonal for path propagation)
                end
            end
        end

        # Allocate and fill per-row U_idx/U_lvl
        Ui = Vector{Int}(undef, ucount)
        Vi = Vector{Int}(undef, ucount)
        pos = 0
        for t in 1:touched_t
            c = touched[t]
            lvl = level[c]
            if lvl <= k && c >= i
                pos += 1
                Ui[pos] = c
                Vi[pos] = lvl
                graph_ops += 1  # store into U_idx/U_lvl
            end
            level[c] = k + 1  # Reset for next iteration
        end
        U_idx[i] = Ui
        U_lvl[i] = Vi
    end

    # 3) Build CSC column pointers
    L_colptr = Vector{Int}(undef, n + 1)
    U_colptr = Vector{Int}(undef, n + 1)
    L_colptr[1] = 1
    U_colptr[1] = 1
    for j in 1:n
        L_colptr[j+1] = L_colptr[j] + countsL[j]
        U_colptr[j+1] = U_colptr[j] + countsU[j]
    end
    graph_ops += 2 * n  # L and U colptr construction
    Lnnz = L_colptr[end] - 1
    Unnz = U_colptr[end] - 1

    # 4) Allocate CSC storage
    L_rowval = Vector{Int}(undef, Lnnz)
    U_rowval = Vector{Int}(undef, Unnz)
    L_vals = ones(Float64, Lnnz)
    U_vals = ones(Float64, Unnz)
    nextL = copy(L_colptr)
    nextU = copy(U_colptr)

    # 5) Second pass: fill CSC arrays
    fill!(level, k + 1)
    for i in 1:n
        touched_t = 0
        tail = 0

        # Seed from row pattern
        for idx in row_ptr[i]:(row_ptr[i+1]-1)
            c = row_pattern[idx]
            graph_ops += 1  # read from row_pattern
            if level[c] == k + 1
                touched_t += 1
                touched[touched_t] = c
                level[c] = 0
                if c < i
                    tail += 1
                    queue[tail] = c
                end
            end
        end

        # BFS
        head = 1
        while head <= tail
            j = queue[head]
            head += 1
            lvlj = level[j]
            for (w, wlvl) in zip(U_idx[j], U_lvl[j])
                graph_ops += 1  # edge traversal: read from U_idx/U_lvl
                new_lvl = lvlj + wlvl + 1
                if new_lvl <= k && new_lvl < level[w]
                    if level[w] == k + 1
                        touched_t += 1
                        touched[touched_t] = w
                        if w < i
                            tail += 1
                            queue[tail] = w
                        end
                    end
                    level[w] = new_lvl
                end
            end
        end

        # Commit to CSC (strictly lower L, strictly upper U)
        for t in 1:touched_t
            c = touched[t]
            lvl = level[c]
            graph_ops += 1  # read touched entry
            if lvl <= k
                if c < i
                    p = nextL[c]
                    L_rowval[p] = i
                    nextL[c] += 1
                    graph_ops += 1  # store into L CSC
                end
                if c > i
                    p = nextU[c]
                    U_rowval[p] = i
                    nextU[c] += 1
                    graph_ops += 1  # store into U CSC
                end
            end
            level[c] = k + 1
        end
    end

    # 6) Construct SparseMatrixCSC and return
    L = SparseMatrixCSC(n, n, L_colptr, L_rowval, L_vals)
    U = SparseMatrixCSC(n, n, U_colptr, U_rowval, U_vals)
    return L, U, graph_ops
end

export symbolic_ilu_k

# =============================================================================
# SYMMETRIC SYMBOLIC FACTORIZATION
# =============================================================================
"""
    symbolic_cholesky(A::SparseMatrixCSC, k::Integer)

Compute symbolic ILU(k) pattern for symmetric matrices.
Returns strictly lower triangular L (no diagonal) since diagonal is handled by D.

# Arguments
- `A`: Symmetric sparse matrix
- `k`: Fill level (0 ≤ k)

# Returns
- `L`: Sparse strictly lower triangular matrix with symbolic pattern (no diagonal)

# Algorithm
Adapted BFS approach for symmetric matrices where U = L^T.
For each row i, we propagate fill through the transpose relationship.
"""
function symbolic_cholesky(A::SparseMatrixCSC{T}, k::Integer) where {T}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    @assert k >= 0 "Fill level k must be non-negative"

    if k == 0
        L = tril(A, -1)
        L.nzval .= 1
        # k=0: single pass through lower triangle of A
        graph_ops = nnz(A) ÷ 2
        return L, graph_ops
    end

    nnzA = nnz(A)
    graph_ops = 0

    # Build row-wise pattern of A (once, O(nnz))
    row_counts = zeros(Int, n)
    @inbounds for col = 1:n
        for p = A.colptr[col]:(A.colptr[col+1]-1)
            row_counts[A.rowval[p]] += 1
        end
    end
    graph_ops += nnzA  # 1 index read per nonzero

    row_ptr = Vector{Int}(undef, n + 1)
    row_ptr[1] = 1
    @inbounds for i = 1:n
        row_ptr[i+1] = row_ptr[i] + row_counts[i]
    end
    graph_ops += n  # prefix sum over n rows

    row_pattern = Vector{Int}(undef, nnzA)
    fill!(row_counts, 0)
    @inbounds for col = 1:n
        for p = A.colptr[col]:(A.colptr[col+1]-1)
            r = A.rowval[p]
            row_pattern[row_ptr[r] + row_counts[r]] = col
            row_counts[r] += 1
        end
    end
    graph_ops += nnzA  # 1 index store per nonzero

    # Work arrays
    level = fill(k + 1, n)
    queue = Vector{Int}(undef, n)
    touched = Vector{Int}(undef, n)
    # L_col[j] = column j of L (rows > j) = row j of U (for symmetric: L = U^T)
    L_col = Vector{Vector{Int}}(undef, n)
    L_col_lvl = Vector{Vector{Int}}(undef, n)

    # Single pass: compute pattern using BFS, store L column by column
    @inbounds for i = 1:n
        touched_t = 0
        tail = 0

        # Seed from row i of A
        for rp = row_ptr[i]:(row_ptr[i+1]-1)
            c = row_pattern[rp]
            graph_ops += 1  # read from row_pattern
            if level[c] == k + 1
                touched_t += 1
                touched[touched_t] = c
                level[c] = 0
                if c < i
                    tail += 1
                    queue[tail] = c
                end
            end
        end

        # BFS over previous columns of L (= rows of U)
        head = 1
        while head <= tail
            j = queue[head]
            head += 1
            lvlj = level[j]
            for (w, wlvl) in zip(L_col[j], L_col_lvl[j])
                graph_ops += 1  # edge traversal: read from L_col/L_col_lvl
                new_lvl = lvlj + wlvl + 1
                if new_lvl <= k && new_lvl < level[w]
                    if level[w] == k + 1
                        touched_t += 1
                        touched[touched_t] = w
                        if w < i
                            tail += 1
                            queue[tail] = w
                        end
                    end
                    level[w] = new_lvl
                end
            end
        end

        # Store column i of L (rows >= i, but we only keep rows > i for strictly lower)
        # This is also row i of U, needed for BFS of future rows
        col_count = 0
        for t = 1:touched_t
            c = touched[t]
            graph_ops += 1  # read touched entry
            if level[c] <= k && c >= i
                col_count += 1
            end
        end

        Li = Vector{Int}(undef, col_count)
        Lv = Vector{Int}(undef, col_count)
        pos = 0
        for t = 1:touched_t
            c = touched[t]
            lvl = level[c]
            if lvl <= k && c >= i
                pos += 1
                Li[pos] = c
                Lv[pos] = lvl
                graph_ops += 1  # store into L_col/L_col_lvl
            end
            level[c] = k + 1
        end
        L_col[i] = Li
        L_col_lvl[i] = Lv
    end

    # Build CSC structure from L_col (which stores column j as rows >= j)
    # For strictly lower triangular, column j has rows > j
    L_colptr = Vector{Int}(undef, n + 1)
    L_colptr[1] = 1
    @inbounds for j = 1:n
        # Count entries > j (exclude diagonal)
        cnt = 0
        for r in L_col[j]
            graph_ops += 1  # read from L_col during CSC construction
            if r > j
                cnt += 1
            end
        end
        L_colptr[j+1] = L_colptr[j] + cnt
    end
    graph_ops += n  # colptr prefix sum
    Lnnz = L_colptr[end] - 1

    L_rowval = Vector{Int}(undef, Lnnz)
    L_nzval = ones(T, Lnnz)

    @inbounds for j = 1:n
        # Collect rows > j and sort
        p_start = L_colptr[j]
        p = p_start
        for r in L_col[j]
            if r > j
                L_rowval[p] = r
                graph_ops += 1  # store into L CSC
                p += 1
            end
        end
        # Sort this column's row indices
        col_nnz = p - p_start
        if col_nnz > 1
            sort!(view(L_rowval, p_start:(p-1)))
            # sort cost: n*log(n) comparisons
            graph_ops += ceil(Int, col_nnz * log2(col_nnz))
        end
    end

    return SparseMatrixCSC(n, n, L_colptr, L_rowval, L_nzval), graph_ops
end

"""
    fill_symbolic_symmetric!(A::SparseMatrixCSC, L::SparseMatrixCSC, D::Vector)

Fill L with values from symmetric matrix A where patterns overlap.
Initialize diagonal vector D with diagonal values from A.

L is strictly lower triangular (no diagonal entries).

# Arguments
- `A`: Symmetric sparse matrix
- `L`: Strictly lower triangular symbolic pattern (modified in-place)
- `D`: Diagonal vector (modified in-place)
"""
function fill_symbolic_symmetric!(
    A::SparseMatrixCSC{T},
    L::SparseMatrixCSC{T},
    D::Vector{T},
) where {T}
    n = size(L, 1)
    graph_ops = 0

    # Initialize D with diagonal of A
    for j = 1:n
        D[j] = T(0)  # Default value
        # Look for A[j,j]
        for idx = A.colptr[j]:(A.colptr[j+1]-1)
            graph_ops += 1  # index read during diagonal search
            if A.rowval[idx] == j
                D[j] = A.nzval[idx]
                break
            end
        end
    end

    # Fill L with values from A where patterns overlap
    # L is strictly lower triangular (i > j for all entries)
    for j = 1:n
        for idx_L = L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_L]
            graph_ops += 1  # index read from L

            # Look for A[i,j] (i > j since L is strictly lower triangular)
            found = false

            # Check column j of A for entry (i,j)
            for idx_A = A.colptr[j]:(A.colptr[j+1]-1)
                graph_ops += 1  # index read during search in A[:,j]
                if A.rowval[idx_A] == i
                    L.nzval[idx_L] = A.nzval[idx_A]
                    found = true
                    break
                end
            end

            # For symmetric matrix, also check column i for entry (j,i) = A[j,i] = A[i,j]
            if !found
                for idx_A = A.colptr[i]:(A.colptr[i+1]-1)
                    graph_ops += 1  # index read during search in A[:,i]
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

    return graph_ops
end

# =============================================================================
# INITIALIZATION WITH MATRIX VALUES
# =============================================================================
"""
    fill_symbolic!(A, L, U, D)

Fill L, U, D with values from matrix A where patterns overlap.
L is strictly lower triangular, U is strictly upper triangular, D is diagonal.
"""
function fill_symbolic!(
    A::SparseMatrixCSC{T},
    Lc::SparseMatrixCSC{T},
    Uc::SparseMatrixCSC{T},
    D::Vector{T},
) where {T}

    n = size(A, 2)
    graph_ops = 0

    Lc.nzval .= T(0)
    Uc.nzval .= T(0)
    D .= T(0)

    @inbounds for j = 1:n
        pa = A.colptr[j]
        pa_end = A.colptr[j+1]
        pl = Lc.colptr[j]
        pl_end = Lc.colptr[j+1]
        pu = Uc.colptr[j]
        pu_end = Uc.colptr[j+1]

        # advance-until helpers ------------------------------------------------
        advance_l!(r) = begin
            while pl < pl_end && Lc.rowval[pl] < r
                pl += 1
                graph_ops += 1  # index read during advance
            end
        end
        advance_u!(r) = begin
            while pu < pu_end && Uc.rowval[pu] < r
                pu += 1
                graph_ops += 1  # index read during advance
            end
        end

        # ---------------------------------------------------------------------
        while pa < pa_end
            ra = A.rowval[pa]
            graph_ops += 1  # index read from A

            if ra == j
                # diagonal
                D[j] = A.nzval[pa]
            elseif ra < j
                # strictly upper (row < col)
                advance_u!(ra)
                if pu < pu_end && Uc.rowval[pu] == ra
                    Uc.nzval[pu] = A.nzval[pa]
                end
            else
                # strictly lower (row > col)
                advance_l!(ra)
                if pl < pl_end && Lc.rowval[pl] == ra
                    Lc.nzval[pl] = A.nzval[pa]
                end
            end

            pa += 1
        end
    end
    return graph_ops
end

function transpose_keepzeros(A::SparseMatrixCSC{T}) where {T}
    m, n = size(A)
    nnzA = nnz(A)

    colptrT = Vector{Int}(undef, m + 1)
    rowvalT = Vector{Int}(undef, nnzA)
    nzvalT = similar(A.nzval)

    graph_ops = 0

    # pass 1 – count how many entries each output column will get
    fill!(colptrT, 0)
    @inbounds for r in A.rowval
        colptrT[r] += 1
    end
    graph_ops += nnzA  # 1 index read per nonzero

    # cumulative sum → true column pointers
    s = 1
    @inbounds for j = 1:m
        t = colptrT[j]
        colptrT[j] = s
        s += t
    end
    colptrT[m+1] = nnzA + 1
    graph_ops += m  # 1 store per column

    # work array that tracks next free slot in each column
    nextptr = copy(colptrT)

    # pass 2 – write row indices & values
    @inbounds for j = 1:n
        for p = A.colptr[j]:(A.colptr[j+1]-1)
            i = A.rowval[p]
            q = nextptr[i]
            rowvalT[q] = j
            nzvalT[q] = A.nzval[p]          # zero values kept verbatim
            nextptr[i] = q + 1
        end
    end
    graph_ops += nnzA  # 1 index store per nonzero

    return SparseMatrixCSC{T,Int}(n, m, colptrT, rowvalT, nzvalT), graph_ops
end
