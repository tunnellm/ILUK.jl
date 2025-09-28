"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.
Works directly with sparse matrices for efficiency.

Algorithm (Crout method):
For j = 1 to n:
  1. Compute column j of L (below diagonal): L[i,j] for i > j
  2. Compute row j of U (including diagonal): U[j,i] for i ≥ j

Based on Hysom & Pothen (1999) numerical factorization phase.
"""
function numerical_ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    shift::Real=T(0)
) where {T}

    n = size(L, 1)

    # Work vectors for current column/row
    work = zeros(T, n)  # Accumulates values for current column
    work_filled = Int[]  # Tracks which indices are nonzero in work

    # First, we need to construct A from L and U
    # After fill_symbolic!, L has lower triangle and U has upper triangle
    # We'll work with these directly but need to track original values

    # Apply diagonal shift if requested
    if shift != 0
        for j = 1:n
            for idx in U.colptr[j]:(U.colptr[j+1]-1)
                if U.rowval[idx] == j
                    U.nzval[idx] += shift
                    break
                end
            end
        end
    end

    # Process each column j using Crout method
    for j = 1:n

        # Step 1: Gather column j from the original matrix pattern
        # Original values are in L (lower) and U (upper)

        # Get column j values from L (for i >= j)
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            work[i] = L.nzval[idx]
            push!(work_filled, i)
        end

        # Get column j values from U (for i <= j)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            i = U.rowval[idx]
            if i <= j && !(i in work_filled)
                work[i] = U.nzval[idx]
                push!(work_filled, i)
            end
        end

        # Also need row j values from U (for i > j)
        for col = j+1:n
            for idx in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx] == j
                    work[col] = U.nzval[idx]
                    if !(col in work_filled)
                        push!(work_filled, col)
                    end
                    break
                end
            end
        end

        # Step 2: Apply Crout updates for column j

        # Update work vector: work[i] -= sum(L[i,k] * U[k,j] for k < j)
        for k = 1:j-1
            # Get U[k,j]
            u_kj = T(0)
            for idx in U.colptr[j]:(U.colptr[j+1]-1)
                if U.rowval[idx] == k
                    u_kj = U.nzval[idx]
                    break
                end
            end

            if u_kj != 0
                # Update all work[i] where L[i,k] exists
                for idx in L.colptr[k]:(L.colptr[k+1]-1)
                    i = L.rowval[idx]
                    if i >= j  # Only for diagonal and below
                        work[i] -= L.nzval[idx] * u_kj
                    end
                end
            end
        end

        # Step 3: Compute U[j,j]
        u_jj = work[j]
        if abs(u_jj) < eps(T) * 100
            error("Zero diagonal encountered at position ($j,$j)")
        end

        # Step 4: Store computed values back

        # Store U[j,j] and U[j,i] for i > j (row j of U)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                U.nzval[idx] = u_jj
                break
            end
        end

        for col = j+1:n
            if work[col] != 0
                for idx in U.colptr[col]:(U.colptr[col+1]-1)
                    if U.rowval[idx] == j
                        U.nzval[idx] = work[col]
                        break
                    end
                end
            end
        end

        # Store L[i,j] for i > j (column j of L)
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            if i > j
                L.nzval[idx] = work[i] / u_jj
            elseif i == j
                L.nzval[idx] = T(1)  # Unit diagonal
            end
        end

        # Clear work vector
        for i in work_filled
            work[i] = T(0)
        end
        empty!(work_filled)
    end

    return nothing
end