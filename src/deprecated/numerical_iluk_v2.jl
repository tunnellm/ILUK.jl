"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.
Processes column-by-column for efficiency with CSC format.
"""
function numerical_ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    shift::Real=T(0)
) where {T}

    n = size(L, 1)

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

    # Work arrays
    work = zeros(T, n)  # Accumulates values for current column

    # Process each column j
    for j = 1:n

        # STEP 1: Gather values for column j from L and row j from U
        # This gives us the j-th column of the original matrix pattern

        # From L[:,j] (column j of L)
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            work[i] = L.nzval[idx]
        end

        # From U[j,:] (row j of U) - need to search all columns ≥ j
        for col = j:n
            for idx in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx] == j
                    work[col] = U.nzval[idx]
                    break
                end
            end
        end

        # STEP 2: Apply updates from previous columns
        # Update work[i] -= L[i,k] * U[k,j] for k < j

        # For each column k < j
        for k = 1:j-1
            # Get U[k,j]
            u_kj = T(0)
            for idx in U.colptr[j]:(U.colptr[j+1]-1)
                if U.rowval[idx] == k
                    u_kj = U.nzval[idx]
                    break
                end
            end

            # If U[k,j] exists, update all work[i] where L[i,k] exists
            if u_kj != 0
                for idx in L.colptr[k]:(L.colptr[k+1]-1)
                    i = L.rowval[idx]
                    if i > k  # Only for strict lower part
                        work[i] -= L.nzval[idx] * u_kj
                    end
                end
            end
        end

        # STEP 3: Scale column j of L by 1/U[j,j]
        u_jj = work[j]
        if abs(u_jj) < eps(T) * 100
            error("Zero diagonal encountered at position ($j,$j)")
        end

        # STEP 4: Store results back

        # Store L[:,j] (below diagonal)
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            if i > j
                L.nzval[idx] = work[i] / u_jj
            elseif i == j
                L.nzval[idx] = T(1)  # Unit diagonal
            end
            work[i] = T(0)  # Clear work array
        end

        # Store U[j,:] (row j, at and above diagonal)
        # First store U[j,j]
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                U.nzval[idx] = u_jj
                break
            end
        end

        # Store rest of row j of U
        for col = j+1:n
            if work[col] != 0
                for idx in U.colptr[col]:(U.colptr[col+1]-1)
                    if U.rowval[idx] == j
                        U.nzval[idx] = work[col]
                        work[col] = T(0)  # Clear
                        break
                    end
                end
            end
        end
    end

    return nothing
end