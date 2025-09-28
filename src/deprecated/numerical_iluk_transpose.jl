"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.
Efficient implementation using transpose for row operations on U.

After fill_symbolic!, L and U contain the values from A in their respective patterns.
This function performs the numerical factorization in-place.
"""
function numerical_ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    shift::Real=T(0)
) where {T}

    n = size(L, 1)

    # Save original matrix values before we start modifying
    # After fill_symbolic!, L has lower triangle and U has upper triangle of A
    L_orig = copy(L.nzval)
    U_orig = copy(U.nzval)

    # Apply diagonal shift to the saved U diagonal values
    if shift != 0
        for j = 1:n
            for idx in U.colptr[j]:(U.colptr[j+1]-1)
                if U.rowval[idx] == j
                    U_orig[idx] += shift
                    break
                end
            end
        end
    end

    # Transpose U to get row-wise access (U^T in CSC is like U in CSR)
    Ut = transpose_keepzeros(U)
    Ut_orig = copy(Ut.nzval)  # Save transposed original values

    # Work vectors for efficient accumulation
    work_L = zeros(T, n)  # For accumulating L[j,:] (row j of L)
    work_U = zeros(T, n)  # For accumulating U[:,j] (column j of U)
    L_row_filled = Int[]  # Track nonzero indices in work_L

    # Process each column j using Crout method
    for j = 1:n

        # Step 1: Compute U[j,j] (diagonal element)
        # U[j,j] = A[j,j] - sum(L[j,k] * U[k,j] for k=1:j-1)

        # First, gather row j of L into work vector
        # L[j,k] for k < j are stored in columns k < j
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

        # Find original A[j,j] value (diagonal is in U)
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                a_jj = U_orig[idx]
                u_jj_idx = idx
                break
            end
        end

        # Also find it in Ut for later update
        ut_jj_idx = 0
        for idx in Ut.colptr[j]:(Ut.colptr[j+1]-1)
            if Ut.rowval[idx] == j
                ut_jj_idx = idx
                break
            end
        end

        # Compute sum(L[j,k] * U[k,j] for k=1:j-1)
        # We have L[j,k] in work_L, need U[k,j] from column j of U
        sum_diag = T(0)
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j && work_L[k] != 0
                sum_diag += work_L[k] * U.nzval[idx]
            end
        end

        u_jj = a_jj - sum_diag
        if abs(u_jj) < eps(T) * 100
            error("Zero diagonal at position ($j,$j)")
        end
        U.nzval[u_jj_idx] = u_jj
        Ut.nzval[ut_jj_idx] = u_jj  # Update in transpose too

        # Step 2: Compute column j of L (below diagonal)
        # L[i,j] = (A[i,j] - sum(L[i,k] * U[k,j] for k=1:j-1)) / U[j,j]

        # First gather column j of U into work vector for efficient access
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j
                work_U[k] = U.nzval[idx]
            end
        end

        # Process column j of L
        for idx_l in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_l]

            if i == j
                # Diagonal of L is always 1
                L.nzval[idx_l] = T(1)
            elseif i > j
                # Get original A[i,j] value (stored in L_orig)
                a_ij = L_orig[idx_l]

                # Compute sum(L[i,k] * U[k,j] for k=1:j-1)
                # U[k,j] values are in work_U
                # Need to get L[i,k] from columns k < j
                sum_l = T(0)
                for k = 1:j-1
                    if work_U[k] != 0  # U[k,j] exists
                        # Find L[i,k] in column k
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

        # Clear work_U for next iteration
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            k = U.rowval[idx]
            if k < j
                work_U[k] = T(0)
            end
        end

        # Step 3: Compute row j of U (above diagonal)
        # U[j,col] = A[j,col] - sum(L[j,k] * U[k,col] for k=1:j-1)
        # We already have L[j,k] in work_L from Step 1
        # Now we can use Ut to efficiently access row j of U!

        # Process row j of U using Ut (which gives us column j of Ut = row j of U)
        for idx_ut in Ut.colptr[j]:(Ut.colptr[j+1]-1)
            col = Ut.rowval[idx_ut]
            if col > j  # Only process upper triangular part
                # Get original A[j,col] value (stored in Ut_orig)
                a_jcol = Ut_orig[idx_ut]

                # Compute sum(L[j,k] * U[k,col] for k=1:j-1)
                # L[j,k] values are in work_L
                # Need U[k,col] from column col of U
                sum_u = T(0)
                for idx in U.colptr[col]:(U.colptr[col+1]-1)
                    k = U.rowval[idx]
                    if k < j && work_L[k] != 0
                        sum_u += work_L[k] * U.nzval[idx]
                    end
                end

                new_val = a_jcol - sum_u
                Ut.nzval[idx_ut] = new_val

                # Also update in U (need to find U[j,col])
                for idx in U.colptr[col]:(U.colptr[col+1]-1)
                    if U.rowval[idx] == j
                        U.nzval[idx] = new_val
                        break
                    end
                end
            end
        end

        # Clear work_L for next iteration
        for k in L_row_filled
            work_L[k] = T(0)
        end
    end

    return nothing
end