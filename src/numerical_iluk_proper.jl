"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.
Works directly with sparse CSC format.

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

    # Process each column j using Crout method
    for j = 1:n

        # Step 1: Compute U[j,j] (diagonal element)
        # U[j,j] = A[j,j] - sum(L[j,k] * U[k,j] for k=1:j-1)

        # Find original A[j,j] value (stored in U_orig)
        a_jj = T(0)
        u_jj_idx = 0
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[idx] == j
                a_jj = U_orig[idx]
                u_jj_idx = idx
                break
            end
        end

        # Compute sum(L[j,k] * U[k,j] for k=1:j-1)
        sum_diag = T(0)
        for k = 1:j-1
            # Get L[j,k]
            l_jk = T(0)
            for idx in L.colptr[k]:(L.colptr[k+1]-1)
                if L.rowval[idx] == j
                    l_jk = L.nzval[idx]
                    break
                end
            end

            # Get U[k,j]
            u_kj = T(0)
            for idx in U.colptr[j]:(U.colptr[j+1]-1)
                if U.rowval[idx] == k
                    u_kj = U.nzval[idx]
                    break
                end
            end

            sum_diag += l_jk * u_kj
        end

        u_jj = a_jj - sum_diag
        if abs(u_jj) < eps(T) * 100
            error("Zero diagonal at position ($j,$j)")
        end
        U.nzval[u_jj_idx] = u_jj

        # Step 2: Compute column j of L (below diagonal)
        # L[i,j] = (A[i,j] - sum(L[i,k] * U[k,j] for k=1:j-1)) / U[j,j]
        for idx_l in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx_l]

            if i == j
                # Diagonal of L is always 1
                L.nzval[idx_l] = T(1)
            elseif i > j
                # Get original A[i,j] value (stored in L_orig)
                a_ij = L_orig[idx_l]

                # Compute sum(L[i,k] * U[k,j] for k=1:j-1)
                sum_l = T(0)
                for k = 1:j-1
                    # Get L[i,k]
                    l_ik = T(0)
                    for idx in L.colptr[k]:(L.colptr[k+1]-1)
                        if L.rowval[idx] == i
                            l_ik = L.nzval[idx]
                            break
                        end
                    end

                    # Get U[k,j]
                    u_kj = T(0)
                    for idx in U.colptr[j]:(U.colptr[j+1]-1)
                        if U.rowval[idx] == k
                            u_kj = U.nzval[idx]
                            break
                        end
                    end

                    sum_l += l_ik * u_kj
                end

                L.nzval[idx_l] = (a_ij - sum_l) / u_jj
            end
        end

        # Step 3: Compute row j of U (above diagonal)
        # U[j,i] = A[j,i] - sum(L[j,k] * U[k,i] for k=1:j-1)
        for col = j+1:n
            # Find U[j,col] if it exists
            for idx_u in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx_u] == j
                    # Get original A[j,col] value (stored in U_orig)
                    a_jcol = U_orig[idx_u]

                    # Compute sum(L[j,k] * U[k,col] for k=1:j-1)
                    sum_u = T(0)
                    for k = 1:j-1
                        # Get L[j,k]
                        l_jk = T(0)
                        for idx in L.colptr[k]:(L.colptr[k+1]-1)
                            if L.rowval[idx] == j
                                l_jk = L.nzval[idx]
                                break
                            end
                        end

                        # Get U[k,col]
                        u_kcol = T(0)
                        for idx in U.colptr[col]:(U.colptr[col+1]-1)
                            if U.rowval[idx] == k
                                u_kcol = U.nzval[idx]
                                break
                            end
                        end

                        sum_u += l_jk * u_kcol
                    end

                    U.nzval[idx_u] = a_jcol - sum_u
                    break
                end
            end
        end
    end

    return nothing
end