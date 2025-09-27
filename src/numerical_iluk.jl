"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.
This implementation prioritizes correctness and clarity.

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

    # Convert to dense for easier debugging and verification
    # This is inefficient but helps ensure correctness
    L_dense = Matrix(L)
    U_dense = Matrix(U)

    # Apply diagonal shift
    if shift != 0
        for j = 1:n
            U_dense[j,j] += shift
        end
    end

    # Crout method: process column by column
    for j = 1:n

        # Step 1: Compute U[j,j] first (diagonal element)
        # U[j,j] = A[j,j] - sum(L[j,k] * U[k,j] for k=1:j-1)
        sum_diag = T(0)
        for k = 1:j-1
            sum_diag += L_dense[j,k] * U_dense[k,j]
        end
        U_dense[j,j] = U_dense[j,j] - sum_diag

        if abs(U_dense[j,j]) < eps(T) * 100
            error("Zero diagonal at position ($j,$j): U[$j,$j] = $(U_dense[j,j])")
        end

        # Step 2: Compute column j of L (below diagonal)
        # L[i,j] = (A[i,j] - sum(L[i,k] * U[k,j] for k=1:j-1)) / U[j,j]
        for i = j+1:n
            if L_dense[i,j] != 0 || U_dense[i,j] != 0  # Only if in pattern
                sum_L = T(0)
                for k = 1:j-1
                    sum_L += L_dense[i,k] * U_dense[k,j]
                end
                # Original value is in L_dense or U_dense (from fill_symbolic!)
                orig_val = L_dense[i,j] + U_dense[i,j]
                L_dense[i,j] = (orig_val - sum_L) / U_dense[j,j]
            end
        end

        # Step 3: Compute row j of U (above diagonal)
        # U[j,i] = A[j,i] - sum(L[j,k] * U[k,i] for k=1:j-1)
        for i = j+1:n
            if U_dense[j,i] != 0 || L_dense[j,i] != 0  # Only if in pattern
                sum_U = T(0)
                for k = 1:j-1
                    sum_U += L_dense[j,k] * U_dense[k,i]
                end
                # Original value
                orig_val = L_dense[j,i] + U_dense[j,i]
                U_dense[j,i] = orig_val - sum_U
            end
        end

        # Set L's diagonal to 1 (unit diagonal)
        L_dense[j,j] = T(1)
    end

    # Copy back to sparse matrices (only non-zero entries)
    for j = 1:n
        # Copy L column j
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            L.nzval[idx] = L_dense[i,j]
        end

        # Copy U column j
        for idx in U.colptr[j]:(U.colptr[j+1]-1)
            i = U.rowval[idx]
            U.nzval[idx] = U_dense[i,j]
        end
    end

    return nothing
end