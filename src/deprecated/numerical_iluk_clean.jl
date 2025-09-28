"""
    numerical_ilu_k!(L::SparseMatrixCSC, U::SparseMatrixCSC, shift::Real=0.0)

Perform numerical ILU(k) factorization using the Crout method.

The function assumes L and U contain:
- The symbolic pattern from symbolic_ilu_k
- Initial values from A (via fill_symbolic!)

Algorithm (Crout method):
For j = 1 to n:
  1. Compute U[j,j] and row j of U
  2. Compute column j of L

Based on Hysom & Pothen (1999) numerical factorization phase.
"""
function numerical_ilu_k!(
    L::SparseMatrixCSC{T},
    U::SparseMatrixCSC{T},
    shift::Real=T(0)
) where {T}

    n = size(L, 1)

    # Create a working copy of the original matrix values
    # After fill_symbolic!, L contains lower triangle and U contains upper triangle
    # We need to preserve these original values
    A_pattern = L + U  # This gives us the pattern
    A_vals = copy(A_pattern.nzval)  # Save the original values

    # Apply diagonal shift
    if shift != 0
        for j = 1:n
            for k in A_pattern.colptr[j]:(A_pattern.colptr[j+1]-1)
                if A_pattern.rowval[k] == j
                    A_vals[k] += shift
                    break
                end
            end
        end
    end

    # Helper function to get A[i,j] from saved values
    function get_A(i::Int, j::Int)
        for k in A_pattern.colptr[j]:(A_pattern.colptr[j+1]-1)
            if A_pattern.rowval[k] == i
                return A_vals[k]
            end
        end
        return T(0)
    end

    # Helper function to set value in L or U
    function set_L!(i::Int, j::Int, val::T)
        for k in L.colptr[j]:(L.colptr[j+1]-1)
            if L.rowval[k] == i
                L.nzval[k] = val
                return
            end
        end
    end

    function set_U!(i::Int, j::Int, val::T)
        for k in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[k] == i
                U.nzval[k] = val
                return
            end
        end
    end

    function get_L(i::Int, j::Int)
        for k in L.colptr[j]:(L.colptr[j+1]-1)
            if L.rowval[k] == i
                return L.nzval[k]
            end
        end
        return T(0)
    end

    function get_U(i::Int, j::Int)
        for k in U.colptr[j]:(U.colptr[j+1]-1)
            if U.rowval[k] == i
                return U.nzval[k]
            end
        end
        return T(0)
    end

    # Crout method: process column by column
    for j = 1:n

        # Step 1: Compute U[j,j] (diagonal)
        sum_diag = T(0)
        for k = 1:j-1
            sum_diag += get_L(j, k) * get_U(k, j)
        end
        u_jj = get_A(j, j) - sum_diag

        if abs(u_jj) < eps(T) * 100
            error("Zero diagonal at position ($j,$j)")
        end
        set_U!(j, j, u_jj)

        # Step 2: Compute column j of L (below diagonal)
        for idx in L.colptr[j]:(L.colptr[j+1]-1)
            i = L.rowval[idx]
            if i > j
                sum_L = T(0)
                for k = 1:j-1
                    sum_L += get_L(i, k) * get_U(k, j)
                end
                set_L!(i, j, (get_A(i, j) - sum_L) / u_jj)
            elseif i == j
                set_L!(i, j, T(1))  # Unit diagonal
            end
        end

        # Step 3: Compute row j of U (above diagonal)
        for col = j+1:n
            # Check if U[j,col] exists in pattern
            for idx in U.colptr[col]:(U.colptr[col+1]-1)
                if U.rowval[idx] == j
                    sum_U = T(0)
                    for k = 1:j-1
                        sum_U += get_L(j, k) * get_U(k, col)
                    end
                    set_U!(j, col, get_A(j, col) - sum_U)
                    break
                end
            end
        end
    end

    return nothing
end