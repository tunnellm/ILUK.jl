function fill_symbolic!(
    A::SparseMatrixCSC{T},
    Lc::SparseMatrixCSC{T},
    Uc::SparseMatrixCSC{T},
) where {T}

    n = size(A, 2)

    Lc.nzval .= T(0)
    Uc.nzval .= T(0)

    @inbounds for j = 1:n
        pa = A.colptr[j]
        pa_end = A.colptr[j+1]
        pl = Lc.colptr[j]
        pl_end = Lc.colptr[j+1]
        pu = Uc.colptr[j]
        pu_end = Uc.colptr[j+1]

        # advance-until helpers ------------------------------------------------
        advance_l!(r) = begin
            while pl ≤ pl_end && (pl > pl_end || Lc.rowval[pl] < r)
                pl += 1
            end
        end
        advance_u!(r) = begin
            while pu ≤ pu_end && (pu > pu_end || Uc.rowval[pu] < r)
                pu += 1
            end
        end

        # ---------------------------------------------------------------------
        while pa < pa_end
            ra = A.rowval[pa]

            # upper / diag -----------------------------------------------------
            if ra ≤ j
                advance_u!(ra)
                if pu ≤ pu_end && Uc.rowval[pu] == ra
                    Uc.nzval[pu] = A.nzval[pa]
                end
            end

            # lower / diag -----------------------------------------------------
            if ra ≥ j
                advance_l!(ra)
                if pl ≤ pl_end && Lc.rowval[pl] == ra
                    Lc.nzval[pl] = A.nzval[pa]
                end
            end

            pa += 1
        end
    end
    return nothing
end

function transpose_keepzeros(A::SparseMatrixCSC{T}) where {T}
    m, n = size(A)
    nnzA = nnz(A)

    colptrT = Vector{Int}(undef, m + 1)
    rowvalT = Vector{Int}(undef, nnzA)
    nzvalT = similar(A.nzval)

    # pass 1 – count how many entries each output column will get
    fill!(colptrT, 0)
    @inbounds for r in A.rowval
        colptrT[r] += 1
    end

    # cumulative sum → true column pointers
    s = 1
    @inbounds for j = 1:m
        t = colptrT[j]
        colptrT[j] = s
        s += t
    end
    colptrT[m+1] = nnzA + 1

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

    return SparseMatrixCSC{T,Int}(n, m, colptrT, rowvalT, nzvalT)
end
