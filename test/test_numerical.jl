using Test
using ILUK
using SparseArrays
using LinearAlgebra

@testset "Numerical ILU(k)" begin

    @testset "ILU(0) exact for triangular" begin
        # For a triangular matrix, ILU(0) should be exact
        n = 5
        L_exact = LowerTriangular(rand(n, n))
        L_exact[diagind(L_exact)] .= 1.0  # Unit diagonal
        U_exact = UpperTriangular(rand(n, n))
        A = sparse(L_exact * U_exact)

        L, U = ilu_k(A, 0)

        # Check factorization accuracy
        @test norm(L * U - A, Inf) < 1e-12
    end

    @testset "ILU(0) tridiagonal" begin
        # Tridiagonal test case
        n = 10
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))

        L, U = ilu_k(A, 0)

        # Check structure
        @test nnz(L) == 2*n - 1  # Diagonal + lower diagonal
        @test nnz(U) == 2*n - 1  # Diagonal + upper diagonal

        # Check unit diagonal of L
        for j = 1:n
            for idx in L.colptr[j]:(L.colptr[j+1]-1)
                if L.rowval[idx] == j
                    @test L.nzval[idx] == 1.0
                end
            end
        end

        # For ILU(0) of tridiagonal, check residual
        residual = norm(L * U - A, Inf)
        @test residual < 1e-12
    end

    @testset "ILU(1) five-point matrix" begin
        # Five-point stencil matrix (2D Laplacian)
        n = 5
        I = [1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,   5, 5, 5]
        J = [1, 2, 5,   1, 2, 3,   2, 3, 4,   3, 4, 5,   1, 4, 5]
        V = [4.0, -1.0, -1.0,   -1.0, 4.0, -1.0,   -1.0, 4.0, -1.0,   -1.0, 4.0, -1.0,   -1.0, -1.0, 4.0]
        A = sparse(I, J, V, n, n)

        L, U = ilu_k(A, 1)

        # Check that factorization is reasonable
        residual = norm(L * U - A, Inf)
        @test residual < 10.0  # Incomplete factorization, so not exact

        # Note: Fill positions exist in symbolic pattern but may be
        # numerically zero due to cancellation

        # Verify L has unit diagonal
        for j = 1:n
            @test L[j,j] == 1.0
        end
    end

    @testset "Diagonal shift for stability" begin
        # Create a near-singular matrix
        n = 5
        A = spdiagm(0 => [1e-10, 1.0, 1.0, 1.0, 1.0], -1 => ones(n-1), 1 => ones(n-1))

        # Without shift, should have issues
        L_bad, U_bad = ilu_k(A, 0, shift=0.0)

        # With shift, should be more stable
        L_good, U_good = ilu_k(A, 0, shift=1e-8)

        # The shifted version should have better conditioned U
        @test abs(U_good[1,1]) > abs(U_bad[1,1])
    end

    @testset "Consistency with symbolic" begin
        # Verify that numerical ILU preserves the symbolic pattern
        n = 6
        A = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => 4*ones(n),
                    1 => -2*ones(n-1), 2 => ones(n-2))

        L_symb, U_symb = symbolic_ilu_k(A, 1)
        L_num, U_num = ilu_k(A, 1)

        # Same sparsity pattern
        @test nnz(L_symb) == nnz(L_num)
        @test nnz(U_symb) == nnz(U_num)
        @test L_symb.colptr == L_num.colptr
        @test L_symb.rowval == L_num.rowval
        @test U_symb.colptr == U_num.colptr
        @test U_symb.rowval == U_num.rowval
    end
end