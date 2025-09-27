using Test
using ILUK
using SparseArrays

@testset "ILUK.jl" begin

    @testset "Basic ILU(0)" begin
        # Simple 5x5 test matrix
        n = 5
        I = [1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,   5, 5, 5]
        J = [1, 2, 5,   1, 2, 3,   2, 3, 4,   3, 4, 5,   1, 4, 5]
        V = [2.0, 1.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 1.0, 2.0]
        A = sparse(I, J, V, n, n)

        L, U = symbolic_ilu_k(A, 0)

        @test nnz(L) == 10
        @test nnz(U) == 10
        @test nnz(L) - n + nnz(U) == 15  # PETSc-style count
    end

    @testset "ILU(1) fill-in" begin
        # Same test matrix
        n = 5
        I = [1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,   5, 5, 5]
        J = [1, 2, 5,   1, 2, 3,   2, 3, 4,   3, 4, 5,   1, 4, 5]
        V = [2.0, 1.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 1.0, 2.0]
        A = sparse(I, J, V, n, n)

        L, U = symbolic_ilu_k(A, 1)

        @test nnz(L) == 11
        @test nnz(U) == 11
        @test U[2,5] != 0  # Expected fill position
    end

    @testset "Tridiagonal matrix" begin
        n = 8
        # Create tridiagonal matrix
        diag = 2.0 * ones(n)
        off = -1.0 * ones(n-1)
        A = spdiagm(-1 => off, 0 => diag, 1 => off)

        L0, U0 = symbolic_ilu_k(A, 0)
        @test nnz(L0) - n + nnz(U0) == nnz(A)

        L1, U1 = symbolic_ilu_k(A, 1)
        @test nnz(L1) - n + nnz(U1) > nnz(A)  # Should have fill
    end

end