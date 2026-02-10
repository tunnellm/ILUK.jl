using Test
using ILUK
using SparseArrays
using LinearAlgebra

@testset "ILUK.jl" begin

    @testset "Symbolic ILU(k)" begin
        # Simple 5x5 test matrix
        n = 5
        I = [1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,   5, 5, 5]
        J = [1, 2, 5,   1, 2, 3,   2, 3, 4,   3, 4, 5,   1, 4, 5]
        V = [2.0, 1.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 2.0, 1.0,   1.0, 1.0, 2.0]
        A = sparse(I, J, V, n, n)

        @testset "ILU(0)" begin
            L, U = symbolic_ilu_k(A, 0)
            # L and U are strictly triangular (no diagonal)
            @test nnz(L) == 5   # Off-diagonal lower entries
            @test nnz(U) == 5   # Off-diagonal upper entries
        end

        @testset "ILU(1)" begin
            L, U = symbolic_ilu_k(A, 1)
            # ILU(1) has fill-in
            @test nnz(L) >= 5
            @test nnz(U) >= 5
            # Expected fill at (2,5) and (5,2)
            @test U[2,5] != 0 || L[5,2] != 0
        end
    end

    @testset "Symbolic Cholesky" begin
        n = 8
        # Symmetric tridiagonal matrix
        diag = 2.0 * ones(n)
        off = -1.0 * ones(n-1)
        A = spdiagm(-1 => off, 0 => diag, 1 => off)

        @testset "Level 0" begin
            L, graph_ops = symbolic_cholesky(A, 0)
            @test nnz(L) == n - 1  # Strictly lower: one subdiagonal
        end

        @testset "Level 1" begin
            L, graph_ops = symbolic_cholesky(A, 1)
            @test nnz(L) >= n - 1
        end
    end

    @testset "Numerical ILU(k)" begin
        n = 10
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))

        @testset "LDU factorization struct" begin
            F = ilu_k(A, 0)
            @test F isa LDUFactorization
            @test F.success == true
            @test size(F) == (n, n)
            @test length(F.D) == n
        end

        @testset "Solve accuracy" begin
            F = ilu_k(A, 0)
            b = ones(n)
            x = F \ b

            # Check that preconditioner solve is accurate
            # (I+L) * D * (I+U) * x ≈ b
            y = (I + F.U) * x
            z = Diagonal(F.D) * y
            w = (I + F.L) * z
            @test norm(w - b) / norm(b) < 1e-12
        end
    end

    @testset "Numerical LDLT(k)" begin
        n = 10
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))
        A = sparse(Symmetric(A))

        @testset "LDL factorization struct" begin
            F = ldlt_k(A, 0)
            @test F isa LDLFactorization
            @test F.success == true
            @test size(F) == (n, n)
            @test length(F.D) == n
        end

        @testset "Solve accuracy" begin
            F = ldlt_k(A, 0)
            b = ones(n)
            x = F \ b

            # Check that preconditioner solve is accurate
            # (I+L) * D * (I+L)' * x ≈ b
            y = (I + F.L)' * x
            z = Diagonal(F.D) * y
            w = (I + F.L) * z
            @test norm(w - b) / norm(b) < 1e-12
        end
    end

    @testset "Adaptive shifting" begin
        n = 5
        # Near-singular matrix
        A = spdiagm(0 => [1e-12, 1.0, 1.0, 1.0, 1.0], -1 => ones(n-1), 1 => ones(n-1))

        F = ilu_k(A, 0, min_pivot=1e-8, max_attempts=3)
        @test F.success == true
        @test F.shift > 0  # Should have applied shift
    end

end
