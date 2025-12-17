using Test
using ILUK
using SparseArrays
using LinearAlgebra

@testset "Numerical ILU(k) extended" begin

    @testset "Tridiagonal exactness" begin
        # For tridiagonal, ILU(0) is exact
        n = 10
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))

        F = ilu_k(A, 0)

        # Reconstruct: (I+L) * D * (I+U)
        LU = (I + F.L) * Diagonal(F.D) * (I + F.U)
        @test norm(LU - A, Inf) < 1e-12
    end

    @testset "Pattern exactness" begin
        # ILU(k) should be exact on its symbolic pattern
        n = 6
        A = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => 4*ones(n),
                    1 => -2*ones(n-1), 2 => ones(n-2))

        for k in 0:2
            L_symb, U_symb = symbolic_ilu_k(A, k)
            F = ilu_k(A, k)

            # Same sparsity pattern
            @test F.L.colptr == L_symb.colptr
            @test F.L.rowval == L_symb.rowval
            @test F.U.colptr == U_symb.colptr
            @test F.U.rowval == U_symb.rowval

            # Compute masked residual
            LU = (I + F.L) * Diagonal(F.D) * (I + F.U)
            residual = LU - A

            # Mask: only check positions in L ∪ U ∪ diagonal
            S = L_symb + U_symb
            for j in 1:n
                S[j, j] = 1.0
            end
            masked = residual .* (S .!= 0)
            @test norm(masked, Inf) < 1e-10
        end
    end

    @testset "Symmetric LDL pattern exactness" begin
        n = 8
        A = spdiagm(-2 => ones(n-2), -1 => -2*ones(n-1), 0 => 4*ones(n),
                    1 => -2*ones(n-1), 2 => ones(n-2))
        A = sparse(Symmetric(A))

        for k in 0:2
            L_symb = symbolic_cholesky(A, k)
            F = ldlt_k(A, k)

            # Same sparsity pattern
            @test F.L.colptr == L_symb.colptr
            @test F.L.rowval == L_symb.rowval

            # Compute masked residual
            LDLt = (I + F.L) * Diagonal(F.D) * (I + F.L)'
            residual = LDLt - A

            # Mask: only check positions in L ∪ L' ∪ diagonal
            S = L_symb + L_symb'
            for j in 1:n
                S[j, j] = 1.0
            end
            masked = residual .* (S .!= 0)
            @test norm(masked, Inf) < 1e-10
        end
    end

    @testset "Solve zero-allocation" begin
        n = 100
        A = spdiagm(-1 => -ones(n-1), 0 => 4*ones(n), 1 => -ones(n-1))

        F = ilu_k(A, 0)
        b = rand(n)
        x = similar(b)

        # First call for compilation
        ldl_solve!(x, F.L, F.D, b)

        # Second call should be allocation-free
        allocs = @allocated ldl_solve!(x, F.L, F.D, b)
        @test allocs == 0

        # Same for LDU solve
        ldu_solve!(x, F.L, F.D, F.U, b)
        allocs = @allocated ldu_solve!(x, F.L, F.D, F.U, b)
        @test allocs == 0
    end

    @testset "High fill levels" begin
        # Test that high k values work without overflow
        n = 50
        # Dense-ish matrix
        A = sprand(n, n, 0.2)
        A = A + A' + 10I
        A = sparse(A)

        for k in [0, 1, 5, 10]
            F = ilu_k(A, k)
            @test F.success == true
            @test all(isfinite.(F.D))
        end
    end

end
