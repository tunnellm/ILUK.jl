"""
    Update LU Factorization with Diagonal Shift

Given: A = LU (already factored)
Want: (A + σI) = L̃Ũ (updated factorization)

Key insight: We can derive update formulas by analyzing the relationship
between the original and shifted factorizations.
"""

using LinearAlgebra
using SparseArrays

"""
    analyze_shift_update()

Mathematically derive what happens when we shift a factored matrix.
"""
function analyze_shift_update()
    println("="^60)
    println("Mathematical Analysis: Updating LU with Diagonal Shift")
    println("="^60)

    println("\nGiven: A = LU where L has unit diagonal")
    println("Want: (A + σI) = L̃Ũ")
    println()

    println("Approach 1: Sherman-Morrison-Woodbury Formula")
    println("-"^40)
    println("(A + σI) = LU + σI")
    println("        = L(U + σL⁻¹)")
    println("        = L(U + σL⁻¹)")
    println()
    println("If we factor: U + σL⁻¹ = L₂U₂")
    println("Then: (A + σI) = (LL₂)U₂")
    println()
    println("Problem: L⁻¹ is dense even if L is sparse!")

    println("\nApproach 2: Rank-1 Updates")
    println("-"^40)
    println("Write σI = Σᵢ σeᵢeᵢᵀ (sum of rank-1 updates)")
    println("Apply rank-1 update formulas sequentially")
    println("Problem: n rank-1 updates, each costs O(n²)")

    println("\nApproach 3: Direct Analysis (Crout-style)")
    println("-"^40)
    println("Let's analyze column by column what changes...")
    println()

    # Small example to show the pattern
    n = 4
    println("Example with 4×4 matrix:")
    println()
    println("Original factorization A = LU:")
    println("Column j: A[:,j] = L * U[:,j]")
    println()
    println("Shifted factorization (A+σI) = L̃Ũ:")
    println("Column j: (A+σI)[:,j] = L̃ * Ũ[:,j]")
    println("        : A[:,j] + σeⱼ = L̃ * Ũ[:,j]")
    println()
    println("Key observation: The shift only affects diagonal elements!")

    return nothing
end

"""
    lu_diagonal_shift_update!(L::Matrix, U::Matrix, shift::Real)

Update an existing LU factorization for a diagonal shift.
Given A = LU, compute factorization of (A + shift*I).

This is a direct algorithm that processes the factorization column by column.
"""
function lu_diagonal_shift_update!(L::Matrix{T}, U::Matrix{T}, shift::Real) where T
    n = size(L, 1)

    # We need to update U and possibly L to account for the shift
    # Key insight: (A + σI) = L(U + σL⁻¹) but L⁻¹ is dense

    # Alternative: Use the fact that for Crout factorization:
    # (A + σI)[i,j] = Σₖ L[i,k] * U[k,j]
    # The diagonal shift affects: (A + σI)[i,i] = A[i,i] + σ

    # For each column j, we need to update based on the shift
    for j = 1:n
        # The diagonal element U[j,j] gets the most complex update
        # Original: A[j,j] = Σₖ L[j,k] * U[k,j] with L[j,j] = 1
        # Shifted: (A+σ)[j,j] = Σₖ L̃[j,k] * Ũ[k,j]

        # Since L has unit diagonal, U's diagonal absorbs the shift
        # But this propagates to off-diagonal elements

        # This is getting complex - let's try a different approach
    end

    return nothing
end

"""
    lu_shift_by_refactorization(A::Matrix, L::Matrix, U::Matrix, shift::Real)

Naive approach: Apply shift and refactor from scratch.
This gives us a baseline to verify our optimized approaches.
"""
function lu_shift_by_refactorization(A::Matrix{T}, L::Matrix{T}, U::Matrix{T},
                                     shift::Real) where T
    n = size(A, 1)
    A_shifted = A + shift * I

    L_new = Matrix{T}(I, n, n)
    U_new = copy(A_shifted)

    # Standard LU factorization
    for k = 1:n-1
        for i = k+1:n
            L_new[i,k] = U_new[i,k] / U_new[k,k]
            for j = k:n
                U_new[i,j] -= L_new[i,k] * U_new[k,j]
            end
        end
    end

    # Zero out lower triangle of U
    for i = 2:n
        for j = 1:i-1
            U_new[i,j] = 0
        end
    end

    return L_new, U_new
end

"""
    analyze_factorization_difference(A::Matrix, shift::Real)

Compare factorizations of A and A+σI to understand the update pattern.
"""
function analyze_factorization_difference(A::Matrix{T}, shift::Real) where T
    n = size(A, 1)

    # Factor original
    L1 = Matrix{T}(I, n, n)
    U1 = copy(A)
    for k = 1:n-1
        for i = k+1:n
            L1[i,k] = U1[i,k] / U1[k,k]
            for j = k:n
                U1[i,j] -= L1[i,k] * U1[k,j]
            end
        end
    end

    # Factor shifted
    L2 = Matrix{T}(I, n, n)
    U2 = copy(A) + shift * I
    for k = 1:n-1
        for i = k+1:n
            L2[i,k] = U2[i,k] / U2[k,k]
            for j = k:n
                U2[i,j] -= L2[i,k] * U2[k,j]
            end
        end
    end

    # Zero out lower triangles
    for i = 2:n
        for j = 1:i-1
            U1[i,j] = 0
            U2[i,j] = 0
        end
    end

    # Analyze differences
    L_diff = L2 - L1
    U_diff = U2 - U1

    println("\n" * "="^60)
    println("Factorization Difference Analysis")
    println("="^60)
    println("\nOriginal: A = L₁U₁")
    println("Shifted:  (A+$(shift)I) = L₂U₂")
    println()
    println("||L₂ - L₁||_F = $(norm(L_diff))")
    println("||U₂ - U₁||_F = $(norm(U_diff))")
    println()

    # Show pattern of changes
    println("Pattern of changes in L:")
    for j = 1:n-1
        col_change = norm(L_diff[:,j])
        if col_change > 1e-10
            println("  Column $j: ||ΔL[:,j]|| = $col_change")
        end
    end

    println("\nPattern of changes in U:")
    for i = 1:n
        row_change = norm(U_diff[i,:])
        if row_change > 1e-10
            println("  Row $i: ||ΔU[i,:]|| = $row_change")
        end
    end

    # Check if there's a simple relationship
    println("\nDiagonal changes in U:")
    for i = 1:n
        println("  U₂[$i,$i] - U₁[$i,$i] = $(U2[i,i] - U1[i,i])")
    end

    return L1, U1, L2, U2, L_diff, U_diff
end

"""
    derive_update_formulas()

Derive the exact formulas for updating LU factorization with diagonal shift.
"""
function derive_update_formulas()
    println("\n" * "="^60)
    println("Deriving Update Formulas")
    println("="^60)

    println("\nConsider the Crout factorization process:")
    println("For each column j:")
    println("  1. U[j,j] = A[j,j] - Σ(k<j) L[j,k]*U[k,j]")
    println("  2. L[i,j] = (A[i,j] - Σ(k<j) L[i,k]*U[k,j])/U[j,j] for i>j")
    println("  3. U[j,i] = A[j,i] - Σ(k<j) L[j,k]*U[k,i] for i>j")

    println("\nWith shift σ, A[j,j] → A[j,j]+σ, so:")
    println("  1. Ũ[j,j] = (A[j,j]+σ) - Σ(k<j) L̃[j,k]*Ũ[k,j]")

    println("\nThe challenge: L̃ and Ũ for k<j depend on the shift too!")
    println("This creates a recursive dependency.")

    println("\nKey insight: The shift propagates through the factorization")
    println("in a specific pattern that we can track.")

    return nothing
end