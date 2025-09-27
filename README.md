# ILUK.jl

Symbolic ILU(k) factorization for sparse matrices in Julia.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tunnellm/ILUK.git")
```

## Usage

```julia
using ILUK
using SparseArrays

# Create a sparse matrix
A = sparse(...)

# Compute symbolic ILU(k) factorization
L, U = symbolic_ilu_k(A, k)
```

## Algorithm

Based on the Graph Search Algorithm from:
- Hysom, D., & Pothen, A. (1999). Efficient Parallel Computation of ILU(k) Preconditioners. *SC99*.

## License

MIT