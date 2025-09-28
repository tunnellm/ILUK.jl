# Deprecated Files

This folder contains old implementations and experimental versions that are no longer used in the main ILUK module.

## File Categories

### Old Numerical Implementations
- `numerical_iluk_*.jl` - Various experimental numerical factorization approaches
  - `numerical_iluk_adaptive.jl` - Early adaptive shifting experiments
  - `numerical_iluk_clean.jl` - Cleanup attempt
  - `numerical_iluk_efficient.jl` - Performance optimization attempt
  - `numerical_iluk_fixed.jl` - Bug fix attempt
  - `numerical_iluk_proper.jl` - Correctness fix attempt
  - `numerical_iluk_restart.jl` - Restart strategy experiments
  - `numerical_iluk_schur.jl` - Schur complement approach
  - `numerical_iluk_simple.jl` - Simplified version
  - `numerical_iluk_sparse.jl` - Sparse-specific optimizations
  - `numerical_iluk_transpose.jl` - Transpose-based approach
  - `numerical_iluk_v2.jl` - Version 2 attempt

### Old Symmetric Implementations
- `symbolic_iluk_symmetric_correct.jl` - Early symmetric symbolic attempt
- `symmetric_iluk_adaptive.jl` - Early symmetric adaptive approach

### Utility Files
- `lu_shift_update.jl` - Shift update utilities

## Current Active Files (in main src/)
- `ILUK.jl` - Main module
- `symbolic_iluk.jl` - General symbolic factorization
- `fill_symbolic.jl` - Initialize L,U with A values
- `ilu_k_basic.jl` - Standard Crout-based ILU(k)
- `ilu_k_adaptive.jl` - Adaptive shifting (LimitedLDL-inspired)
- `ldlt_k_symbolic.jl` - Specialized symmetric symbolic patterns
- `ldlt_k_basic.jl` - Standard LDL^T factorization
- `ilu_k.jl` - Main interface functions

## History
These files represent the development history of the ILUK package, including:
- Multiple approaches to numerical stability
- Various performance optimization attempts
- Different algorithmic strategies
- Bug fixes and correctness improvements

All functionality has been consolidated into the clean, minimal API with just 4 exported functions.