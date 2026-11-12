# gaussian-elimination-solver-tools

Solves square systems of linear equations and computes matrix ranks using Gaussian elimination with partial pivoting. Zero runtime dependencies, ESM-only, works in Node 18+.

## Usage

```js
import { solveLinearSystem, computeRank, solveWithSteps } from 'gaussian-elimination-solver-tools';

// Solve Ax = b for x. Returns null when A is singular.
const A = [[2, 1], [1, -3]];
const b = [5, -5];
const x = solveLinearSystem(A, b); // [2, 1]

// Compute the rank of any rectangular matrix.
const rank = computeRank([[1, 0], [0, 1], [1, 1]]); // 2

// Get both at once: solution (or null) plus the rank of A.
const { solution, rank: r } = solveWithSteps(A, b);
```

## Why this exists

Numerical linear algebra libraries tend to be large and general-purpose. This library is small and focused: it does one thing — Gaussian elimination with partial pivoting — and does it in pure JavaScript with no dependencies. The trade-off is that it is not optimized for huge dense matrices; it uses straightforward row-major `number[][]` arrays and does not vectorize or use typed arrays. For systems up to a few hundred equations it is fine; for large-scale work use a BLAS-backed library.

`solveLinearSystem` accepts only square coefficient matrices and returns `null` for singular systems rather than throwing. Non-square systems are handled by `solveWithSteps`, which reports the rank and returns a `null` solution when no unique solution exists. This library does not attempt least-squares solutions for over-determined systems — if that is what you need, use a different tool.

## The awkward edge

Floating-point. Near-singular matrices where the pivot is small-but-nonzero will produce solutions with large relative error. We treat a pivot as zero when its magnitude is below `max_abs_entry * 1e-10`, which works for most inputs but is a heuristic, not a guarantee. If your matrix has entries spanning many orders of magnitude, rescale before solving. Comparing two solutions for equality with `===` will fail; use a tolerance.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

## Limitations

Values are coerced to floats, so very large integers lose precision. If you need
exact integer aggregates over a window, this is the wrong tool.

## Contributing

Issues and pull requests are welcome. Please keep the dependency list empty —
that constraint is the point of the project, not an oversight.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

