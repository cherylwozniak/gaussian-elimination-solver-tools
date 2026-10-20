/**
 * Gaussian elimination with partial pivoting.
 *
 * All matrices are stored in row-major order as arrays of rows (number[][]).
 * Partial pivoting means at each elimination step we swap rows so the largest
 * absolute-value entry in the current column is the pivot. This reduces
 * floating-point error accumulation, which would otherwise be catastrophic
 * when the leading entry is near zero.
 */

/**
 * Tolerance for treating a near-zero pivot as an actual zero.
 *
 * We define ZERO_TOLERANCE relative to the matrix scale, because a fixed
 * absolute epsilon like 1e-12 is wrong for matrices with very small or very
 * large entries. We measure the largest-magnitude entry in the matrix and
 * scale the tolerance by it, so "effectively zero" is context-dependent.
 */
function computeZeroTolerance(matrix) {
  let maxAbs = 0;
  for (const row of matrix) {
    for (const v of row) {
      const a = Math.abs(v);
      if (a > maxAbs) {
        maxAbs = a;
      }
    }
  }
  if (maxAbs === 0) {
    return 1e-12;
  }
  return maxAbs * 1e-10;
}

/**
 * Deep-copy a row-major matrix. Used so callers' input is never mutated.
 */
function cloneMatrix(matrix) {
  const result = new Array(matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    result[i] = matrix[i].slice();
  }
  return result;
}

/**
 * Solve a system of linear equations Ax = b.
 *
 * @param {number[][]} A - n×n coefficient matrix (square).
 * @param {number[]} b - right-hand side vector of length n.
 * @returns {number[]|null} solution vector x, or null if the system is
 *   singular (no unique solution exists).
 * @throws {Error} if A is not square or b has incompatible length.
 */
export function solveLinearSystem(A, b) {
  if (!Array.isArray(A) || A.length === 0) {
    throw new Error('Coefficient matrix A must be a non-empty array of rows.');
  }
  const n = A.length;
  for (const row of A) {
    if (!Array.isArray(row) || row.length !== n) {
      throw new Error('Coefficient matrix A must be square (n rows of length n).');
    }
  }
  if (!Array.isArray(b) || b.length !== n) {
    throw new Error('Right-hand side vector b must have length equal to A.');
  }

  // Augmented matrix [A | b], copied so we never mutate the caller's data.
  const aug = new Array(n);
  for (let i = 0; i < n; i++) {
    aug[i] = A[i].slice();
    aug[i].push(b[i]);
  }

  const tol = computeZeroTolerance(A);

  // Forward elimination with partial pivoting.
  for (let col = 0; col < n; col++) {
    // Find the row (at or below col) with the largest absolute value in this column.
    let pivotRow = col;
    let bestAbs = Math.abs(aug[col][col]);
    for (let r = col + 1; r < n; r++) {
      const candidate = Math.abs(aug[r][col]);
      if (candidate > bestAbs) {
        bestAbs = candidate;
        pivotRow = r;
      }
    }

    if (bestAbs < tol) {
      // The pivot is effectively zero, meaning the matrix is singular.
      return null;
    }

    if (pivotRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }

    const pivot = aug[col][col];
    for (let r = col + 1; r < n; r++) {
      const factor = aug[r][col] / pivot;
      if (factor === 0) {
        continue;
      }
      for (let c = col; c <= n; c++) {
        aug[r][c] -= factor * aug[col][c];
      }
    }
  }

  // Back-substitution.
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i][j] * x[j];
    }
    x[i] = sum / aug[i][i];
  }
  return x;
}

/**
 * Compute the rank of a matrix (the number of linearly independent rows).
 *
 * @param {number[][]} matrix - m×n matrix in row-major order.
 * @returns {number} the rank.
 * @throws {Error} if matrix is not a non-empty array of rows with equal length.
 */
export function computeRank(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new Error('Matrix must be a non-empty array of rows.');
  }
  const rows = matrix.length;
  const cols = matrix[0].length;
  for (const row of matrix) {
    if (!Array.isArray(row) || row.length !== cols) {
      throw new Error('All matrix rows must have the same length.');
    }
  }

  const work = cloneMatrix(matrix);
  const tol = computeZeroTolerance(matrix);
  let rank = 0;
  let row = 0;

  // Column-by-column elimination; skip columns with no eligible pivot.
  for (let col = 0; col < cols && row < rows; col++) {
    let pivotRow = row;
    let bestAbs = Math.abs(work[row][col]);
    for (let r = row + 1; r < rows; r++) {
      const candidate = Math.abs(work[r][col]);
      if (candidate > bestAbs) {
        bestAbs = candidate;
        pivotRow = r;
      }
    }

    if (bestAbs < tol) {
      continue; // no pivot in this column
    }

    if (pivotRow !== row) {
      const tmp = work[row];
      work[row] = work[pivotRow];
      work[pivotRow] = tmp;
    }

    const pivot = work[row][col];
    for (let r = row + 1; r < rows; r++) {
      const factor = work[r][col] / pivot;
      if (factor === 0) {
        continue;
      }
      for (let c = col; c < cols; c++) {
        work[r][c] -= factor * work[row][c];
      }
    }
    rank++;
    row++;
  }

  return rank;
}

/**
 * Solve Ax = b, returning the solution AND the rank of A. This lets callers
 * detect under- or over-determined systems in one call instead of two.
 *
 * @param {number[][]} A - m×n coefficient matrix.
 * @param {number[]} b - right-hand side, length m.
 * @returns {{solution: number[]|null, rank: number}}
 *   `solution` is null when A is not square or the system is singular.
 *   `rank` is always computed.
 * @throws {Error} if A is empty, ragged, or incompatible with b.
 */
export function solveWithSteps(A, b) {
  if (!Array.isArray(A) || A.length === 0) {
    throw new Error('Coefficient matrix A must be a non-empty array of rows.');
  }
  const m = A.length;
  for (const row of A) {
    if (!Array.isArray(row) || row.length !== A[0].length) {
      throw new Error('Coefficient matrix A must be rectangular.');
    }
  }
  if (!Array.isArray(b) || b.length !== m) {
    throw new Error('Right-hand side vector b must have length equal to A\'s row count.');
  }

  const rank = computeRank(A);

  // We only return a unique solution for square, full-rank systems.
  let solution = null;
  const cols = A[0].length;
  if (m === cols && rank === cols) {
    solution = solveLinearSystem(A, b);
  }
  return { solution, rank };
}
