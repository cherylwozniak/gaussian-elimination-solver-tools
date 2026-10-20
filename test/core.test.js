import { test } from 'node:test';
import assert from 'node:assert/strict';
import { solveLinearSystem, computeRank, solveWithSteps } from '../src/index.js';

/**
 * Helper: compare two arrays of floats with relative tolerance.
 * Asserts element-wise that |a[i]-b[i]| <= tol * max(1, |b[i]|).
 */
function approxEqual(a, b, tol = 1e-8) {
  assert.equal(a.length, b.length, 'lengths differ');
  for (let i = 0; i < a.length; i++) {
    const expected = b[i];
    const diff = Math.abs(a[i] - expected);
    const allowed = tol * Math.max(1, Math.abs(expected));
    assert.ok(diff <= allowed, `index ${i}: ${a[i]} != ${expected} within tolerance (diff=${diff})`);
  }
}

test('solveLinearSystem: 3x3 identity', () => {
  const A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const b = [5, 7, 9];
  const x = solveLinearSystem(A, b);
  approxEqual(x, [5, 7, 9]);
});

test('solveLinearSystem: general 2x2', () => {
  const A = [[2, 1], [1, -3]];
  const b = [5, -5];
  // [2 1; 1 -3] x = [5; -5]  =>  x = [10/7, 15/7]
  const x = solveLinearSystem(A, b);
  approxEqual(x, [10/7, 15/7]);
});

test('solveLinearSystem: 3x3 with nontrivial solution', () => {
  const A = [[1, 1, 1], [0, 2, 5], [2, 5, -1]];
  const b = [6, -4, 27];
  // Solution is x0 = 5, x1 = 3, x2 = -2 (classic textbook example)
  const x = solveLinearSystem(A, b);
  approxEqual(x, [5, 3, -2]);
});

test('solveLinearSystem: requires partial pivoting', () => {
  const A = [[0, 2, 3], [1, 1, 1], [2, 1, -1]];
  const b = [8, 6, 1];
  const x = solveLinearSystem(A, b);
  // Verify by substitution rather than asserting a fixed expected vector.
  const residual = [];
  for (let i = 0; i < 3; i++) {
    let s = 0;
    for (let j = 0; j < 3; j++) s += A[i][j] * x[j];
    residual.push(s);
  }
  approxEqual(residual, b);
});

test('solveLinearSystem: singular 2x2 returns null', () => {
  // Two proportional rows => rank 1, no unique solution.
  const A = [[1, 2], [2, 4]];
  const b = [3, 6];
  const x = solveLinearSystem(A, b);
  assert.equal(x, null);
});

test('solveLinearSystem: does not mutate input', () => {
  const A = [[2, 1], [1, -3]];
  const b = [5, -5];
  const A_copy = A.map(r => r.slice());
  const b_copy = b.slice();
  solveLinearSystem(A, b);
  assert.deepEqual(A, A_copy);
  assert.deepEqual(b, b_copy);
});

test('solveLinearSystem: throws on non-square A', () => {
  assert.throws(() => solveLinearSystem([[1, 2, 3], [4, 5, 6]], [1, 2]), /square/);
});

test('solveLinearSystem: throws when b length mismatch', () => {
  assert.throws(() => solveLinearSystem([[1, 0], [0, 1]], [1, 2, 3]), /length/);
});

test('computeRank: full-rank 3x3', () => {
  assert.equal(computeRank([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 3);
});

test('computeRank: rank-deficient 3x3', () => {
  // Row 3 is row 1 + row 2, so rank = 2.
  assert.equal(computeRank([[1, 2, 3], [4, 5, 6], [5, 7, 9]]), 2);
});

test('computeRank: rectangular tall matrix full column rank', () => {
  assert.equal(computeRank([[1, 0], [0, 1], [1, 1]]), 2);
});

test('computeRank: rectangular wide matrix', () => {
  // [1 0 0; 0 1 0] has rank 2 (two pivot columns).
  assert.equal(computeRank([[1, 0, 0], [0, 1, 0]]), 2);
});

test('computeRank: all-zero matrix', () => {
  assert.equal(computeRank([[0, 0], [0, 0]]), 0);
});

test('solveWithSteps: square full-rank returns solution and rank', () => {
  const A = [[1, 1, 1], [0, 2, 5], [2, 5, -1]];
  const b = [6, -4, 27];
  const result = solveWithSteps(A, b);
  assert.equal(result.rank, 3);
  approxEqual(result.solution, [5, 3, -2]);
});

test('solveWithSteps: singular square returns null solution but correct rank', () => {
  const A = [[1, 2], [2, 4]];
  const b = [3, 6];
  const result = solveWithSteps(A, b);
  assert.equal(result.rank, 1);
  assert.equal(result.solution, null);
});
