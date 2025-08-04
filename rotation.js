// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(A, K) {
 const N = A.length;
    if (N === 0) return A; // Empty array remains the same

    K = K % N;
    if (K === 0) return A; // No rotation needed

    return A.slice(-K).concat(A.slice(0, N - K));
}
