// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(A) {
    // Implement your solution here
    const N = A.length;
    const expectedSum = (N + 1) * (N + 2) / 2;
    const actualSum = A.reduce((acc, num) => acc + num, 0);
    return expectedSum - actualSum;

}
