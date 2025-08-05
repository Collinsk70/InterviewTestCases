// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(A) {
let unpaired = 0;
    for (let i = 0; i < A.length; i++) {
        unpaired ^= A[i]; // XOR each element
    }
    return unpaired;
    }
