function solution(A) {
    let totalSum = 0;
    for (let num of A) {
        totalSum += num;
    }
    
    let minDiff = Infinity;
    let leftSum = 0;
    
    // Loop until the second last element is reached
    for (let i = 0; i < A.length - 1; i++) {
        leftSum += A[i];
        let rightSum = totalSum - leftSum;
        let diff = Math.abs(leftSum - rightSum);
        
        if (diff < minDiff) {
            minDiff = diff;
        }
    }
    
    return minDiff;
}